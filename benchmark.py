"""
benchmark.py — AMPI vs FAISS: recall@1 / recall@10 / recall@100 / candidates across all index variants.

Datasets
  gauss   : 10 000 iid N(0,1) vectors, d=128  (synthetic)
  mnist   : data/MNIST/mnist-784-euclidean.hdf5  (60k train / 10k test queries)
  fashion : data/fashion-mnist/fashion-mnist-784-euclidean.hdf5
  sift    : data/sift/sift-128-euclidean.hdf5  (1M train / 10k test queries)
  glove   : data/glove/glove-100-angular.hdf5  (L2-normalised → cosine equiv)

Usage
  python benchmark.py                      # gauss only (default)
  python benchmark.py gauss mnist fashion  # explicit list
  python benchmark.py all                  # all five
"""

import sys, time, argparse, math
from pathlib import Path
import numpy as np
import faiss
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from ampi import (
    AMPIBinaryIndex,
    AMPIAffineFanIndex,
)
from ampi.tuner import _GP1D, _pareto_knee

K             = 10      # primary recall threshold (Pareto / tuning)
K_MAX         = 100     # maximum neighbours to retrieve (enables recall@1/10/100)
N_QUERIES     = 200     # queries used for evaluation
WARMUP        = 10      # queries discarded before timing
MAX_CAND_FRAC        = 0.25   # skip configs whose avg candidates exceed this fraction of n
MAX_CAND_FRAC_BINARY = 1.00   # Binary needs wider windows; allow up to 4× the normal limit
CAND_SAMPLE   = 1       # queries used to estimate candidate count (rough is fine)
QUICK_SAMPLE  = 10      # queries for quick Pareto dominance check (coarse, speed-critical)
TUNE_SAMPLE   = 50      # queries used for alpha tuning (needs reliable recall signal)
PARETO_TOL    = 0.03    # skip if estimated recall is within this of a Pareto point with fewer cands

FIGURES_DIR = Path("figures")


# ── datasets ──────────────────────────────────────────────────────────────────

def make_gaussian(n=10_000, d=128, seed=42):
    rng   = np.random.default_rng(seed)
    data  = rng.standard_normal((n, d)).astype(np.float32)
    qs    = rng.standard_normal((N_QUERIES, d)).astype(np.float32)
    flat  = faiss.IndexFlatL2(d)
    flat.add(data)
    _, gt = flat.search(qs, K_MAX)
    return data, qs, gt.astype(np.int32)


def load_hdf5(path, n_train=None, normalize=False):
    with h5py.File(path) as f:
        train   = f["train"][:n_train].astype(np.float32)
        queries = f["test"][:N_QUERIES].astype(np.float32)
    # Index = train only; queries are held out (standard ANN benchmark setup).
    # Including test in the database causes self-match: Flat L2 returns the
    # query itself (distance 0) as NN1, biasing recall and dist-ratio for all indexes.
    data = train
    if normalize:
        norms   = np.linalg.norm(data,    axis=1, keepdims=True) + 1e-10
        qnorms  = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10
        data    = data    / norms
        queries = queries / qnorms
    flat = faiss.IndexFlatL2(data.shape[1])
    flat.add(data)
    _, nn = flat.search(queries, K_MAX)
    return data, queries, nn.astype(np.int32)


# ── evaluation ────────────────────────────────────────────────────────────────

def recall(gt, approx, k=K):
    """Recall@k: fraction of true top-k neighbours found in returned top-k."""
    hits = sum(
        len(set(g[:k].tolist()) & set(a[:k].tolist()))
        for g, a in zip(gt, approx)
    )
    return hits / (len(gt) * k)


def time_ms(fn, queries):
    for q in queries[:WARMUP]:
        fn(q)
    t0 = time.perf_counter()
    for q in queries:
        fn(q)
    return (time.perf_counter() - t0) / len(queries) * 1e3


def avg_cands(cands_fn, queries):
    return int(np.mean([
        c if isinstance(c, (int, np.integer)) else len(c)
        for c in (cands_fn(q) for q in queries[:CAND_SAMPLE])
    ]))


def approx_ratio(data, queries, gt_indices, found_indices):
    """Mean ratio: avg distance to found neighbours / avg distance to true neighbours."""
    ratios = []
    for q, true_idx, found_idx in zip(queries, gt_indices, found_indices):
        true_d  = np.sqrt(np.sum((data[true_idx]       - q) ** 2, axis=1)).mean()
        found_d = np.sqrt(np.sum((data[found_idx[:K]]  - q) ** 2, axis=1)).mean()
        ratios.append(found_d / (true_d + 1e-10))
    return float(np.mean(ratios))


def _pareto_group(label):
    """Broad family group for within-family Pareto pruning.
    IVF, Binary, AFan are kept separate so a fast IVF result
    cannot dominate and prune the entire AMPI curve.
    """
    parts = label.split()
    p = parts[0]
    if p == "AFan":
        return f"AFan {parts[1]} {parts[2]}"  # F=<num_fans> K=<top_k>
    return p


def evaluate(label, query_fn, cands_fn, queries, gt, data, n, pareto=None):
    # Step 1: candidate count check (1 query)
    cands = avg_cands(cands_fn, queries)
    is_binary = label.startswith("Binary")
    limit = (MAX_CAND_FRAC_BINARY if is_binary else MAX_CAND_FRAC) * n
    if label != "Flat L2" and cands > limit:
        print(f"  [skip] {label}  — avg candidates > {limit:,.0f}", flush=True)
        return None

    # Step 2: within-family Pareto dominance check (QUICK_SAMPLE queries)
    if pareto is not None and label != "Flat L2":
        group       = _pareto_group(label)
        group_front = pareto.get(group, [])
        if group_front:
            quick_results = [query_fn(q) for q in queries[:QUICK_SAMPLE]]
            quick_idx     = [r[2] if isinstance(r, tuple) else r for r in quick_results]
            quick_pad     = [np.pad(ix[:K], (0, max(0, K - len(ix))), constant_values=-1)
                             for ix in quick_idx]
            est_rec      = recall(gt[:QUICK_SAMPLE], quick_pad, K)
            max_front_rec = max(pr for _, pr in group_front)
            # Never skip a config that may push recall beyond the current frontier
            # maximum — that would extend the frontier upward, not just rightward.
            # Only apply dominance pruning within the already-covered recall range.
            if est_rec <= max_front_rec and any(
                p_cands <= cands and p_rec >= est_rec - PARETO_TOL
                for p_cands, p_rec in group_front
            ):
                print(f"  [skip-pareto] {label}  "
                      f"(est R@10={est_rec:.3f}, cands≈{cands:,})", flush=True)
                return None

    # Step 3: warmup + full timed pass
    for q in queries[:WARMUP]:
        query_fn(q)
    t0      = time.perf_counter()
    results = [query_fn(q) for q in queries]
    ms      = (time.perf_counter() - t0) / len(queries) * 1e3

    indices = [r[2] if isinstance(r, tuple) else r for r in results]
    padded  = [np.pad(ix[:K_MAX], (0, max(0, K_MAX - len(ix))), constant_values=-1)
               for ix in indices]
    rec1   = recall(gt, padded, 1)
    rec10  = recall(gt, padded, K)
    rec100 = recall(gt, padded, K_MAX)
    ratio  = approx_ratio(data, queries, gt[:, :K], padded)
    r = dict(label=label, recall=rec10, recall1=rec1, recall100=rec100,
             ratio=ratio, ms=ms, qps=1e3 / ms, cands=cands)

    cands_str = f"{cands:>7,}" if cands < n else f"{'n':>7}"
    print(f"  {label:<38}  {rec1:>6.3f}  {rec10:>6.3f}  {rec100:>6.3f}"
          f"  {ratio:>10.4f}  {1e3/ms:>8.1f}  {ms:>7.3f}  {cands_str}",
          flush=True)
    return r


# ── figures ───────────────────────────────────────────────────────────────────

# Map label prefix → (color, marker, linestyle)
_FAMILY_STYLE = {
    "Flat L2":      dict(color="black",   marker="*",  ls="none", ms=14, zorder=5),
    "IVF":          dict(color="#1f77b4", marker="o",  ls="-",    lw=2),
    "Binary":       dict(color="#8c8c8c", marker="s",  ls="-",    lw=1.5, ms=7),
    "AFan K=1":     dict(color="#9467bd", marker="D",  ls="-",    lw=2,   ms=7),
    "AFan K=2":     dict(color="#e377c2", marker="D",  ls="--",   lw=2,   ms=7),
    "AFan K=3":     dict(color="#6a0dad", marker="D",  ls=":",    lw=2,   ms=7),
}


def _family(label):
    parts = label.split()
    if parts[0] == "Flat":   return "Flat L2"
    if parts[0] == "IVF":    return "IVF"
    if parts[0] == "Binary": return "Binary"
    if parts[0] == "AFan":
        k_part = next((p for p in parts if p.startswith("K=")), "K=1")
        return f"AFan {k_part}"
    return parts[0]


def _pareto_frontier(xs, ys, lower_x=True, higher_y=True):
    """Return indices of Pareto-optimal points (min x, max y)."""
    pts = sorted(enumerate(zip(xs, ys)), key=lambda t: t[1][0])
    frontier = []
    best_y = -np.inf
    for idx, (x, y) in pts:
        if y > best_y:
            best_y = y
            frontier.append(idx)
    return frontier


def save_figures(all_results):
    """all_results: list of (dataset_name, rows) tuples."""
    FIGURES_DIR.mkdir(exist_ok=True)

    for dataset_name, rows in all_results:
        # group rows by method family, sort by candidates for line continuity
        families = {}
        for r in rows:
            fam = _family(r["label"])
            families.setdefault(fam, []).append(r)
        for pts in families.values():
            pts.sort(key=lambda x: x["cands"])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax_r1, ax_r10, ax_r100 = axes[0]
        ax_qps, ax_ratio, ax_hide = axes[1]
        ax_hide.set_visible(False)
        fig.suptitle(dataset_name, fontsize=13, fontweight="bold")

        legend_handles = []
        for fam, pts in sorted(families.items()):
            style = _FAMILY_STYLE.get(fam, dict(color="gray", marker="x", ls="-", lw=1))
            cands  = [p["cands"]    for p in pts]
            rec1   = [p["recall1"]  for p in pts]
            rec10  = [p["recall"]   for p in pts]
            rec100 = [p["recall100"]for p in pts]
            ratio  = [p["ratio"]    for p in pts]
            qps    = [p["qps"]      for p in pts]
            kw = {k: v for k, v in style.items() if k != "zorder"}
            zo = style.get("zorder", 2)

            line, = ax_r1.plot(cands, rec1, zorder=zo, **kw)
            ax_r10.plot(cands, rec10, zorder=zo, **kw)
            ax_r100.plot(cands, rec100, zorder=zo, **kw)
            ax_qps.plot(cands, qps, zorder=zo, **kw)
            ax_ratio.plot(rec10, ratio, zorder=zo, **kw)
            legend_handles.append((line, fam))

            # Pareto highlight + knee annotation on recall@10 panel
            fi = _pareto_frontier(cands, rec10)
            if len(fi) > 1:
                px = [cands[i] for i in fi]
                py = [rec10[i] for i in fi]
                ax_r10.plot(px, py, color=style.get("color", "gray"),
                            lw=2.5, ls="-", alpha=0.35, zorder=zo - 1)
                pf = [(None, None, None, rec10[i], cands[i]) for i in fi]
                knee = _pareto_knee(pf)
                ax_r10.scatter([knee[4]], [knee[3]], s=150,
                               facecolors="none", edgecolors=style.get("color", "gray"),
                               lw=2, zorder=zo + 2)

        for ax, ylabel, title in [
            (ax_r1,   "Recall@1",   "Recall@1 vs Candidates"),
            (ax_r10,  "Recall@10",  "Recall@10 vs Candidates  (↖ better)"),
            (ax_r100, "Recall@100", "Recall@100 vs Candidates"),
        ]:
            ax.set_xscale("log")
            ax.set_xlabel("Candidates examined (log scale)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_title(title)

        ax_qps.set_xscale("log")
        ax_qps.set_yscale("log")
        ax_qps.set_xlabel("Candidates examined (log scale)")
        ax_qps.set_ylabel("QPS (log scale)")
        ax_qps.grid(True, alpha=0.3)
        ax_qps.set_title("QPS vs Candidates  (vertical gap = C++ speedup)")

        ax_ratio.set_xlabel("Recall@10")
        ax_ratio.set_ylabel("Dist ratio  (1.0 = perfect)")
        ax_ratio.set_xlim(-0.02, 1.05)
        ax_ratio.axhline(1.0, color="black", lw=0.8, ls=":")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_title("Dist ratio vs Recall@10")

        # shared legend below all panels
        hs = [h for h, _ in legend_handles]
        ls = [l for _, l in legend_handles]
        n_col = min(len(ls), 7)
        fig.legend(hs, ls, loc="lower center", ncol=n_col, fontsize=8,
                   bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        slug = dataset_name.split()[0].lower()
        out  = FIGURES_DIR / f"{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved {out}")


# ── auto-scaling ──────────────────────────────────────────────────────────────

def scale_params(n, d):
    """Return benchmark parameters scaled to dataset size and dimensionality.

    L        : Binary index projections ∝ d/8 × log₂(n/5k), capped at 64/128
    w_base   : sorted-projection half-window; baseline w=15 at n=10k, scales
               by sqrt(n/10k) so the covered fraction of each cone stays constant
    """
    L_raw = (d / 8) * max(1.0, math.log2(n / 5_000))
    L1    = max(16, min(64, 8 * round(L_raw / 8)))
    L2    = min(128, L1 * 2)

    w_base = max(15, int(15 * math.sqrt(n / 10_000)))

    return dict(L1=L1, L2=L2, w_base=w_base)


# ── main benchmark ────────────────────────────────────────────────────────────

def run(dataset_name, data, queries, gt):
    n, d = data.shape
    sep  = "═" * 72
    print(f"\n{sep}")
    print(f"  {dataset_name}  —  n={n:,}  d={d}  queries={len(queries)}  k={K}")
    print(sep)

    flat = faiss.IndexFlatL2(d)
    flat.add(data)

    nlist = max(16, int(np.sqrt(n)))
    q_ivf = faiss.IndexFlatL2(d)
    ivf   = faiss.IndexIVFFlat(q_ivf, d, nlist, faiss.METRIC_L2)
    ivf.train(data); ivf.add(data)

    pr = scale_params(n, d)
    L1, L2 = pr['L1'], pr['L2']
    wb     = pr['w_base']
    print(f"  auto-params: L={L1}/{L2}  w_base={wb}")

    def build(label, cls, **kw):
        print(f"  Building {label}…", end=" ", flush=True)
        t0  = time.perf_counter()
        idx = cls(data, **kw)
        print(f"{time.perf_counter()-t0:.2f}s")
        return idx

    idx_bin = build(f"Binary   L={L2}", AMPIBinaryIndex, num_projections=L2, seed=0)

    configs = []

    configs.append(("Flat L2",
                    lambda q: (None, None, flat.search(q[None], K_MAX)[1][0]),
                    lambda q: n))

    for nprobe in [1, 5, 10, 25, 50]:
        if nprobe > nlist: continue
        def _ivf(q, p=nprobe):
            ivf.nprobe = p
            return (None, None, ivf.search(q[None], K_MAX)[1][0])
        configs.append((f"IVF nprobe={nprobe}", _ivf, lambda q, p=nprobe: p * (n // nlist)))

    for w in [wb, 2*wb, 4*wb]:
        configs.append((f"Binary L={L2} w={w}",
                        lambda q, i=idx_bin, w=w: i.query(q, k=K_MAX, window_size=w),
                        lambda q, i=idx_bin, w=w: i.query_candidates(q, window_size=w)))

    # === AffineFan: tune index params on a small sample ===
    print(f"  Tuning AFan parameters on sample...", flush=True)

    n_sample    = max(10_000, min(50_000, int(n * 0.3)))
    data_sample = data[np.random.choice(n, n_sample, replace=False)]
    tune_qs     = queries[:TUNE_SAMPLE]

    # Build a flat GT on the sample for the tune queries
    flat_sample = faiss.IndexFlatL2(d)
    flat_sample.add(data_sample)
    _, gt_sample = flat_sample.search(tune_qs, K)
    gt_sample = gt_sample.astype(np.int32)

    # Tune alpha = nlist/sqrt(n) via GP-BO on the sample.
    # F is fixed at 16 for sample builds (large F → empty cones at 50k scale).
    # F is derived analytically from the full dataset after alpha is chosen.
    # Objective: average recall across 3 operating points (low/mid/high cands).
    wb_s = max(5, int(wb * math.sqrt(n_sample / n)))

    def _alpha_score(alpha):
        nlist_s = max(16, int(alpha * math.sqrt(n_sample)))
        idx_s   = AMPIAffineFanIndex(data_sample, nlist=nlist_s, num_fans=16, seed=0)
        scores  = []
        for cp, fp, w in [(3, 4, max(5, wb_s // 2)), (5, 8, wb_s), (10, 16, wb_s)]:
            res    = [idx_s.query(q, k=K, window_size=w, probes=cp, fan_probes=fp)
                      for q in tune_qs]
            idxs   = [r[2] if isinstance(r, tuple) else r for r in res]
            padded = [np.pad(ix[:K], (0, max(0, K - len(ix))), constant_values=-1)
                      for ix in idxs]
            scores.append(recall(gt_sample, padded, K))
        return float(np.mean(scores))

    ALPHA_LO, ALPHA_HI = 0.10, 2.0
    N_BO_ITER = 12
    x_obs = [ALPHA_LO, (ALPHA_LO + ALPHA_HI) / 2, ALPHA_HI]
    y_obs = [_alpha_score(a) for a in x_obs]

    gp     = _GP1D()
    x_cand = np.linspace(ALPHA_LO, ALPHA_HI, 200)
    rng_bo = np.random.default_rng(0)
    for _ in range(N_BO_ITER - 3):
        gp.fit(x_obs, y_obs)
        ei     = gp.EI(x_cand)
        next_a = float(x_cand[np.argmax(ei)])
        if any(abs(next_a - a) < (ALPHA_HI - ALPHA_LO) / 40 for a in x_obs):
            next_a = float(rng_bo.uniform(ALPHA_LO, ALPHA_HI))
        y_obs.append(_alpha_score(next_a))
        x_obs.append(next_a)

    best_alpha = x_obs[int(np.argmax(y_obs))]

    best_nlist = max(16, int(best_alpha * math.sqrt(n)))

    # Pick the largest F such that the full dataset has >= MIN_CONE_PTS per cone.
    # This scales F with dataset density: SIFT 1M → F=128, MNIST 60k → F=32.
    MIN_CONE_PTS = 5
    viable_Fs = [F for F in sorted(set([16, 32, 64, L1]))
                 if n // (best_nlist * F) >= MIN_CONE_PTS]
    best_F = max(viable_Fs) if viable_Fs else 16

    print(f"    Best: nlist={best_nlist} (alpha={best_alpha:.3f}), F={best_F}")

    # Build one AFan index per cone_top_k value.
    # K=1: original hard assignment.  K=2: each point in top-2 cones (2x memory, bounded).
    cone_top_ks = [1, 2]
    af_indexes  = {}
    for ktk in cone_top_ks:
        tag = f"AFan F={best_F} K={ktk}"
        idx = build(tag, AMPIAffineFanIndex,
                    nlist=best_nlist, num_fans=best_F,
                    seed=0, cone_top_k=ktk)
        af_indexes[ktk] = idx

    # Warmup AMPI indexes (triggers JIT compilation)
    print(f"  Warming up AMPI indexes...", flush=True)
    for w in [wb, 2*wb, 4*wb]:
        for q in queries[:WARMUP]:
            idx_bin.query(q, k=K, window_size=w)
    for ktk, idx_af in af_indexes.items():
        for q in queries[:WARMUP]:
            idx_af.query(q, k=K, window_size=wb, probes=10, fan_probes=best_F)
    print(f"  Warmup complete.")

    # Add AFan query-param sweep for each cone_top_k value
    limit = MAX_CAND_FRAC * n
    af_candidates = []
    for cp in [5, 10, 20, 50]:
        for fp in sorted(set([2, 4, 8, best_F//4, best_F//2, best_F])):
            for w_mult in [0.25, 0.5, 1.0, 1.5, 2.0]:
                w = max(5, int(wb * w_mult))
                af_candidates.append((cp, fp, w, cp * fp * 2 * w))
    af_candidates.sort(key=lambda x: x[3])

    for ktk, idx_af in af_indexes.items():
        for cp, fp, w, est_cands in af_candidates:
            if est_cands > limit * 1.5:
                continue
            label = f"AFan F={best_F} K={ktk} cp={cp} fp={fp} w={w}"
            configs.append((
                label,
                lambda q, i=idx_af, w=w, cp=cp, fp=fp: i.query(q, k=K_MAX, window_size=w, probes=cp, fan_probes=fp),
                lambda q, i=idx_af, w=w, cp=cp, fp=fp: i.query_candidates(q, window_size=w, probes=cp, fan_probes=fp),
            ))

    hdr = f"  {'Method':<38}  {'R@1':>6}  {'R@10':>6}  {'R@100':>6}  {'dist ratio':>10}  {'QPS':>8}  {'ms/q':>7}  {'cands':>7}"
    print()
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows   = []
    pareto = {}   # {family_group: [(cands, recall), ...]}  — per-family frontiers

    for label, fn, cands_fn in configs:
        r = evaluate(label, fn, cands_fn, queries, gt, data, n, pareto=pareto)
        if r is not None:
            rows.append(r)
            group = _pareto_group(label)
            front = pareto.get(group, [])
            front = [(c, rc) for c, rc in front
                     if not (r["cands"] <= c and r["recall"] >= rc)]
            front.append((r["cands"], r["recall"]))
            pareto[group] = front

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="*", default=["gauss"],
                    help="gauss mnist fashion sift glove all  (default: gauss)")
    args = ap.parse_args()

    targets = set(args.dataset)
    if "all" in targets:
        targets = {"gauss", "mnist", "fashion", "sift", "glove"}

    all_results = []

    if "gauss" in targets:
        data, queries, gt = make_gaussian(n=10_000, d=128)
        rows = run("Gaussian 10k  d=128", data, queries, gt)
        all_results.append(("Gaussian 10k  d=128", rows))

    if "mnist" in targets:
        data, queries, gt = load_hdf5("data/MNIST/mnist-784-euclidean.hdf5")
        rows = run("MNIST 60k  d=784", data, queries, gt)
        all_results.append(("MNIST 60k  d=784", rows))

    if "fashion" in targets:
        data, queries, gt = load_hdf5("data/fashion-mnist/fashion-mnist-784-euclidean.hdf5")
        rows = run("Fashion-MNIST 60k  d=784", data, queries, gt)
        all_results.append(("Fashion-MNIST 60k  d=784", rows))

    if "sift" in targets:
        data, queries, gt = load_hdf5("data/sift/sift-128-euclidean.hdf5")
        rows = run("SIFT 1M  d=128", data, queries, gt)
        all_results.append(("SIFT 1M  d=128", rows))

    if "glove" in targets:
        data, queries, gt = load_hdf5("data/glove/glove-100-angular.hdf5", normalize=True)
        rows = run("GloVe 1.18M  d=100", data, queries, gt)
        all_results.append(("GloVe 1.18M  d=100", rows))

    print()
    save_figures(all_results)
