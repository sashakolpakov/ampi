"""
benchmark.py — AMPI vs FAISS: recall@10 / candidates across all index variants.

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

K             = 10      # neighbours to retrieve
N_QUERIES     = 200     # queries used for evaluation
WARMUP        = 10      # queries discarded before timing
MAX_CAND_FRAC = 0.40    # skip configs whose avg candidates exceed this fraction of n …
MIN_CAND_ABS  = 100_000 # … but always allow at least this many candidates
CAND_SAMPLE   = 1       # queries used to estimate candidate count (rough is fine)
QUICK_SAMPLE  = 5       # queries for quick Pareto dominance check
PARETO_TOL    = 0.03    # skip if estimated recall is within this of a Pareto point with fewer cands

FIGURES_DIR = Path("figures")


# ── datasets ──────────────────────────────────────────────────────────────────

def make_gaussian(n=10_000, d=128, seed=42):
    rng   = np.random.default_rng(seed)
    data  = rng.standard_normal((n, d)).astype(np.float32)
    qs    = rng.standard_normal((N_QUERIES, d)).astype(np.float32)
    flat  = faiss.IndexFlatL2(d)
    flat.add(data)
    _, gt = flat.search(qs, K)
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
    _, nn = flat.search(queries, K)
    return data, queries, nn.astype(np.int32)


# ── evaluation ────────────────────────────────────────────────────────────────

def recall(gt, approx):
    hits = sum(
        len(set(g.tolist()) & set(a[:K].tolist()))
        for g, a in zip(gt, approx)
    )
    return hits / (len(gt) * K)


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
        return f"AFan {parts[1]}"  # F=<num_fans>
    return p


def evaluate(label, query_fn, cands_fn, queries, gt, data, n, pareto=None):
    # Step 1: candidate count check (1 query)
    cands = avg_cands(cands_fn, queries)
    limit = max(MAX_CAND_FRAC * n, MIN_CAND_ABS)
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
            est_rec = recall(gt[:QUICK_SAMPLE], quick_pad)
            if any(p_cands <= cands and p_rec >= est_rec - PARETO_TOL
                   for p_cands, p_rec in group_front):
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
    padded  = [np.pad(ix[:K], (0, max(0, K - len(ix))), constant_values=-1)
               for ix in indices]
    rec   = recall(gt, padded)
    ratio = approx_ratio(data, queries, gt, padded)
    r     = dict(label=label, recall=rec, ratio=ratio, ms=ms, qps=1e3 / ms, cands=cands)

    cands_str = f"{cands:>7,}" if cands < n else f"{'n':>7}"
    print(f"  {label:<38}  {rec:>6.3f}  {ratio:>10.4f}  {1e3/ms:>8.1f}  {ms:>7.3f}  {cands_str}",
          flush=True)
    return r


# ── figures ───────────────────────────────────────────────────────────────────

# Map label prefix → (color, marker, linestyle)
_FAMILY_STYLE = {
    "Flat L2":        dict(color="black",   marker="*",  ls="none", ms=14, zorder=5),
    "IVF":            dict(color="#1f77b4", marker="o",  ls="-",    lw=2),
    "Binary":         dict(color="#8c8c8c", marker="s",  ls="-",    lw=1.5, ms=7),
    "AFan F=16":      dict(color="#9467bd", marker="D",  ls="-",    lw=2,   ms=7),
    "AFan F=32":      dict(color="#6a0dad", marker="D",  ls="--",   lw=2,   ms=7),
}


def _family(label):
    parts = label.split()
    if parts[0] == "Flat":            return "Flat L2"
    if parts[0] == "IVF":             return "IVF"
    if parts[0] == "Binary":          return "Binary"
    if parts[0] == "AFan-vote":       return f"AFan-vote {parts[1]}"
    if parts[0] == "AFan":            return f"AFan {parts[1]}"
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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(dataset_name, fontsize=13, fontweight="bold")

        legend_handles = []
        for fam, pts in sorted(families.items()):
            style = _FAMILY_STYLE.get(fam, dict(color="gray", marker="x", ls="-", lw=1))
            cands = [p["cands"]  for p in pts]
            rec   = [p["recall"] for p in pts]
            ratio = [p["ratio"]  for p in pts]
            qps   = [p["qps"]    for p in pts]
            kw = {k: v for k, v in style.items() if k != "zorder"}
            zo = style.get("zorder", 2)

            line, = ax1.plot(cands, rec, zorder=zo, **kw)
            ax2.plot(cands, qps, zorder=zo, **kw)
            ax3.plot(rec, ratio, zorder=zo, **kw)
            legend_handles.append((line, fam))

            # highlight Pareto-optimal configs on panel 1
            fi = _pareto_frontier(cands, rec)
            if len(fi) > 1:
                px = [cands[i] for i in fi]
                py = [rec[i]   for i in fi]
                ax1.plot(px, py, color=style.get("color", "gray"),
                         lw=2.5, ls="-", alpha=0.35, zorder=zo - 1)

        # ── panel 1: Recall vs Candidates (algorithm quality, impl-agnostic) ──
        ax1.set_xscale("log")
        ax1.set_xlabel("Candidates examined (log scale)")
        ax1.set_ylabel("Recall@10")
        ax1.set_ylim(-0.02, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Recall@10 vs Candidates  (↖ better)")
        ax1.annotate("↖ better", xy=(0.04, 0.92), xycoords="axes fraction",
                     fontsize=9, color="gray")

        # ── panel 2: QPS vs Candidates (implementation gap) ──────────────────
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Candidates examined (log scale)")
        ax2.set_ylabel("QPS (log scale)")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("QPS vs Candidates  (vertical gap = C++ speedup)")

        # ── panel 3: Dist ratio vs Recall ─────────────────────────────────────
        ax3.set_xlabel("Recall@10")
        ax3.set_ylabel("Dist ratio  (1.0 = perfect)")
        ax3.set_xlim(-0.02, 1.05)
        ax3.axhline(1.0, color="black", lw=0.8, ls=":")
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Dist ratio vs Recall@10")

        # shared legend below all panels
        hs = [h for h, _ in legend_handles]
        ls = [l for _, l in legend_handles]
        n_col = min(len(ls), 7)
        fig.legend(hs, ls, loc="lower center", ncol=n_col, fontsize=8,
                   bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

        fig.tight_layout(rect=[0, 0.08, 1, 1])
        slug = dataset_name.split()[0].lower()
        out  = FIGURES_DIR / f"{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved {out}")


# ── auto-scaling ──────────────────────────────────────────────────────────────

def scale_params(n, d):
    """Return AMPI build/query parameters scaled to dataset size and dimensionality.

    S        : bootstrap NN-pairs for geometry estimation (local power iter needs
               enough pairs to span the d-dim subspace) → O(sqrt(n)), min 500
    L        : projections ∝ d/8 (angular resolution) × log₂(n/5k) (scale),
               rounded to multiples of 8, capped at 64/128
    w_base   : sorted-projection half-window; baseline w=15 at n=10k scales by
               sqrt(n/10k) so the covered fraction of the array stays constant
    bins     : equal-frequency hash bins (subspace_dim=2) sized so each bucket
               holds ~500 vectors → bins = sqrt(n/500); narrow build doubles it
    """
    S = max(500, min(10_000, int(3 * math.sqrt(n))))

    L_raw = (d / 8) * max(1.0, math.log2(n / 5_000))
    L1    = max(16, min(64, 8 * round(L_raw / 8)))
    L2    = min(128, L1 * 2)

    w_base = max(15, int(15 * math.sqrt(n / 10_000)))

    return dict(S=S, L1=L1, L2=L2, w_base=w_base)


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
    S      = pr['S']
    wb     = pr['w_base']
    print(f"  auto-params: L={L1}/{L2}  S={S}  w_base={wb}")

    def build(label, cls, **kw):
        print(f"  Building {label}…", end=" ", flush=True)
        t0  = time.perf_counter()
        idx = cls(data, **kw)
        print(f"{time.perf_counter()-t0:.2f}s")
        return idx

    idx_bin = build(f"Binary   L={L2}", AMPIBinaryIndex, num_projections=L2, seed=0)

    configs = []

    configs.append(("Flat L2",
                    lambda q: (None, None, flat.search(q[None], K)[1][0]),
                    lambda q: n))

    for nprobe in [1, 5, 10, 25, 50]:
        if nprobe > nlist: continue
        def _ivf(q, p=nprobe):
            ivf.nprobe = p
            return (None, None, ivf.search(q[None], K)[1][0])
        configs.append((f"IVF nprobe={nprobe}", _ivf, lambda q, p=nprobe: p * (n // nlist)))

    for w in [wb, 2*wb, 4*wb]:
        configs.append((f"Binary L={L2} w={w}",
                        lambda q, i=idx_bin, w=w: i.query(q, k=K, window_size=w),
                        lambda q, i=idx_bin, w=w: i.query_candidates(q, window_size=w)))

    # === AffineFan: tune index params on a small sample ===
    print(f"  Tuning AFan parameters on sample...", flush=True)

    sample_frac = min(0.1, 50_000 / n)
    n_sample    = max(5000, int(n * sample_frac))
    data_sample = data[np.random.choice(n, n_sample, replace=False)]
    tune_qs     = queries[:QUICK_SAMPLE]

    # Build a flat GT on the sample for the tune queries
    flat_sample = faiss.IndexFlatL2(d)
    flat_sample.add(data_sample)
    _, gt_sample = flat_sample.search(tune_qs, K)
    gt_sample = gt_sample.astype(np.int32)

    # Tune alpha = nlist/sqrt(n) only, using a fixed small F on the sample.
    # F cannot be tuned on the sample: large F gives near-empty cones at 50k scale
    # (e.g. 50k/(223×128)≈1.7 pts/cone), so the sample always favours F=16.
    # Instead, F is derived analytically from the full dataset after picking alpha.
    tune_results = []
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        nlist_tune = max(16, int(alpha * math.sqrt(n_sample)))
        F_tune     = 16  # fixed; only alpha (nlist density) is being selected here
        idx_tune = AMPIAffineFanIndex(data_sample, nlist=nlist_tune, num_fans=F_tune,
                                      C_factor=5, S=min(S, 500), power_iter=1, seed=0)
        for cp in [5, 10]:
            for fp in sorted(set([F_tune // 4, F_tune // 2, F_tune])):
                for w_mult in [0.5, 1.0]:
                    w      = max(5, int(wb * w_mult))
                    cands  = idx_tune.query_candidates(tune_qs[0], window_size=w, probes=cp, fan_probes=fp)
                    res    = [idx_tune.query(q, k=K, window_size=w, probes=cp, fan_probes=fp) for q in tune_qs]
                    idxs   = [r[2] if isinstance(r, tuple) else r for r in res]
                    padded = [np.pad(ix[:K], (0, max(0, K - len(ix))), constant_values=-1) for ix in idxs]
                    rec    = recall(gt_sample, padded)
                    tune_results.append((alpha, cp, fp, w, rec, int(len(cands))))

    pareto_tune = []
    for r in tune_results:
        _, _, _, _, rec, cands = r
        if any(pc <= cands and pr >= rec - PARETO_TOL for pc, pr in pareto_tune):
            continue
        pareto_tune = [(c, rc) for c, rc in pareto_tune if not (c <= cands and rc >= rec)]
        pareto_tune.append((cands, rec))

    best_cands, best_rec = max(pareto_tune, key=lambda x: x[1])
    best_alpha = 1.0
    for alpha, _, _, _, rec, cands in tune_results:
        if abs(cands - best_cands) < best_cands * 0.2 and abs(rec - best_rec) < 0.01:
            best_alpha = alpha
            break

    best_nlist = max(16, int(best_alpha * math.sqrt(n)))

    # Pick the largest F such that the full dataset has >= MIN_CONE_PTS per cone.
    # This scales F with dataset density: SIFT 1M → F=128, MNIST 60k → F=32.
    MIN_CONE_PTS = 5
    viable_Fs = [F for F in sorted(set([16, 32, 64, L2]))
                 if n // (best_nlist * F) >= MIN_CONE_PTS]
    best_F = max(viable_Fs) if viable_Fs else 16

    print(f"    Best: nlist={best_nlist} (alpha={best_alpha:.1f}), F={best_F}")

    idx_af = build(f"AFan     F={best_F}", AMPIAffineFanIndex,
                   nlist=best_nlist, num_fans=best_F,
                   C_factor=5, S=S, power_iter=1, seed=0)

    # Warmup AMPI indexes (triggers JIT compilation)
    print(f"  Warming up AMPI indexes...", flush=True)
    for w in [wb, 2*wb, 4*wb]:
        for q in queries[:WARMUP]:
            idx_bin.query(q, k=K, window_size=w)
    for q in queries[:WARMUP]:
        idx_af.query(q, k=K, window_size=wb, probes=10, fan_probes=best_F)
    print(f"  Warmup complete.")

    # Add AFan query-param sweep to configs (sorted by estimated candidates)
    limit = max(MAX_CAND_FRAC * n, MIN_CAND_ABS)
    af_candidates = []
    for cp in [5, 10, 20]:
        for fp in sorted(set([2, 4, 8, best_F//4, best_F//2, best_F])):
            for w_mult in [0.25, 0.5, 1.0, 1.5, 2.0]:
                w = max(5, int(wb * w_mult))
                af_candidates.append((cp, fp, w, cp * fp * 2 * w))
    af_candidates.sort(key=lambda x: x[3])

    for cp, fp, w, est_cands in af_candidates:
        if est_cands > limit * 1.5:
            continue
        label = f"AFan F={best_F} cp={cp} fp={fp} w={w}"
        configs.append((
            label,
            lambda q, i=idx_af, w=w, cp=cp, fp=fp: i.query(q, k=K, window_size=w, probes=cp, fan_probes=fp),
            lambda q, i=idx_af, w=w, cp=cp, fp=fp: i.query_candidates(q, window_size=w, probes=cp, fan_probes=fp),
        ))

    hdr = f"  {'Method':<38}  {'R@10':>6}  {'dist ratio':>10}  {'QPS':>8}  {'ms/q':>7}  {'cands':>7}"
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
