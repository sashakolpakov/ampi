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
    AMPITomographicIndex, AMPITwoStageIndex,
    AMPIPrincipalFanIndex,
)

K             = 10      # neighbours to retrieve
N_QUERIES     = 200     # queries used for evaluation
WARMUP        = 10      # queries discarded before timing
MAX_CAND_FRAC = 0.40    # skip configs whose avg candidates exceed this fraction of n …
MAX_CAND_ABS  = 50_000  # … but never allow more than this regardless of n
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
    IVF, Tomo-loc, TwoStage, Fan are kept separate so a fast IVF result
    cannot dominate and prune the entire AMPI curve.
    """
    parts = label.split()
    p = parts[0]
    if p == "Fan":
        return f"Fan {parts[1]}"   # separate frontier per K value
    return p


def evaluate(label, query_fn, cands_fn, queries, gt, data, n, pareto=None):
    # Step 1: candidate count check (1 query)
    cands = avg_cands(cands_fn, queries)
    limit = min(MAX_CAND_FRAC * n, MAX_CAND_ABS)
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
    print(f"  {label:<24}  {rec:>6.3f}  {ratio:>10.4f}  {1e3/ms:>8.1f}  {ms:>7.3f}  {cands_str}",
          flush=True)
    return r


# ── figures ───────────────────────────────────────────────────────────────────

# Map label prefix → (color, marker, linestyle)
_FAMILY_STYLE = {
    "Flat L2":        dict(color="black",   marker="*",  ls="none", ms=14, zorder=5),
    "IVF":            dict(color="#1f77b4", marker="o",  ls="-",    lw=2),
    "Binary":         dict(color="#8c8c8c", marker="s",  ls="-",    lw=1.5, ms=7),
    "Tomo-loc L=16":  dict(color="#2ca02c", marker="^",  ls="-",    lw=1.5),
    "Tomo-loc L=32":  dict(color="#006400", marker="^",  ls="--",   lw=1.5),
    "TwoStage L=16":  dict(color="#e6194b", marker="P",  ls="-",    lw=2,  ms=9),
    "TwoStage L=32":  dict(color="#911eb4", marker="P",  ls="--",   lw=2,  ms=9),
    "Fan K=16":       dict(color="#e07020", marker="D",  ls="-",    lw=2,   ms=7),
    "Fan K=32":       dict(color="#a03000", marker="D",  ls="--",   lw=2,   ms=7),
}


def _family(label):
    parts = label.split()
    if parts[0] == "Flat":            return "Flat L2"
    if parts[0] == "IVF":             return "IVF"
    if parts[0] == "Binary":          return "Binary"
    if parts[0].startswith("Tomo-"):  return f"{parts[0]} {parts[1]}"
    if parts[0] == "TwoStage":        return f"TwoStage {parts[1]}"
    if parts[0] == "Fan":             return f"Fan {parts[1]}"
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

    bins_broad  = max(4,  min(32, int(math.sqrt(n / 500))))
    bins_narrow = max(6,  min(64, bins_broad * 2))

    return dict(S=S, L1=L1, L2=L2, w_base=w_base,
                bins_broad=bins_broad, bins_narrow=bins_narrow)


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
    bins_b = pr['bins_broad']
    print(f"  auto-params: L={L1}/{L2}  S={S}  w_base={wb}  bins={bins_b}")

    def build(label, cls, **kw):
        print(f"  Building {label}…", end=" ", flush=True)
        t0  = time.perf_counter()
        idx = cls(data, **kw)
        print(f"{time.perf_counter()-t0:.2f}s")
        return idx

    idx_tomo16 = build(f"Tomo-loc L={L1}", AMPITomographicIndex, num_projections=L1, C_factor=5, S=S, power_iter=1, local=True, seed=0)
    idx_tomo32 = build(f"Tomo-loc L={L2}", AMPITomographicIndex, num_projections=L2, C_factor=5, S=S, power_iter=1, local=True, seed=0)
    idx_2st16  = build(f"TwoStage L={L1}", AMPITwoStageIndex,    num_projections=L1, bins_per_axis=bins_b, subspace_dim=2, C_factor=5, S=S, power_iter=1, local=True, seed=0)
    idx_2st32  = build(f"TwoStage L={L2}", AMPITwoStageIndex,    num_projections=L2, bins_per_axis=bins_b, subspace_dim=2, C_factor=5, S=S, power_iter=1, local=True, seed=0)

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

    # Binary: random-direction baseline — the simplest AMPI method, equivalent to the
    # original C++ code.  Everything else should beat this on structured data.
    idx_bin = build("Binary L=16", AMPIBinaryIndex, num_projections=L2, seed=0)
    for w in [wb, 2*wb, 4*wb]:
        configs.append((f"Binary L={L2} w={w}",
                        lambda q, i=idx_bin, w=w: i.query(q, k=K, window_size=w),
                        lambda q, i=idx_bin, w=w: i.query_candidates(q, window_size=w)))

    # Tomo-loc: three window sizes per L value
    for idx, L in [(idx_tomo16, L1), (idx_tomo32, L2)]:
        for w in [wb, 2*wb, 4*wb]:
            configs.append((f"Tomo-loc L={L} w={w}",
                            lambda q, i=idx, w=w: i.query(q, k=K, window_size=w),
                            lambda q, i=idx, w=w: i.query_candidates(q, window_size=w)))

    # Two-stage: coarse subspace hash → voting precision pass
    for idx, L in [(idx_2st16, L1), (idx_2st32, L2)]:
        for sp, fw in [(1, 0.08), (1, 0.20), (2, 0.20)]:
            label = f"TwoStage L={L} sp={sp} fw={fw}"
            configs.append((label,
                            lambda q, i=idx, sp=sp, fw=fw: i.query(q, k=K, sub_probes=sp, fine_window=fw),
                            lambda q, i=idx, sp=sp, fw=fw: i.query_candidates(q, sub_probes=sp, fine_window=fw, min_return=K)))

    # Fan: principal-direction cones (unit-sphere assignment); sorted projections within each cone.
    # Window scales with per-cone size n/K.
    for K_fans in [L1, L2]:
        w_fan = max(5, int(15 * math.sqrt(n / (K_fans * 10_000))))
        idx_fan = build(f"Fan K={K_fans}", AMPIPrincipalFanIndex,
                        num_fans=K_fans, C_factor=5, S=S, power_iter=1, seed=0)
        for w, probes in [(w_fan, 1), (2*w_fan, 1), (w_fan, 2), (2*w_fan, 2)]:
            configs.append((f"Fan K={K_fans} w={w} p={probes}",
                            lambda q, i=idx_fan, w=w, p=probes: i.query(q, k=K, window_size=w, probes=p),
                            lambda q, i=idx_fan, w=w, p=probes: i.query_candidates(q, window_size=w, probes=p)))

    hdr = f"  {'Method':<24}  {'R@10':>6}  {'dist ratio':>10}  {'QPS':>8}  {'ms/q':>7}  {'cands':>7}"
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
