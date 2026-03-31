"""_bench_common.py — shared constants, dataset loaders, evaluation helpers,
figure generation, and AMPI index builder.

Imported by benchmark_vs_faiss.py and benchmark_vs_hnsw.py.
No FAISS dependency: ground truth is computed via _brute_knn (BLAS gemm).
"""

import sys, time, math
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from ampi import AMPIBinaryIndex, AMPIAffineFanIndex
from ampi.tuner import _GP1D, _pareto_knee, _scale_params as scale_params, _brute_knn

# Re-export so benchmark scripts only need to import from _bench_common.
from download_data import ensure_datasets, DATASETS  # noqa: F401


# ── constants ─────────────────────────────────────────────────────────────────

K             = 10      # primary recall threshold
K_MAX         = 100     # maximum neighbours to retrieve
N_QUERIES     = 200     # queries used for evaluation
WARMUP        = 10      # queries discarded before timing
MAX_CAND_FRAC        = 0.25
MAX_CAND_FRAC_BINARY = 1.00
CAND_SAMPLE   = 1
QUICK_SAMPLE  = 10
TUNE_SAMPLE   = 50
PARETO_TOL    = 0.03

_REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR    = _REPO_ROOT / "data"
FIGURES_DIR = _REPO_ROOT / "figures"


# ── datasets ──────────────────────────────────────────────────────────────────

def _sort_gt(data, queries, gt):
    """Sort each gt row by ascending L2 distance so gt[:, 0] is the true NN.

    _brute_knn uses argpartition which returns the K nearest neighbours in
    arbitrary order.  Recall@k requires gt[:, :k] to be the *actual* top-k,
    so we must sort each row before using it as ground truth.
    """
    sorted_gt = np.empty_like(gt)
    for i, (q, row) in enumerate(zip(queries, gt)):
        diffs = data[row] - q
        dists = np.einsum('ij,ij->i', diffs, diffs)
        sorted_gt[i] = row[np.argsort(dists)]
    return sorted_gt


def make_gaussian(n=10_000, d=128, seed=42):
    rng  = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    qs   = rng.standard_normal((N_QUERIES, d)).astype(np.float32)
    gt   = _brute_knn(data, qs, K_MAX)
    return data, qs, _sort_gt(data, qs, gt).astype(np.int32)


def load_hdf5(path, n_train=None, normalize=False, mmap_dir=None):
    """Load an ANN-benchmark HDF5 file and return (data, queries, gt).

    Parameters
    ----------
    path      : path to .hdf5 file with 'train' and 'test' datasets
    n_train   : optional cap on training vectors (None = use all)
    normalize : if True, L2-normalise both data and queries (for cosine benchmarks)
    mmap_dir  : when set, stream training vectors into a flat file here and return
                a np.memmap instead of a heap array — keeps RSS low for large datasets

    Returns
    -------
    data    : (n, d) float32  (np.memmap when mmap_dir is set)
    queries : (N_QUERIES, d) float32
    gt      : (N_QUERIES, K_MAX) int32 — exact k-NN indices sorted by ascending distance

    When the HDF5 file contains a pre-computed 'neighbors' dataset (ANN-benchmarks
    standard), it is used directly instead of running brute-force kNN.  This avoids
    the ~4 GB peak allocation for large datasets like GIST 1M.
    """
    import os
    with h5py.File(path) as f:
        n_total = f["train"].shape[0]
        d       = f["train"].shape[1]
        n       = n_total if n_train is None else min(n_train, n_total)
        queries = f["test"][:N_QUERIES].astype(np.float32)
        if "neighbors" in f and n_train is None:
            gt = f["neighbors"][:N_QUERIES, :K_MAX].astype(np.int32)
            precomputed_gt = True
        else:
            precomputed_gt = False

        if mmap_dir is not None:
            # Stream HDF5 → flat file in chunks; return memmap so the OS pages on demand.
            os.makedirs(mmap_dir, exist_ok=True)
            fpath = os.path.join(mmap_dir, "_train_data.dat")
            data  = np.memmap(fpath, mode="w+", dtype="float32", shape=(n, d))
            chunk = 50_000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                data[start:end] = f["train"][start:end].astype(np.float32)
            data.flush()
        else:
            data = f["train"][:n].astype(np.float32)

    if normalize:
        data    = data    / (np.linalg.norm(data,    axis=1, keepdims=True) + 1e-10)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    if not precomputed_gt:
        gt = _brute_knn(data, queries, K_MAX)
        gt = _sort_gt(data, queries, gt).astype(np.int32)
    return data, queries, gt


# ── evaluation ────────────────────────────────────────────────────────────────

def recall(gt, approx, k=K):
    """Recall@k: fraction of true top-k neighbours found in returned top-k."""
    hits = sum(
        len(set(g[:k].tolist()) & set(a[:k].tolist()))
        for g, a in zip(gt, approx)
    )
    return hits / (len(gt) * k)


def avg_cands(cands_fn, queries):
    return int(np.mean([
        c if isinstance(c, (int, np.integer)) else len(c)
        for c in (cands_fn(q) for q in queries[:CAND_SAMPLE])
    ]))


def approx_ratio(data, queries, gt_indices, found_indices):
    """Mean ratio: avg distance to found neighbours / avg distance to true neighbours."""
    ratios = []
    for q, true_idx, found_idx in zip(queries, gt_indices, found_indices):
        true_d  = np.sqrt(np.sum((data[true_idx]      - q) ** 2, axis=1)).mean()
        found_d = np.sqrt(np.sum((data[found_idx[:K]] - q) ** 2, axis=1)).mean()
        ratios.append(found_d / (true_d + 1e-10))
    return float(np.mean(ratios))


def _pareto_group(label):
    """Broad family group for within-family Pareto pruning."""
    parts = label.split()
    p = parts[0]
    if p == "AFan":
        return f"AFan {parts[1]} {parts[2]}"
    return p


def evaluate(label, query_fn, cands_fn, queries, gt, data, n, pareto=None,
             exact_labels=()):
    """Run one benchmark configuration and print a result row.

    Parameters
    ----------
    exact_labels : labels exempt from candidate-count and Pareto pruning
                   (e.g. "Flat L2" for FAISS, or a brute-force baseline).
    """
    cands     = avg_cands(cands_fn, queries)
    is_binary = label.startswith("Binary")
    limit     = (MAX_CAND_FRAC_BINARY if is_binary else MAX_CAND_FRAC) * n

    if label not in exact_labels and cands > limit:
        print(f"  [skip] {label}  — avg candidates > {limit:,.0f}", flush=True)
        return None

    if pareto is not None and label not in exact_labels:
        group       = _pareto_group(label)
        group_front = pareto.get(group, [])
        if group_front:
            quick_results = [query_fn(q) for q in queries[:QUICK_SAMPLE]]
            quick_idx     = [r[2] if isinstance(r, tuple) else r for r in quick_results]
            quick_pad     = [np.pad(ix[:K], (0, max(0, K - len(ix))), constant_values=-1)
                             for ix in quick_idx]
            est_rec       = recall(gt[:QUICK_SAMPLE], quick_pad, K)
            max_front_rec = max(pr for _, pr in group_front)
            if est_rec <= max_front_rec and any(
                p_cands <= cands and p_rec >= est_rec - PARETO_TOL
                for p_cands, p_rec in group_front
            ):
                print(f"  [skip-pareto] {label}  "
                      f"(est R@10={est_rec:.3f}, cands≈{cands:,})", flush=True)
                return None

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


def print_table_header():
    hdr = (f"  {'Method':<38}  {'R@1':>6}  {'R@10':>6}  {'R@100':>6}"
           f"  {'dist ratio':>10}  {'QPS':>8}  {'ms/q':>7}  {'cands':>7}")
    print()
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))


def run_evaluation(configs, queries, gt, data, n, exact_labels=()):
    """Evaluate all configs and return result rows."""
    rows   = []
    pareto = {}
    for label, fn, cands_fn in configs:
        r = evaluate(label, fn, cands_fn, queries, gt, data, n,
                     pareto=pareto, exact_labels=exact_labels)
        if r is not None:
            rows.append(r)
            group = _pareto_group(label)
            front = pareto.get(group, [])
            front = [(c, rc) for c, rc in front
                     if not (r["cands"] <= c and r["recall"] >= rc)]
            front.append((r["cands"], r["recall"]))
            pareto[group] = front
    return rows


# ── figures ───────────────────────────────────────────────────────────────────

# Base styles shared across benchmarks; each benchmark extends this dict.
FAMILY_STYLE_BASE = {
    "Flat L2":  dict(color="black",   marker="*",  ls="none", ms=14, zorder=5),
    "Binary":   dict(color="#8c8c8c", marker="s",  ls="-",    lw=1.5, ms=7),
    "AFan K=1": dict(color="#9467bd", marker="D",  ls="-",    lw=2,   ms=7),
    "AFan K=2": dict(color="#e377c2", marker="D",  ls="--",   lw=2,   ms=7),
    "AFan K=3": dict(color="#6a0dad", marker="D",  ls=":",    lw=2,   ms=7),
}


def _family(label):
    parts = label.split()
    if parts[0] == "Flat":   return "Flat L2"
    if parts[0] == "Binary": return "Binary"
    if parts[0] == "AFan":
        k_part = next((p for p in parts if p.startswith("K=")), "K=1")
        return f"AFan {k_part}"
    return parts[0]   # IVF, HNSW, …


def _pareto_frontier(xs, ys):
    """Return indices of Pareto-optimal (min-x, max-y) points."""
    pts      = sorted(enumerate(zip(xs, ys)), key=lambda t: t[1][0])
    frontier = []
    best_y   = -np.inf
    for idx, (x, y) in pts:
        if y > best_y:
            best_y = y
            frontier.append(idx)
    return frontier


def save_figures(all_results, family_style=None, suffix=""):
    """Save one PNG per dataset into FIGURES_DIR.

    Parameters
    ----------
    all_results  : list of (dataset_name, rows) tuples
    family_style : dict mapping family name → matplotlib style kwargs;
                   defaults to FAMILY_STYLE_BASE
    suffix       : appended to the output filename slug (e.g. "_vs_hnsw")
    """
    if family_style is None:
        family_style = FAMILY_STYLE_BASE
    FIGURES_DIR.mkdir(exist_ok=True)

    for dataset_name, rows in all_results:
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
            style = family_style.get(fam, dict(color="gray", marker="x", ls="-", lw=1))
            cands  = [p["cands"]     for p in pts]
            rec1   = [p["recall1"]   for p in pts]
            rec10  = [p["recall"]    for p in pts]
            rec100 = [p["recall100"] for p in pts]
            ratio  = [p["ratio"]     for p in pts]
            qps    = [p["qps"]       for p in pts]
            kw = {k: v for k, v in style.items() if k != "zorder"}
            zo = style.get("zorder", 2)

            line, = ax_r1.plot(cands, rec1, zorder=zo, **kw)
            ax_r10.plot(cands, rec10, zorder=zo, **kw)
            ax_r100.plot(cands, rec100, zorder=zo, **kw)
            ax_qps.plot(cands, qps, zorder=zo, **kw)
            ax_ratio.plot(rec10, ratio, zorder=zo, **kw)
            legend_handles.append((line, fam))

            fi = _pareto_frontier(cands, rec10)
            if len(fi) > 1:
                px = [cands[i] for i in fi]
                py = [rec10[i]  for i in fi]
                ax_r10.plot(px, py, color=style.get("color", "gray"),
                            lw=2.5, ls="-", alpha=0.35, zorder=zo - 1)
                pf   = [(None, None, None, rec10[i], cands[i]) for i in fi]
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
        ax_qps.set_title("QPS vs Candidates")

        ax_ratio.set_xlabel("Recall@10")
        ax_ratio.set_ylabel("Dist ratio  (1.0 = perfect)")
        ax_ratio.set_xlim(-0.02, 1.05)
        ax_ratio.axhline(1.0, color="black", lw=0.8, ls=":")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_title("Dist ratio vs Recall@10")

        hs     = [h for h, _ in legend_handles]
        labels = [l for _, l in legend_handles]
        fig.legend(hs, labels, loc="lower center", ncol=min(len(labels), 7),
                   fontsize=8, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        slug = dataset_name.split()[0].lower()
        out  = FIGURES_DIR / f"{slug}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved {out}")


# ── AMPI index builder ────────────────────────────────────────────────────────

def build_ampi_configs(data, queries, gt, metric='l2', data_path=None):
    """Tune and build AMPI indexes; return a list of (label, query_fn, cands_fn).

    Handles Binary index, GP-BO alpha tuning, AffineFan index construction,
    JIT warmup, and the query-parameter sweep.  Caller appends the returned
    list to their competitor configs before calling run_evaluation().
    """
    n, d = data.shape
    pr   = scale_params(n, d)
    L1, L2 = pr['L1'], pr['L2']
    wb      = pr['w_base']
    print(f"  auto-params: L={L1}/{L2}  w_base={wb}")

    def _build(label, cls, **kw):
        print(f"  Building {label}…", end=" ", flush=True)
        t0  = time.perf_counter()
        idx = cls(data, **kw)
        print(f"{time.perf_counter() - t0:.2f}s")
        return idx

    idx_bin = _build(f"Binary   L={L2}", AMPIBinaryIndex, num_projections=L2, seed=0)

    # Tune alpha = nlist / sqrt(n) via GP-BO on a data subsample
    print("  Tuning AFan parameters on sample...", flush=True)
    n_sample    = max(10_000, min(100_000, int(n * 0.2)))
    data_sample = data[np.random.choice(n, n_sample, replace=False)]
    tune_qs     = queries[:TUNE_SAMPLE]
    gt_sample   = _brute_knn(data_sample, tune_qs, K)

    def _alpha_score(alpha):
        nlist_s      = max(16, int(alpha * math.sqrt(n_sample)))
        cluster_size = max(1, n_sample // nlist_s)
        idx_s        = AMPIAffineFanIndex(data_sample, nlist=nlist_s, num_fans=16,
                                          seed=0, metric=metric)
        scores = []
        fp = 8
        for target_frac in [0.02, 0.05, 0.15]:
            T  = max(K + 1, int(n_sample * target_frac))
            cp = max(1, min(nlist_s, T // cluster_size))
            w  = max(5, T // max(1, 2 * cp * fp))
            res    = [idx_s.query(q, k=K, window_size=w, probes=cp, fan_probes=fp)
                      for q in tune_qs]
            idxs   = [r[2] if isinstance(r, tuple) else r for r in res]
            padded = [np.pad(ix[:K], (0, max(0, K - len(ix))), constant_values=-1)
                      for ix in idxs]
            scores.append(recall(gt_sample, padded, K))
        return float(np.mean(scores))

    ALPHA_LO, ALPHA_HI = 0.50, 2.0
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
    MIN_CONE_PTS = 5
    viable_Fs  = [F for F in sorted({16, 32, 64, L1})
                  if n // (best_nlist * F) >= MIN_CONE_PTS]
    best_F     = max(viable_Fs) if viable_Fs else 16
    print(f"    Best: nlist={best_nlist} (alpha={best_alpha:.3f}), F={best_F}")

    # Use streaming build for large datasets (avoids random mmap page faults).
    _STREAMING_THRESHOLD = 200_000
    _use_streaming = (data_path is not None and n > _STREAMING_THRESHOLD)

    def _build_streaming(label, ktk):
        import os
        from ampi.streaming import streaming_build
        s_path = os.path.join(data_path, f"_ampi_K{ktk}")
        os.makedirs(s_path, exist_ok=True)
        print(f"  Building {label} (streaming)…", end=" ", flush=True)
        t0  = time.perf_counter()
        idx = streaming_build(
            data_source=lambda s, e: data[s:e],
            n=n, d=d, nlist=best_nlist, num_fans=best_F,
            cone_top_k=ktk, seed=0, metric=metric,
            data_path=s_path,
        )
        print(f"{time.perf_counter() - t0:.2f}s")
        return idx

    af_indexes = {}
    for ktk in [1, 2]:
        tag = f"AFan F={best_F} K={ktk}"
        if _use_streaming:
            af_indexes[ktk] = _build_streaming(tag, ktk)
        else:
            kw  = dict(nlist=best_nlist, num_fans=best_F,
                       seed=0, cone_top_k=ktk, metric=metric)
            if data_path is not None:
                import os
                kw["data_path"] = os.path.join(data_path, f"_ampi_K{ktk}")
                os.makedirs(kw["data_path"], exist_ok=True)
            af_indexes[ktk] = _build(tag, AMPIAffineFanIndex, **kw)

    print("  Warming up AMPI indexes...", flush=True)
    for w in [wb, 2 * wb, 4 * wb]:
        for q in queries[:WARMUP]:
            idx_bin.query(q, k=K, window_size=w)
    for ktk, idx_af in af_indexes.items():
        for q in queries[:WARMUP]:
            idx_af.query(q, k=K, window_size=wb, probes=10, fan_probes=best_F)
    print("  Warmup complete.")

    # Build the query-parameter sweep configs
    limit         = MAX_CAND_FRAC * n
    af_candidates = []
    for cp in [5, 10, 20, 50]:
        for fp in sorted({2, 4, 8, best_F // 4, best_F // 2, best_F}):
            for w_mult in [0.25, 0.5, 1.0, 1.5, 2.0]:
                w = max(5, int(wb * w_mult))
                af_candidates.append((cp, fp, w, cp * fp * 2 * w))
    af_candidates.sort(key=lambda x: x[3])

    configs = []
    for w in [wb, 2 * wb, 4 * wb]:
        configs.append((
            f"Binary L={L2} w={w}",
            lambda q, i=idx_bin, w=w: i.query(q, k=K_MAX, window_size=w),
            lambda q, i=idx_bin, w=w: i.query_candidates(q, window_size=w),
        ))
    for ktk, idx_af in af_indexes.items():
        for cp, fp, w, est_cands in af_candidates:
            if est_cands > limit * 1.5:
                continue
            label = f"AFan F={best_F} K={ktk} cp={cp} fp={fp} w={w}"
            configs.append((
                label,
                lambda q, i=idx_af, w=w, cp=cp, fp=fp:
                    i.query(q, k=K_MAX, window_size=w, probes=cp, fan_probes=fp),
                lambda q, i=idx_af, w=w, cp=cp, fp=fp:
                    i.query_candidates(q, window_size=w, probes=cp, fan_probes=fp),
            ))
    return configs
