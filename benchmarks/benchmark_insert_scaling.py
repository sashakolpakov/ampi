"""benchmark_insert_scaling.py — measure per-insert latency vs index size.

Tests whether AMPI's insert cost scales as O(F·√n) (sorted-array shift in the
affected cone) or is closer to O(1) in practice, and compares against HNSW
(O(log n)) and FAISS IVF (O(1) amortised, no quality guarantee post-insert).

The benchmark builds a base index of n_base vectors, then inserts vectors one
by one, sampling a burst of `batch` inserts at log-spaced checkpoints and
recording the median per-insert wall-clock time in µs.

Usage
-----
    python benchmarks/benchmark_insert_scaling.py
    python benchmarks/benchmark_insert_scaling.py --n-max 500000
    python benchmarks/benchmark_insert_scaling.py --dataset sift --save
"""

import sys, time, argparse, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ampi import AMPIAffineFanIndex
from _bench_common import DATA_DIR, FIGURES_DIR
from download_data import ensure_datasets, DATASETS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── index builders ────────────────────────────────────────────────────────────

def build_ampi(train, nlist, F):
    return AMPIAffineFanIndex(train, nlist=nlist, num_fans=F, seed=0)


def build_hnsw(train, n_max_total):
    try:
        import hnswlib
        d   = train.shape[1]
        idx = hnswlib.Index(space="l2", dim=d)
        idx.init_index(max_elements=n_max_total, ef_construction=200, M=16)
        ids = np.arange(len(train), dtype=np.int64)
        idx.add_items(train, ids, num_threads=1)
        return idx
    except ImportError:
        return None


def build_faiss_ivf(train, nlist):
    try:
        import faiss
        d   = train.shape[1]
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist,
                                 faiss.METRIC_L2)
        ivf.train(train)
        ivf.add(train)
        return ivf
    except ImportError:
        return None


# ── slope analysis ────────────────────────────────────────────────────────────

def fit_slope(ns, ys):
    """OLS on log-log scale.  Returns exponent a in t ∝ n^a."""
    lx = np.log(np.asarray(ns, float))
    ly = np.log(np.asarray(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])


# ── main ──────────────────────────────────────────────────────────────────────

def run(dataset, n_base, n_max, batch, save):
    rng = np.random.default_rng(42)

    # ── load / generate data ──────────────────────────────────────────────────
    if dataset == "gauss":
        d        = 128
        all_data = rng.standard_normal((n_max + batch, d)).astype("float32")
    else:
        ensure_datasets([dataset])
        import h5py
        path = DATA_DIR / DATASETS[dataset]["file"]
        with h5py.File(path) as fh:
            raw = np.array(fh["train"], dtype="float32")
        need = n_max + batch
        if len(raw) < need:
            n_max = len(raw) - batch
            print(f"[warn] dataset has only {len(raw)} vectors; "
                  f"n_max capped at {n_max:,}")
        all_data = raw[:n_max + batch]
        d        = all_data.shape[1]

    train = all_data[:n_base]
    pool  = all_data[n_base:]          # vectors inserted after initial build
    n_pool = len(pool)

    print(f"Dataset: {dataset}  d={d}  n_base={n_base:,}  "
          f"n_max={n_max:,}  batch={batch}")

    # ── AMPI nlist/F (reuse nlist for FAISS IVF) ──────────────────────────────
    nlist = max(16, int(round(math.sqrt(n_base))))
    F     = 64

    print(f"\nBuilding AMPI  nlist={nlist}  F={F} …", end=" ", flush=True)
    t0   = time.perf_counter()
    ampi = build_ampi(train, nlist, F)
    print(f"{time.perf_counter()-t0:.1f}s")

    print("Building HNSW …", end=" ", flush=True)
    t0   = time.perf_counter()
    hnsw = build_hnsw(train, n_max + batch)
    print(f"{time.perf_counter()-t0:.1f}s" if hnsw else "hnswlib not installed — skipped")

    print("Building FAISS IVF …", end=" ", flush=True)
    t0   = time.perf_counter()
    faiss_ivf = build_faiss_ivf(train, nlist)
    print(f"{time.perf_counter()-t0:.1f}s" if faiss_ivf else "faiss not installed — skipped")

    # ── log-spaced checkpoints between n_base and n_max ──────────────────────
    n_steps     = max(6, int(math.log2(n_max / n_base) * 5))
    checkpoints = sorted(set(
        np.geomspace(n_base, n_max - batch, n_steps).astype(int).tolist()))

    # State tracking — how many pool vectors have been inserted so far.
    # Inserts are permanent; we advance the state monotonically.
    ampi_n    = n_base
    hnsw_id   = n_base      # next hnswlib integer label
    ampi_ptr  = 0           # next index into pool for non-timed advances
    hnsw_ptr  = 0
    faiss_ptr = 0

    results   = {label: {"ns": [], "us": []}
                 for label in ("AMPI", "HNSW", "FAISS-IVF")}

    print(f"\n{'n':>10}  {'AMPI µs':>10}  {'HNSW µs':>10}  {'FAISS µs':>10}")
    print("-" * 46)

    for ckpt in checkpoints:
        # How many pool vectors should already be in the index at this checkpoint?
        need = ckpt - n_base

        # ── advance AMPI (non-timed, one-by-one) to ckpt ─────────────────────
        while ampi_ptr < need:
            ampi.add(pool[ampi_ptr])
            ampi_ptr += 1
        ampi_n = n_base + ampi_ptr

        # ── advance HNSW (non-timed, batched) to ckpt ────────────────────────
        if hnsw and hnsw_ptr < need:
            ids = np.arange(n_base + hnsw_ptr, n_base + need, dtype=np.int64)
            hnsw.add_items(pool[hnsw_ptr:need], ids)
            hnsw_ptr = need

        # ── advance FAISS IVF (non-timed, batched) to ckpt ───────────────────
        if faiss_ivf and faiss_ptr < need:
            faiss_ivf.add(pool[faiss_ptr:need])
            faiss_ptr = need

        # ── timed burst of `batch` fresh inserts ──────────────────────────────
        end = min(need + batch, n_pool)
        if end <= need:
            break
        vecs = pool[need:end]   # these have NOT been inserted yet

        # AMPI
        t0 = time.perf_counter()
        for v in vecs:
            ampi.add(v)
        us_ampi = (time.perf_counter() - t0) / len(vecs) * 1e6
        ampi_ptr = end

        # HNSW
        if hnsw:
            base_id = n_base + need
            ids     = np.arange(base_id, base_id + len(vecs), dtype=np.int64)
            t0      = time.perf_counter()
            hnsw.add_items(vecs, ids)
            us_hnsw = (time.perf_counter() - t0) / len(vecs) * 1e6
            hnsw_ptr = end
        else:
            us_hnsw = float("nan")

        # FAISS IVF
        if faiss_ivf:
            t0      = time.perf_counter()
            faiss_ivf.add(vecs)
            us_faiss = (time.perf_counter() - t0) / len(vecs) * 1e6
            faiss_ptr = end
        else:
            us_faiss = float("nan")

        results["AMPI"]["ns"].append(ckpt)
        results["AMPI"]["us"].append(us_ampi)
        results["HNSW"]["ns"].append(ckpt)
        results["HNSW"]["us"].append(us_hnsw)
        results["FAISS-IVF"]["ns"].append(ckpt)
        results["FAISS-IVF"]["us"].append(us_faiss)

        hstr  = f"{us_hnsw:10.1f}" if not math.isnan(us_hnsw)  else "      skip"
        fstr  = f"{us_faiss:10.1f}" if not math.isnan(us_faiss) else "      skip"
        print(f"{ckpt:10,}  {us_ampi:10.1f}  {hstr}  {fstr}")

    # ── slope analysis ────────────────────────────────────────────────────────
    print("\n── Scaling exponent (log-log OLS; O(1)→0  O(log n)→~0.3  "
          "O(√n)→0.5  O(n)→1) ──")
    for label, r in results.items():
        ns    = np.array(r["ns"])
        us    = np.array(r["us"])
        valid = ~np.isnan(us)
        if valid.sum() < 3:
            print(f"  {label:<12}  (insufficient data)")
            continue
        a       = fit_slope(ns[valid], us[valid])
        verdict = ("≈O(1)"     if a < 0.15 else
                   "≈O(log n)" if a < 0.35 else
                   "≈O(√n)"    if a < 0.65 else
                   "≈O(n)")
        print(f"  {label:<12}  slope={a:+.3f}  {verdict}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"AMPI": "#1f77b4", "HNSW": "#ff7f0e", "FAISS-IVF": "#2ca02c"}

    for label, r in results.items():
        ns    = np.array(r["ns"])
        us    = np.array(r["us"])
        valid = ~np.isnan(us)
        if not valid.any():
            continue
        ax.plot(ns[valid], us[valid], "o-", label=label,
                color=colours[label], linewidth=1.5, markersize=5)

    # reference lines anchored at the first AMPI measurement
    ns_ref  = np.array([n_base, n_max], dtype=float)
    u0      = results["AMPI"]["us"][0] if results["AMPI"]["us"] else 1.0
    ax.plot(ns_ref, [u0, u0],
            "--", color="grey", linewidth=0.9, alpha=0.55, label="O(1) ref")
    ax.plot(ns_ref, u0 * (ns_ref / n_base) ** 0.5,
            ":",  color="grey", linewidth=0.9, alpha=0.55, label="O(√n) ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Index size  n")
    ax.set_ylabel("Per-insert latency  (µs)")
    ax.set_title(f"Insert scaling — {dataset}  d={d}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / f"insert_scaling_{dataset}.png"
        fig.savefig(out, dpi=150)
        print(f"\nFigure saved to {out}")
    else:
        print("\n(pass --save to write figure to figures/)")

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Benchmark per-insert latency vs index size for AMPI, HNSW, FAISS IVF.")
    ap.add_argument("--dataset",  default="gauss",
                    choices=["gauss"] + list(DATASETS),
                    help="Dataset (default: gauss synthetic)")
    ap.add_argument("--n-base",   type=int, default=50_000,
                    help="Initial build size (default: 50000)")
    ap.add_argument("--n-max",    type=int, default=500_000,
                    help="Final index size (default: 500000)")
    ap.add_argument("--batch",    type=int, default=500,
                    help="Vectors per timed burst at each checkpoint (default: 500)")
    ap.add_argument("--save",     action="store_true",
                    help="Save figure to figures/insert_scaling_<dataset>.png")
    args = ap.parse_args()

    run(dataset=args.dataset,
        n_base=args.n_base,
        n_max=args.n_max,
        batch=args.batch,
        save=args.save)
