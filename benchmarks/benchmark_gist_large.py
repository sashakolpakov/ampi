"""benchmark_gist_large.py — AMPI vs FAISS vs HNSW on GIST, always mmap-backed.

Memory for FAISS and HNSW is estimated before building; each is skipped
automatically if the estimate exceeds the available limit.

Usage
  python benchmark_gist_large.py              # full 1M, auto-skip heavy indexes
  python benchmark_gist_large.py --n 250000  # 250k subset
  python benchmark_gist_large.py --no-faiss --no-hnsw
  python benchmark_gist_large.py --mem-limit 8  # GB cap (default: auto-detect)

Output
  figures/gist_large.png
  stdout table
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from _bench_common import (
    K, K_MAX, DATA_DIR,
    load_hdf5,
    print_table_header, run_evaluation,
    build_ampi_configs,
    FAMILY_STYLE_BASE, save_figures,
    ensure_datasets,
)

_FAMILY_STYLE = {
    **FAMILY_STYLE_BASE,
    "HNSW":  dict(color="#2ca02c", marker="^", ls="-", lw=2),
    "IVF":   dict(color="#1f77b4", marker="o", ls="-", lw=2),
    "Flat L2": dict(color="black", marker="*", ls="none", ms=14, zorder=5),
}

_GIST_HDF5 = DATA_DIR / "gist" / "gist-960-euclidean.hdf5"
_MMAP_BASE  = DATA_DIR / "gist" / "_ampi_large"


# ── memory helpers ─────────────────────────────────────────────────────────────

def _available_ram_gb():
    """Best-effort available RAM in GB; falls back to 6 GB if psutil missing."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return 6.0


def _faiss_est_gb(n, d):
    """FAISS Flat/IVF stores one float32 copy of the data."""
    return n * d * 4 / 1e9


def _hnsw_est_gb(n, d, M=16):
    """hnswlib stores the data vectors + graph edges.
    Graph overhead ≈ n * M * 2 * sizeof(int32) * avg_levels (≈ 1.2 levels).
    """
    data_bytes  = n * d * 4
    graph_bytes = n * M * 2 * 4 * 1.2   # int32 neighbour IDs
    return (data_bytes + graph_bytes) / 1e9


# ── competitor builders ────────────────────────────────────────────────────────

def _build_faiss(data, n, d, mem_limit_gb):
    est = _faiss_est_gb(n, d)
    if est > mem_limit_gb:
        print(f"  [skip FAISS] estimated {est:.1f} GB > {mem_limit_gb:.1f} GB limit")
        return []
    try:
        import faiss
    except ImportError:
        print("  [skip FAISS] faiss not installed")
        return []

    configs = []
    try:
        print(f"  Building FAISS Flat (est {est:.1f} GB)...", end=" ", flush=True)
        t0   = time.perf_counter()
        flat = faiss.IndexFlatL2(d)
        flat.add(np.asarray(data))          # FAISS needs a contiguous heap copy
        print(f"{time.perf_counter() - t0:.2f}s")
        configs.append((
            "Flat L2",
            lambda q: (None, None, flat.search(q[None], K_MAX)[1][0]),
            lambda q: n,
        ))

        nlist = max(16, int(np.sqrt(n)))
        print(f"  Building FAISS IVF nlist={nlist}...", end=" ", flush=True)
        t0  = time.perf_counter()
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        ivf.train(np.asarray(data))
        ivf.add(np.asarray(data))
        print(f"{time.perf_counter() - t0:.2f}s")
        for nprobe in [1, 5, 10, 25, 50]:
            if nprobe > nlist:
                continue
            def _ivf(q, p=nprobe):
                ivf.nprobe = p
                return (None, None, ivf.search(q[None], K_MAX)[1][0])
            configs.append((
                f"IVF nprobe={nprobe}", _ivf,
                lambda q, p=nprobe: p * (n // nlist),
            ))
    except (MemoryError, RuntimeError) as exc:
        print(f"\n  [skip FAISS] OOM during build: {exc}")
    return configs


def _build_hnsw(data, n, d, mem_limit_gb):
    est = _hnsw_est_gb(n, d)
    if est > mem_limit_gb:
        print(f"  [skip HNSW] estimated {est:.1f} GB > {mem_limit_gb:.1f} GB limit")
        return []
    try:
        import hnswlib
    except ImportError:
        print("  [skip HNSW] hnswlib not installed")
        return []

    try:
        print(f"  Building HNSW M=16 ef_construction=200 (est {est:.1f} GB)...",
              end=" ", flush=True)
        t0   = time.perf_counter()
        hnsw = hnswlib.Index(space="l2", dim=d)
        hnsw.init_index(max_elements=n, ef_construction=200, M=16)
        hnsw.add_items(data, num_threads=os.cpu_count())
        print(f"{time.perf_counter() - t0:.2f}s")
    except (MemoryError, RuntimeError) as exc:
        print(f"\n  [skip HNSW] OOM during build: {exc}")
        return []

    configs = []
    for ef in [10, 20, 50, 100, 200, 400, 800]:
        def _q(q, ef=ef):
            hnsw.set_ef(ef)
            labels, _ = hnsw.knn_query(q[None], k=min(K_MAX, n))
            return (None, None, labels[0].astype(np.int32))
        configs.append((f"HNSW ef={ef}", _q, lambda q, ef=ef: ef))
    return configs


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="Number of training vectors (default: 1M)")
    ap.add_argument("--no-faiss", dest="faiss", action="store_false",
                    help="Skip FAISS indexes")
    ap.add_argument("--no-hnsw",  dest="hnsw",  action="store_false",
                    help="Skip HNSW")
    ap.add_argument("--mem-limit", type=float, default=None, metavar="GB",
                    help="RAM budget in GB for competitor indexes "
                         "(default: auto-detect available RAM × 0.6)")
    args = ap.parse_args()

    avail = _available_ram_gb()
    mem_limit = args.mem_limit if args.mem_limit is not None else avail * 0.6
    print(f"Available RAM: {avail:.1f} GB  →  competitor budget: {mem_limit:.1f} GB")

    ensure_datasets({"gist"})

    mmap_dir = str(_MMAP_BASE / f"n{args.n}")
    os.makedirs(mmap_dir, exist_ok=True)

    print(f"Loading GIST n={args.n:,} d=960 (mmap)...")
    t0 = time.perf_counter()
    data, queries, gt = load_hdf5(_GIST_HDF5, n_train=args.n, mmap_dir=mmap_dir)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    n, d  = data.shape
    label = f"GIST {n:,}  d={d}"
    sep   = "═" * 72
    print(f"\n{sep}")
    print(f"  {label}  —  queries={len(queries)}  k={K}")
    print(sep)

    faiss_configs = _build_faiss(data, n, d, mem_limit) if args.faiss else []
    hnsw_configs  = _build_hnsw(data,  n, d, mem_limit) if args.hnsw  else []
    ampi_configs  = build_ampi_configs(data, queries, gt, metric='l2',
                                       data_path=mmap_dir)

    exact_labels = ("Flat L2",) if faiss_configs else ()
    print_table_header()
    rows = run_evaluation(faiss_configs + hnsw_configs + ampi_configs,
                          queries, gt, data, n,
                          exact_labels=exact_labels)

    print()
    save_figures([(label, rows)], family_style=_FAMILY_STYLE, suffix="_gist_large")

    # ── log results ────────────────────────────────────────────────────────────
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    log_dir = Path(__file__).parent / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "gist_large.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha":   git_sha,
        "dataset":   "gist",
        "n":         n,
        "d":         d,
        "n_queries": len(queries),
        "k":         K,
        "configs":   rows,   # list of dicts: label/recall/recall1/recall100/ratio/ms/qps/cands
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  → logged to {log_path}")
