"""benchmark_vs_faiss.py — AMPI vs FAISS: recall@1/10/100 / QPS across index variants.

Datasets
  gauss   : 10 000 iid N(0,1) vectors, d=128  (synthetic)
  mnist   : data/MNIST/mnist-784-euclidean.hdf5
  fashion : data/fashion-mnist/fashion-mnist-784-euclidean.hdf5
  sift    : data/sift/sift-128-euclidean.hdf5  (1M train / 10k test queries)
  glove   : data/glove/glove-100-angular.hdf5  (L2-normalised → cosine equiv)
  gist    : data/gist/gist-960-euclidean.hdf5  (1M train / 1k test queries, d=960)

Usage
  python benchmark_vs_faiss.py                      # gauss only (default)
  python benchmark_vs_faiss.py gauss mnist fashion
  python benchmark_vs_faiss.py all
"""

import argparse
import numpy as np
import faiss

from _bench_common import (
    K, K_MAX, DATA_DIR,
    make_gaussian, load_hdf5,
    print_table_header, run_evaluation,
    build_ampi_configs,
    FAMILY_STYLE_BASE, save_figures,
    ensure_datasets,
)

_FAMILY_STYLE = {
    **FAMILY_STYLE_BASE,
    "IVF": dict(color="#1f77b4", marker="o", ls="-", lw=2),
}


_FAISS_RAM_LIMIT = 1.5e9   # 1.5 GB — skip FAISS build above this threshold


def _try_build_faiss(data):
    """Build FAISS Flat + IVF indexes; return (flat, ivf, nlist) or None on OOM."""
    n, d    = data.shape
    est_ram = n * d * 4          # float32 data copy inside FAISS
    if est_ram > _FAISS_RAM_LIMIT:
        gb = est_ram / 1e9
        print(f"  [skip FAISS] estimated RAM {gb:.1f} GB > {_FAISS_RAM_LIMIT/1e9:.1f} GB limit")
        return None

    try:
        flat = faiss.IndexFlatL2(d)
        flat.add(data)

        nlist = max(16, int(np.sqrt(n)))
        ivf   = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        ivf.train(data)
        ivf.add(data)
        return flat, ivf, nlist
    except (MemoryError, RuntimeError) as exc:
        print(f"  [skip FAISS] OOM during build: {exc}")
        return None


def run(dataset_name, data, queries, gt, metric='l2', data_path=None):
    n, d = data.shape
    sep  = "═" * 72
    print(f"\n{sep}")
    print(f"  {dataset_name}  —  n={n:,}  d={d}  queries={len(queries)}  k={K}")
    print(sep)

    faiss_result = _try_build_faiss(data)

    faiss_configs = []
    if faiss_result is not None:
        flat, ivf, nlist = faiss_result
        faiss_configs.append((
            "Flat L2",
            lambda q: (None, None, flat.search(q[None], K_MAX)[1][0]),
            lambda q: n,
        ))
        for nprobe in [1, 5, 10, 25, 50]:
            if nprobe > nlist:
                continue
            def _ivf(q, p=nprobe):
                ivf.nprobe = p
                return (None, None, ivf.search(q[None], K_MAX)[1][0])
            faiss_configs.append((
                f"IVF nprobe={nprobe}", _ivf,
                lambda q, p=nprobe: p * (n // nlist),
            ))

    ampi_configs = build_ampi_configs(data, queries, gt, metric, data_path=data_path)

    exact_labels = ("Flat L2",) if faiss_result is not None else ()
    print_table_header()
    return run_evaluation(
        faiss_configs + ampi_configs,
        queries, gt, data, n,
        exact_labels=exact_labels,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="*", default=["gauss"],
                    help="gauss mnist fashion sift glove all  (default: gauss)")
    ap.add_argument("--force", action="store_true",
                    help="re-download dataset files even if they already exist")
    args    = ap.parse_args()
    targets = set(args.dataset)
    if "all" in targets:
        targets = {"gauss", "mnist", "fashion", "sift", "glove", "gist"}

    ensure_datasets(targets, force=args.force)
    all_results = []

    if "gauss" in targets:
        data, queries, gt = make_gaussian(n=10_000, d=128)
        all_results.append(("Gaussian 10k  d=128",
                             run("Gaussian 10k  d=128", data, queries, gt)))

    if "mnist" in targets:
        data, queries, gt = load_hdf5(DATA_DIR / "MNIST/mnist-784-euclidean.hdf5")
        all_results.append(("MNIST 60k  d=784",
                             run("MNIST 60k  d=784", data, queries, gt)))

    if "fashion" in targets:
        data, queries, gt = load_hdf5(DATA_DIR / "fashion-mnist/fashion-mnist-784-euclidean.hdf5")
        all_results.append(("Fashion-MNIST 60k  d=784",
                             run("Fashion-MNIST 60k  d=784", data, queries, gt)))

    if "sift" in targets:
        data, queries, gt = load_hdf5(DATA_DIR / "sift/sift-128-euclidean.hdf5")
        all_results.append(("SIFT 1M  d=128",
                             run("SIFT 1M  d=128", data, queries, gt)))

    if "glove" in targets:
        data, queries, gt = load_hdf5(DATA_DIR / "glove/glove-100-angular.hdf5", normalize=True)
        all_results.append(("GloVe 1.18M  d=100",
                             run("GloVe 1.18M  d=100", data, queries, gt, metric='cosine')))

    if "gist" in targets:
        # Full 1M: GT from HDF5 (no brute-force alloc); training data loaded via
        # mmap so the OS pages in on demand; AMPI C++ data buffer also mmap-backed.
        # Peak RSS should stay ~4-5 GB on an 8 GB machine.
        import os
        gist_mmap_dir = str(DATA_DIR / "gist" / "_ampi_mmap")
        os.makedirs(gist_mmap_dir, exist_ok=True)
        data, queries, gt = load_hdf5(DATA_DIR / "gist/gist-960-euclidean.hdf5",
                                      mmap_dir=gist_mmap_dir)
        all_results.append(("GIST 1M  d=960",
                             run("GIST 1M  d=960", data, queries, gt,
                                 data_path=gist_mmap_dir)))

    print()
    save_figures(all_results, family_style=_FAMILY_STYLE, suffix="_vs_faiss")
