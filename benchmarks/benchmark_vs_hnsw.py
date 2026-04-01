"""benchmark_vs_hnsw.py — AMPI vs hnswlib: recall@1/10/100 / QPS across index variants.

Datasets
  gauss   : 10 000 iid N(0,1) vectors, d=128  (synthetic)
  mnist   : data/MNIST/mnist-784-euclidean.hdf5
  fashion : data/fashion-mnist/fashion-mnist-784-euclidean.hdf5
  sift    : data/sift/sift-128-euclidean.hdf5  (1M train / 10k test queries)
  glove   : data/glove/glove-100-angular.hdf5  (L2-normalised → cosine equiv)
  gist    : data/gist/gist-960-euclidean.hdf5  (1M train / 1k test queries, d=960)

Usage
  python benchmark_vs_hnsw.py                      # gauss only (default)
  python benchmark_vs_hnsw.py gauss mnist fashion
  python benchmark_vs_hnsw.py all

Notes
  HNSW's `ef` (dynamic candidate list size at query time) is used as the
  x-axis proxy for "candidates examined".  It is a control parameter, not
  a direct count of distance computations, so the x-axis is not strictly
  comparable to AMPI's candidate counts — treat the QPS vs Recall panel
  as the primary comparison.
"""

import argparse, os, time
import numpy as np
import hnswlib

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
    "HNSW": dict(color="#2ca02c", marker="^", ls="-", lw=2),
}


def run(dataset_name, data, queries, gt, metric='l2'):
    n, d = data.shape
    sep  = "═" * 72
    print(f"\n{sep}")
    print(f"  {dataset_name}  —  n={n:,}  d={d}  queries={len(queries)}  k={K}")
    print(sep)

    space = "cosine" if metric == "cosine" else "l2"
    print(f"  Building HNSW (M=16, ef_construction=200)…", end=" ", flush=True)
    t0   = time.perf_counter()
    hnsw = hnswlib.Index(space=space, dim=d)
    hnsw.init_index(max_elements=n, ef_construction=200, M=16)
    hnsw.add_items(data, num_threads=os.cpu_count())
    print(f"{time.perf_counter() - t0:.2f}s")

    hnsw_configs = []
    for ef in [10, 20, 50, 100, 200, 400, 800]:
        def _hnsw(q, ef=ef):
            hnsw.set_ef(ef)
            labels, _ = hnsw.knn_query(q[None], k=min(K_MAX, n))
            return (None, None, labels[0].astype(np.int32))
        hnsw_configs.append((
            f"HNSW ef={ef}", _hnsw,
            lambda q, ef=ef: ef,   # ef as candidate-count proxy
        ))

    ampi_configs = build_ampi_configs(data, queries, gt, metric)

    print_table_header()
    return run_evaluation(hnsw_configs + ampi_configs, queries, gt, data, n)


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
        # Capped at 200k here; use benchmark_gist_large.py for the full 1M run.
        data, queries, gt = load_hdf5(DATA_DIR / "gist/gist-960-euclidean.hdf5",
                                      n_train=200_000)
        all_results.append(("GIST 200k  d=960",
                             run("GIST 200k  d=960", data, queries, gt)))

    print()
    save_figures(all_results, family_style=_FAMILY_STYLE, suffix="_vs_hnsw")
