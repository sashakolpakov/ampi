"""download_data.py — fetch ANN-benchmark HDF5 datasets used by benchmark_vs_*.py

All files are hosted by the ann-benchmarks project (https://github.com/erikbern/ann-benchmarks).
Total download size: ~1 GB (SIFT 1M + GloVe dominate).

Usage
  python download_data.py               # download all datasets
  python download_data.py mnist sift    # download specific datasets
  python download_data.py --list        # show dataset info without downloading
"""

import argparse, sys, urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR  = _REPO_ROOT / "data"

_BASE_URL = "http://ann-benchmarks.com"

DATASETS = {
    "mnist": dict(
        url=f"{_BASE_URL}/mnist-784-euclidean.hdf5",
        dest=_DATA_DIR / "MNIST" / "mnist-784-euclidean.hdf5",
        desc="MNIST handwritten digits, 60k × d=784, Euclidean",
    ),
    "fashion": dict(
        url=f"{_BASE_URL}/fashion-mnist-784-euclidean.hdf5",
        dest=_DATA_DIR / "fashion-mnist" / "fashion-mnist-784-euclidean.hdf5",
        desc="Fashion-MNIST, 60k × d=784, Euclidean",
    ),
    "sift": dict(
        url=f"{_BASE_URL}/sift-128-euclidean.hdf5",
        dest=_DATA_DIR / "sift" / "sift-128-euclidean.hdf5",
        desc="SIFT, 1M × d=128, Euclidean  (~350 MB)",
    ),
    "glove": dict(
        url=f"{_BASE_URL}/glove-100-angular.hdf5",
        dest=_DATA_DIR / "glove" / "glove-100-angular.hdf5",
        desc="GloVe Twitter 1.18M × d=100, angular/cosine  (~500 MB)",
    ),
}


def _download(name, info, force=False):
    dest: Path = info["dest"]
    if dest.exists() and not force:
        print(f"  [{name}] already exists, skipping  ({dest})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = info["url"]
    print(f"  [{name}] downloading {url} …")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"\r    [{bar}] {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r    done — saved to {dest}")
    except Exception as e:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        print(f"\r    ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def ensure_datasets(targets=None, force=False):
    """Download any missing dataset files before a benchmark run.

    Parameters
    ----------
    targets : iterable of dataset names, or None for all real datasets.
              "gauss" is silently ignored (synthetic, no file needed).
    force   : if True, re-download even when the file already exists.
    """
    names = set(DATASETS) if targets is None else (set(targets) & set(DATASETS))
    for name in names:
        _download(name, DATASETS[name], force=force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="*", default=list(DATASETS),
                    help="mnist fashion sift glove  (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file already exists")
    ap.add_argument("--list",  action="store_true",
                    help="print dataset info and exit without downloading")
    args = ap.parse_args()

    if args.list:
        for name, info in DATASETS.items():
            status = "OK" if info["dest"].exists() else "missing"
            print(f"  {name:<10}  [{status}]  {info['desc']}")
        sys.exit(0)

    targets = [t for t in args.dataset if t in DATASETS]
    unknown  = [t for t in args.dataset if t not in DATASETS]
    if unknown:
        print(f"Unknown datasets: {unknown}.  Valid: {list(DATASETS)}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {len(targets)} dataset(s) to {_DATA_DIR}/")
    for name in targets:
        _download(name, DATASETS[name], force=args.force)
    print("All done.")
