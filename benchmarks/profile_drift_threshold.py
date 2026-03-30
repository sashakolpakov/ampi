"""profile_drift_threshold.py — Profile per-cluster fan-axis variance to validate θ_drift.

For each cluster in a built AMPIAffineFanIndex, we compute the leading eigenvector
of the cluster's data covariance (via truncated SVD on the centred points) and
measure the angle between that eigenvector and the nearest fan axis.  The resulting
distribution of "min-angle-to-any-fan-axis" per cluster tells us whether the
current _DRIFT_THETA = 15.0° is a tight or loose trigger.

If the median build-time angle is already close to θ_drift, drift will fire
spuriously; if it is much smaller, θ_drift has headroom.

Usage
  # synthetic Gaussian data (fast, no file needed)
  python profile_drift_threshold.py

  # GIST high-d stress test (requires the file; ~3.6 GB)
  python profile_drift_threshold.py --dataset gist

  # any HDF5 dataset
  python profile_drift_threshold.py --dataset sift
  python profile_drift_threshold.py --dataset fashion

  # cap training set for a quicker run
  python profile_drift_threshold.py --dataset gist --n-train 100000

  # save figure
  python profile_drift_threshold.py --dataset gist --save
"""

import argparse, sys, math, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO      = Path(__file__).parent.parent
_BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_BENCH_DIR))

from ampi import AMPIAffineFanIndex
from ampi.affine_fan import _DRIFT_THETA
from _bench_common import (
    DATA_DIR, FIGURES_DIR, make_gaussian, load_hdf5,
    ensure_datasets,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _leading_eigvec(X: np.ndarray) -> np.ndarray:
    """Return the leading right-singular vector of centred matrix X (float64)."""
    # Randomised SVD via power iteration — cheap for large n, small k=1.
    rng = np.random.default_rng(0)
    d   = X.shape[1]
    v   = rng.standard_normal(d)
    v  /= np.linalg.norm(v)
    for _ in range(20):
        u  = X @ v
        v  = X.T @ u
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            break
        v /= nv
    return v.astype(np.float64)


def _min_angle_to_axes(eigvec: np.ndarray, axes: np.ndarray) -> float:
    """Minimum angle (degrees) between eigvec and any row of axes."""
    cos = np.abs(axes.astype(np.float64) @ eigvec)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos.max())))


def _explained_variance_ratio(X: np.ndarray, eigvec: np.ndarray) -> float:
    """Fraction of total variance captured by the leading eigenvector."""
    proj_var  = float(np.var(X @ eigvec))
    total_var = float(np.mean(np.sum(X ** 2, axis=1)))
    return proj_var / (total_var + 1e-12)


# ── main analysis ─────────────────────────────────────────────────────────────

def profile(data: np.ndarray, nlist: int = None, num_fans: int = 64,
            min_pts: int = 10) -> dict:
    """Build an index and compute per-cluster angle statistics.

    Returns a dict with keys:
      angles          — (nlist,) array of min-angle-to-nearest-fan-axis per cluster
      ev_ratios       — (nlist,) explained-variance ratios for the leading eigvec
      cluster_sizes   — (nlist,) number of points per cluster
      F               — number of fan axes used
      drift_theta     — _DRIFT_THETA constant from affine_fan.py
    """
    n, d = data.shape
    if nlist is None:
        nlist = max(16, int(np.sqrt(n)))

    print(f"  Building AMPIAffineFanIndex  n={n:,}  d={d}  "
          f"nlist={nlist}  F={num_fans}…", flush=True)
    t0  = time.perf_counter()
    idx = AMPIAffineFanIndex(data, nlist=nlist, num_fans=num_fans, seed=0)
    print(f"  Build time: {time.perf_counter() - t0:.1f}s")

    angles    = []
    ev_ratios = []
    sizes     = []

    for c in range(idx.nlist):
        c_idx = idx.cluster_global[c]
        if len(c_idx) < min_pts:
            continue

        pts = data[c_idx].astype(np.float64)
        pts -= idx.centroids[c].astype(np.float64)   # centre

        # Use cluster-local axes if a refresh has already happened, else global.
        axes_c = (idx.cluster_axes[c]
                  if idx.cluster_axes[c] is not None
                  else idx.axes)

        v    = _leading_eigvec(pts)
        ang  = _min_angle_to_axes(v, axes_c)
        evr  = _explained_variance_ratio(pts, v)

        angles.append(ang)
        ev_ratios.append(evr)
        sizes.append(len(c_idx))

    return dict(
        angles        = np.array(angles),
        ev_ratios     = np.array(ev_ratios),
        cluster_sizes = np.array(sizes),
        F             = idx.F,
        drift_theta   = _DRIFT_THETA,
    )


def report(res: dict, dataset_name: str) -> None:
    angles = res["angles"]
    evr    = res["ev_ratios"]
    theta  = res["drift_theta"]
    F      = res["F"]
    n_cl   = len(angles)

    pcts   = np.percentile(angles, [25, 50, 75, 90, 95, 99])
    frac_above = float(np.mean(angles > theta))

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {dataset_name}  —  {n_cl} clusters  F={F}  θ_drift={theta}°")
    print(sep)
    print(f"  Min-angle-to-nearest-axis  (degrees):")
    print(f"    mean  = {angles.mean():.2f}°")
    print(f"    p25   = {pcts[0]:.2f}°   p50 = {pcts[1]:.2f}°   p75 = {pcts[2]:.2f}°")
    print(f"    p90   = {pcts[3]:.2f}°   p95 = {pcts[4]:.2f}°   p99 = {pcts[5]:.2f}°")
    print(f"    max   = {angles.max():.2f}°")
    print(f"  Clusters with angle > θ_drift={theta}°: "
          f"{int(frac_above * n_cl)}/{n_cl}  ({100*frac_above:.1f}%)")
    print(f"\n  Leading-eigvec explained-variance ratio:")
    pev = np.percentile(evr, [25, 50, 75])
    print(f"    p25={pev[0]:.3f}  median={pev[1]:.3f}  p75={pev[2]:.3f}")

    # Interpretation — three regimes based on how median_ang compares to theta
    median_ang = pcts[1]
    median_evr = float(np.median(evr))
    if median_ang > 3 * theta:
        # Random axes are far from all cluster principal directions.
        # This is the normal state at build time with isotropic/random axes:
        # the drift EMA hasn't accumulated anything yet, so _local_refresh has
        # not run.  The threshold θ_drift applies to the *change* in the EMA
        # eigenvector after streaming inserts, not to this build-time baseline.
        iso_note = (f"  (explained-variance median={median_evr:.3f} — "
                    + ("near-isotropic clusters; no strong principal direction)"
                       if median_evr < 0.05 else
                       "clusters have structured variance; axes will adapt quickly)"))
        verdict = (
            f"Build-time angles far exceed θ_drift "
            f"(median {median_ang:.1f}° >> θ_drift={theta}°). "
            f"This is expected: random axes are not aligned with cluster principal "
            f"directions at construction time. θ_drift governs the *displacement* "
            f"covariance EMA after streaming inserts — it does not apply here. "
            f"After the first _local_refresh each cluster's axes adapt to the data, "
            f"and subsequent drift is measured from that adapted baseline.\n"
            + iso_note
        )
    elif median_ang < theta * 0.3:
        verdict = (f"Axes cover the dominant directions well at build time "
                   f"(median {median_ang:.1f}° << θ_drift={theta}°). "
                   f"θ_drift has ample headroom — consider tightening for faster refresh.")
    elif median_ang < theta:
        verdict = (f"Build-time coverage is moderate (median {median_ang:.1f}°, "
                   f"θ_drift={theta}°). Threshold is a reasonable trigger.")
    else:
        verdict = (f"Majority of clusters exceed θ_drift at build time "
                   f"(median {median_ang:.1f}° > θ_drift={theta}°). "
                   f"Drift will fire immediately on every cluster after the first EMA "
                   f"update. Consider raising θ_drift or increasing F.")
    print(f"\n  Assessment: {verdict}")
    print(sep)


def save_figure(res: dict, dataset_name: str, suffix: str = "") -> None:
    angles = res["angles"]
    evr    = res["ev_ratios"]
    theta  = res["drift_theta"]

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fan-axis angle profile — {dataset_name}", fontweight="bold")

    ax1.hist(angles, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.axvline(theta, color="crimson", lw=2, ls="--",
                label=f"θ_drift = {theta}°")
    ax1.axvline(float(np.median(angles)), color="orange", lw=1.5, ls=":",
                label=f"median = {np.median(angles):.1f}°")
    ax1.set_xlabel("Min angle to nearest fan axis (degrees)")
    ax1.set_ylabel("Number of clusters")
    ax1.set_title("Distribution of min-angle-to-axis at build time")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(evr, angles, s=12, alpha=0.5, color="#4C72B0")
    ax2.axhline(theta, color="crimson", lw=2, ls="--",
                label=f"θ_drift = {theta}°")
    ax2.set_xlabel("Leading-eigvec explained-variance ratio")
    ax2.set_ylabel("Min angle to nearest fan axis (degrees)")
    ax2.set_title("Angle vs explained variance per cluster")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    slug = dataset_name.split()[0].lower()
    out  = FIGURES_DIR / f"drift_profile_{slug}{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gauss",
                    choices=["gauss", "mnist", "fashion", "sift", "glove", "gist"],
                    help="Dataset to profile (default: gauss)")
    ap.add_argument("--n-train", type=int, default=None,
                    help="Cap on training vectors (default: use all)")
    ap.add_argument("--nlist", type=int, default=None,
                    help="Override nlist (default: sqrt(n))")
    ap.add_argument("--num-fans", type=int, default=64,
                    help="Number of fan axes F (default: 64)")
    ap.add_argument("--save", action="store_true",
                    help="Save figure to figures/")
    args = ap.parse_args()

    ds = args.dataset
    if ds != "gauss":
        ensure_datasets({ds})

    if ds == "gauss":
        n    = args.n_train or 50_000
        data, _, _ = make_gaussian(n=n, d=128)
        name = f"Gaussian {n//1000}k  d=128"
    elif ds == "mnist":
        data, _, _ = load_hdf5(DATA_DIR / "MNIST/mnist-784-euclidean.hdf5",
                               n_train=args.n_train)
        name = "MNIST 60k  d=784"
    elif ds == "fashion":
        data, _, _ = load_hdf5(DATA_DIR / "fashion-mnist/fashion-mnist-784-euclidean.hdf5",
                               n_train=args.n_train)
        name = "Fashion-MNIST 60k  d=784"
    elif ds == "sift":
        data, _, _ = load_hdf5(DATA_DIR / "sift/sift-128-euclidean.hdf5",
                               n_train=args.n_train)
        name = "SIFT 1M  d=128"
    elif ds == "glove":
        data, _, _ = load_hdf5(DATA_DIR / "glove/glove-100-angular.hdf5",
                               n_train=args.n_train, normalize=True)
        name = "GloVe 1.18M  d=100"
    elif ds == "gist":
        data, _, _ = load_hdf5(DATA_DIR / "gist/gist-960-euclidean.hdf5",
                               n_train=args.n_train)
        name = f"GIST{' '+str(args.n_train//1000)+'k' if args.n_train else ' 1M'}  d=960"

    res = profile(data, nlist=args.nlist, num_fans=args.num_fans)
    report(res, name)

    if args.save:
        save_figure(res, name)
