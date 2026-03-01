"""
Benchmark: AMPI (L2-LSH) vs FAISS FlatL2 vs FAISS IVFFlat
Recall@k for k=1,5,10 across n=10k, 100k, 1M  (d=128, N(0,1) data)

AMPI uses h(x) = floor((u·x + b) / w), b ~ U[0,w) (Datar et al. 2004).
Three operating points are shown for AMPI: fast / medium / accurate.
IVF is shown at three nprobe levels.
"""

import sys
import time
import textwrap
import numpy as np
import faiss

sys.path.insert(0, "ampi/src")
from ampi import AMPIIndex  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────────

DIM = 128
K_VALUES = [1, 5, 10]
NUM_QUERIES = 500
SEED = 42

# AMPI: (label, num_projections, window_size)
AMPI_CONFIGS = [
    ("AMPI fast",    16, 100),
    ("AMPI medium",  16, 500),
    ("AMPI accurate",32, 1000),
]

IVF_NPROBE_FRACTIONS = [0.01, 0.05, 0.20]
IVF_NLIST_RATIO = 0.01


# ── Helpers ──────────────────────────────────────────────────────────────────────

def make_dataset(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def recall_at_k(true_indices, approx_indices, k):
    hits = sum(
        len(set(t[:k].tolist()) & set(a[:k].tolist()))
        for t, a in zip(true_indices, approx_indices)
    )
    return hits / (len(true_indices) * k)


def warmup_and_time(fn, items, warmup=5):
    for x in items[:warmup]:
        fn(x)
    t0 = time.perf_counter()
    for x in items:
        fn(x)
    return (time.perf_counter() - t0) / len(items) * 1000


def cands_per_table(n, w, probes):
    """Expected candidates per hash table lookup (all probed buckets)."""
    from scipy.stats import norm
    half = (2 * probes - 1) * w / 2
    return int(n * (norm.cdf(half) - norm.cdf(-half)))


# ── Benchmark ────────────────────────────────────────────────────────────────────

def run_benchmark(n):
    k_max = max(K_VALUES)
    print(f"\n{'═'*72}")
    print(f"  n={n:,}  d={DIM}  queries={NUM_QUERIES}  seed={SEED}")
    print(f"{'═'*72}")

    data = make_dataset(n, DIM, SEED)
    rng = np.random.default_rng(SEED + 1)
    queries = data[rng.choice(n, NUM_QUERIES, replace=False)]

    # Ground truth
    print("  Ground truth (FlatL2)…", end=" ", flush=True)
    flat = faiss.IndexFlatL2(DIM)
    flat.add(data)
    _, I_true = flat.search(queries, k_max)
    flat_ms = warmup_and_time(lambda q: flat.search(q[None], k_max), queries)
    print(f"done  ({flat_ms:.2f} ms/query)")

    # IVF
    nlist = max(64, int(n * IVF_NLIST_RATIO))
    print(f"  Building IVF (nlist={nlist})…", end=" ", flush=True)
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(DIM)
    ivf = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_L2)
    ivf.train(data); ivf.add(data)
    print(f"{time.perf_counter()-t0:.2f}s")

    # AMPI — build each unique (L,) once
    ampi_built = {}
    for label, L, window in AMPI_CONFIGS:
        key = (L,)
        if key not in ampi_built:
            print(f"  Building AMPI L={L} window={window}…", end=" ", flush=True)
            t0 = time.perf_counter()
            ampi_built[key] = AMPIIndex(data, num_projections=L,
                                        bucket_size=1.0, seed=SEED)
            print(f"{time.perf_counter()-t0:.2f}s")

    # Collect results
    results = {}
    results["Flat (exact)"] = {"I": I_true, "ms": flat_ms, "cands": n}

    for frac in IVF_NPROBE_FRACTIONS:
        nprobe = max(1, int(nlist * frac))
        ivf.nprobe = nprobe
        lbl = f"IVF  np={nprobe}"
        _, I_ivf = ivf.search(queries, k_max)
        ms = warmup_and_time(lambda q: ivf.search(q[None], k_max), queries)
        results[lbl] = {"I": I_ivf, "ms": ms, "cands": nprobe * (n // nlist)}

    for label, L, window in AMPI_CONFIGS:
        idx = ampi_built[(L,)]
        # Count actual unique candidates from a sample query (before k limit)
        all_cands = idx.query_candidates(queries[0], window_size=window)
        actual_cands = len(all_cands)
        rows = [idx.query(q, k=k_max, window_size=window)[2] for q in queries]
        padded = [np.pad(r, (0, max(0, k_max - len(r))), constant_values=-1)
                  for r in rows]
        I_ampi = np.array(padded, dtype=np.int32)
        ms = warmup_and_time(
            lambda q: idx.query(q, k=k_max, window_size=window), queries
        )
        results[label] = {"I": I_ampi, "ms": ms, "cands": actual_cands}

    # Print
    k_hdr = "  ".join(f"R@{k}" for k in K_VALUES)
    hdr = f"  {'Method':<22}  {'ms/q':>6}  {'~cands':>8}  {'%data':>5}  {k_hdr}"
    print()
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for lbl, res in results.items():
        rcols = "  ".join(f"{recall_at_k(I_true, res['I'], k):.3f}"
                          for k in K_VALUES)
        pct = 100 * min(res["cands"], n) / n
        print(f"  {lbl:<22}  {res['ms']:>6.2f}  {res['cands']:>8,}"
              f"  {pct:>4.1f}%  {rcols}")

    return data, queries, I_true, ampi_built, ivf, nlist


def run_degradation_curves(data, queries, I_true, ampi_built, ivf, nlist):
    """Show recall vs candidates for AMPI and FAISS."""
    k_max = max(K_VALUES)
    print(f"\n{'═'*72}")
    print(f"  Degradation curves (recall vs candidates)")
    print(f"{'═'*72}")

    # AMPI: vary window_size for L=16
    idx = ampi_built[(16,)]
    windows = [10, 25, 50, 100, 200, 500, 1000, 2000]
    print("\n  AMPI (L=16):")
    print(f"    {'window':>8}  {'cands':>8}  {'ms/q':>8}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}")
    for w in windows:
        all_cands = idx.query_candidates(queries[0], window_size=w)
        cands = len(all_cands)
        rows = [idx.query(q, k=k_max, window_size=w)[2] for q in queries]
        padded = [np.pad(r, (0, max(0, k_max - len(r))), constant_values=-1) for r in rows]
        I_ampi = np.array(padded, dtype=np.int32)
        ms = warmup_and_time(lambda q: idx.query(q, k=k_max, window_size=w), queries)
        r1 = recall_at_k(I_true, I_ampi, 1)
        r5 = recall_at_k(I_true, I_ampi, 5)
        r10 = recall_at_k(I_true, I_ampi, 10)
        print(f"    {w:>8}  {cands:>8,}  {ms:>8.2f}  {r1:>6.3f}  {r5:>6.3f}  {r10:>6.3f}")

    # FAISS IVF: vary nprobe
    print("\n  FAISS IVF:")
    nprobes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    print(f"    {'nprobe':>8}  {'cands':>8}  {'ms/q':>8}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}")
    for np_ in nprobes:
        if np_ > nlist:
            continue
        ivf.nprobe = np_
        cands = np_ * (data.shape[0] // nlist)
        _, I_ivf = ivf.search(queries, k_max)
        ms = warmup_and_time(lambda q: ivf.search(q[None], k_max), queries)
        r1 = recall_at_k(I_true, I_ivf, 1)
        r5 = recall_at_k(I_true, I_ivf, 5)
        r10 = recall_at_k(I_true, I_ivf, 10)
        print(f"    {np_:>8}  {cands:>8,}  {ms:>8.2f}  {r1:>6.3f}  {r5:>6.3f}  {r10:>6.3f}")


if __name__ == "__main__":
    print(textwrap.dedent("""
        AMPI (L2-LSH) vs FAISS  —  Recall@k  (d=128, N(0,1))
        h(x)=floor((u·x+b)/w), b~U[0,w)  [Datar et al. 2004]
        ~cands = expected unique candidates searched
    """).strip())

    for n in [10_000, 100_000]:
        data, queries, I_true, ampi_built, ivf, nlist = run_benchmark(n)
        run_degradation_curves(data, queries, I_true, ampi_built, ivf, nlist)
