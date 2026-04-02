"""benchmark_sketch_rerank.py — QPS / recall impact of sketch-based lazy rerank.

Sketch pruning (Bessel lower bound) avoids mmap reads for candidates that are
provably too far away.  This benchmark measures the speedup on GIST 200k (d=960),
the dataset where cold-cache mmap page-fault storms hurt most.

Baseline numbers (main branch, exact rerank):
  AFan K=1 cp=50 fp=32 w=16  →  R@10=0.964  31 QPS  (from BENCHMARKS.md)

Usage:
  python benchmarks/benchmark_sketch_rerank.py
  python benchmarks/benchmark_sketch_rerank.py --n 100000
"""

import sys, time, gc, argparse, os
from pathlib import Path
import numpy as np
import h5py

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ampi import AMPIAffineFanIndex
from ampi.streaming import streaming_build
from ampi.tuner import _brute_knn

DATA_PATH = REPO_ROOT / "data" / "gist"
HDF5_PATH = DATA_PATH / "gist-960-euclidean.hdf5"

K       = 10
K_MAX   = 100
N_Q     = 200
WARMUP  = 10
CHUNK   = 50_000   # GT brute-force chunk size to keep RAM under ~500 MB


def load_gist_memmap(n_train, mmap_dir):
    """Stream GIST into a flat memmap file; compute GT in chunks."""
    os.makedirs(mmap_dir, exist_ok=True)
    dat_path = os.path.join(mmap_dir, "_train_data.dat")

    with h5py.File(HDF5_PATH) as f:
        n_total = f["train"].shape[0]
        d       = f["train"].shape[1]
        n       = min(n_train, n_total)
        queries = f["test"][:N_Q].astype(np.float32)

        if not os.path.exists(dat_path):
            print(f"  Streaming HDF5 → {dat_path} …", flush=True)
            data = np.memmap(dat_path, mode="w+", dtype="float32", shape=(n, d))
            for s in range(0, n, CHUNK):
                e = min(s + CHUNK, n)
                data[s:e] = f["train"][s:e].astype(np.float32)
            data.flush()
        else:
            print(f"  Re-using existing memmap at {dat_path}", flush=True)
            data = np.memmap(dat_path, mode="r", dtype="float32", shape=(n, d))

    # Compute GT via ||q-x||² = ||q||² - 2·q·xᵀ + ||x||²
    # Process data chunks so max alloc ≈ CHUNK × d (one chunk) + N_Q × CHUNK (dot products).
    print(f"  Computing GT via dot-product trick, chunk={CHUNK:,} …", flush=True)

    # Precompute ||x||² in chunks (stays in RAM: n × 4 bytes = ~800 KB at 200k)
    x_sq = np.empty(n, dtype=np.float64)
    for s in range(0, n, CHUNK):
        e         = min(s + CHUNK, n)
        chunk     = np.array(data[s:e], dtype=np.float32)
        x_sq[s:e] = np.einsum("id,id->i", chunk, chunk)
        del chunk

    q_sq = np.einsum("qd,qd->q", queries, queries).astype(np.float64)  # (N_Q,)

    # Running best per query
    best_dists = np.full((N_Q, K_MAX), np.inf, dtype=np.float64)
    best_ids   = np.zeros((N_Q, K_MAX), dtype=np.int64)

    for s in range(0, n, CHUNK):
        e      = min(s + CHUNK, n)
        chunk  = np.array(data[s:e], dtype=np.float32)   # (chunk_sz, d)
        # dot[q, i] = queries[q] · chunk[i]   shape: (N_Q, chunk_sz)
        dots   = queries.astype(np.float64) @ chunk.T.astype(np.float64)
        dists  = q_sq[:, None] - 2.0 * dots + x_sq[None, s:e]  # (N_Q, chunk_sz)
        del dots, chunk

        combined_d = np.concatenate([best_dists, dists], axis=1)
        combined_i = np.concatenate([
            best_ids,
            np.tile(np.arange(s, e, dtype=np.int64), (N_Q, 1))
        ], axis=1)
        part_idx   = np.argpartition(combined_d, K_MAX, axis=1)[:, :K_MAX]
        best_ids   = np.take_along_axis(combined_i, part_idx, axis=1)
        best_dists = np.take_along_axis(combined_d, part_idx, axis=1)
        del combined_d, combined_i, dists
        gc.collect()
        if (e // CHUNK) % 2 == 0:
            print(f"    GT chunk {e:,}/{n:,}", flush=True)

    order = np.argsort(best_dists, axis=1)
    gt    = np.take_along_axis(best_ids, order, axis=1).astype(np.int32)
    return data, queries, gt


def recall_at(gt, indices, k):
    hits = sum(len(set(g[:k].tolist()) & set(r[:k].tolist()))
               for g, r in zip(gt, indices))
    return hits / (len(gt) * k)


def run_config(label, idx, cp, fp, w, queries, gt):
    def qfn(q):
        return idx.query(q, k=K_MAX, probes=cp, fan_probes=fp, window_size=w)

    for q in queries[:WARMUP]:
        qfn(q)

    t0      = time.perf_counter()
    results = [qfn(q) for q in queries]
    ms      = (time.perf_counter() - t0) / len(queries) * 1e3
    qps     = 1e3 / ms

    indices = [r[2] if isinstance(r, tuple) else r for r in results]
    r1   = recall_at(gt, indices, 1)
    r10  = recall_at(gt, indices, K)
    r100 = recall_at(gt, indices, K_MAX)

    print(f"  {label:<46}  R@1={r1:.3f}  R@10={r10:.3f}  R@100={r100:.3f}"
          f"  {qps:6.1f} QPS  {ms:.2f} ms/q", flush=True)
    return dict(label=label, r1=r1, r10=r10, r100=r100, qps=qps, ms=ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()

    mmap_dir = str(DATA_PATH / f"_ampi_large")
    data, queries, gt = load_gist_memmap(args.n, mmap_dir)
    n, d = data.shape

    nlist = max(16, int(2.0 * np.sqrt(n)))
    F     = 32
    alpha = 2.0

    # ── build ──────────────────────────────────────────────────────────────────
    bench_dir = str(DATA_PATH / f"_sketch_bench_{n}")
    print(f"\nBuilding streaming index: n={n:,}  d={d}  nlist={nlist}  F={F}", flush=True)
    t0 = time.perf_counter()
    idx = streaming_build(
        lambda s, e: np.array(data[s:e]),   # copy chunk to RAM for streaming
        n, d,
        nlist=nlist, num_fans=F, cone_top_k=1,
        metric='l2', data_path=bench_dir,
    )
    build_s = time.perf_counter() - t0
    print(f"  Build: {build_s:.1f} s", flush=True)

    try:
        sk = idx._cpp.get_sketch()
        print(f"  Sketch: {sk.shape}  (F={F}, {sk.nbytes/1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"  Sketch unavailable: {e}", flush=True)

    idx.alpha = alpha

    # ── benchmark ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*100}")
    print(f"  GIST {n//1000}k · d={d} · sketch-rerank (F={F})")
    print(f"  Baseline from main: AFan K=1 cp=50 → R@10=0.964  31 QPS")
    print(f"{'─'*100}")

    configs = [
        (5,  F, 12),
        (10, F, 14),
        (20, F, 16),
        (30, F, 16),
        (50, F, 16),
    ]
    rows = []
    for cp, fp, w in configs:
        label = f"AFan K=1 cp={cp:>2} fp={fp} w={w}"
        rows.append(run_config(label, idx, cp, fp, w, queries, gt))

    # ── sketch pruning efficiency ───────────────────────────────────────────────
    print(f"\nSketch pruning analysis (cp=50, fp={F}, w=16):")
    # query_candidates() returns the raw candidate set before exact rerank
    M2 = max(3 * K_MAX, 50)
    cand_sizes = []
    for q in queries[:50]:
        cands = idx.query_candidates(q, window_size=16, probes=50, fan_probes=F)
        cand_sizes.append(len(cands))
    m_mean = float(np.mean(cand_sizes))
    m_max  = max(cand_sizes)
    m_min  = min(cand_sizes)
    prunable = max(0.0, m_mean - M2)
    print(f"  Raw candidates per query: mean={m_mean:.0f}  min={m_min}  max={m_max}")
    print(f"  Sketch M2 (guaranteed exact): {M2}  ({M2/m_mean*100:.1f}% of candidates)")
    print(f"  Candidates subject to sketch pruning: {prunable:.0f} ({prunable/m_mean*100:.1f}%)")
    print(f"  Sketch coverage: F/d = {F}/{d} ≈ {F/d:.3f}  (looser bound → fewer prunings in high-d)")
    print(f"  Note: warm cache after build — mmap reads cheap; cold-cache would widen gap.")


if __name__ == "__main__":
    main()
