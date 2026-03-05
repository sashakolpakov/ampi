"""
Smoke test: build small indexes, verify exact NN is found at high recall.
Runs in ~10s on a laptop, no datasets required.
"""
import numpy as np
import faiss

from ampi import AMPIBinaryIndex, AMPIAffineFanIndex

rng  = np.random.default_rng(42)
n, d = 5_000, 64
data = rng.standard_normal((n, d)).astype("float32")
qs   = rng.standard_normal((50, d)).astype("float32")

# Ground truth
flat = faiss.IndexFlatL2(d)
flat.add(data)
_, gt = flat.search(qs, 10)


def recall10(gt, found):
    hits = sum(len(set(g.tolist()) & set(f[:10].tolist())) for g, f in zip(gt, found))
    return hits / (len(gt) * 10)


# ── Binary ────────────────────────────────────────────────────────────────────
idx_b = AMPIBinaryIndex(data, num_projections=64, seed=0)
results = [idx_b.query(q, k=10, window_size=100) for q in qs]
found_b = [r[2] for r in results]
rec_b   = recall10(gt, found_b)
print(f"Binary   recall@10 = {rec_b:.3f}")
assert rec_b >= 0.80, f"Binary recall too low: {rec_b:.3f}"

# ── AffineFan ─────────────────────────────────────────────────────────────────
idx_a = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0)
results = [idx_a.query(q, k=10, window_size=200, probes=5, fan_probes=16) for q in qs]
found_a = [r[2] for r in results]
rec_a   = recall10(gt, found_a)
print(f"AffineFan recall@10 = {rec_a:.3f}")
assert rec_a >= 0.80, f"AffineFan recall too low: {rec_a:.3f}"

print("OK")
