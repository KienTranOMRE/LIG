#!/usr/bin/env python3
"""Debug why chunking causes different results."""

import torch
from signal_lam import joint_star_ig
from signal_lam_batched import joint_star_ig_batched
from lam import load_image_and_model
from utilss import get_device
import time

device = get_device()
print("Loading model...")
model, x, baseline, info = load_image_and_model(device, min_conf=0.70)

# Set seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("\n" + "="*70)
print("TEST 1: Original vs Batched (no chunking, chunk_size=None)")
print("="*70)

torch.manual_seed(42)
print("\nRunning original...")
t0 = time.time()
result_orig = joint_star_ig(model, x, baseline, N=50, G=16, n_alternating=1)
time_orig = time.time() - t0

torch.cuda.empty_cache()
torch.manual_seed(42)
print("Running batched (chunk_size=None)...")
t0 = time.time()
try:
    result_full = joint_star_ig_batched(model, x, baseline, N=50, G=16, n_alternating=1,
                                         chunk_size=None)
    time_full = time.time() - t0

    print(f"\nOriginal Q: {result_orig.Q:.8f}")
    print(f"Batched Q:  {result_full.Q:.8f}")
    print(f"Difference: {abs(result_orig.Q - result_full.Q):.2e}")
    print(f"\nOriginal time: {time_orig:.1f}s")
    print(f"Batched time:  {time_full:.1f}s")
    print(f"Speedup: {time_orig/time_full:.2f}×")

    attr_diff = (result_orig.attributions - result_full.attributions).abs().max().item()
    print(f"Max attr diff: {attr_diff:.2e}")

except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM with chunk_size=None (expected)")
    else:
        raise

print("\n" + "="*70)
print("TEST 2: Batched with different chunk sizes")
print("="*70)

for cs in [1, 2, 4, 8, 16]:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    print(f"\nchunk_size={cs}...")
    try:
        t0 = time.time()
        result_chunk = joint_star_ig_batched(model, x, baseline, N=50, G=16, n_alternating=1,
                                              chunk_size=cs)
        time_chunk = time.time() - t0

        q_diff = abs(result_orig.Q - result_chunk.Q)
        attr_diff = (result_orig.attributions - result_chunk.attributions).abs().max().item()

        print(f"  Q diff: {q_diff:.2e}, attr diff: {attr_diff:.2e}, time: {time_chunk:.1f}s")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  OOM")
        else:
            raise
