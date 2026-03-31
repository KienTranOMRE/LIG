#!/usr/bin/env python3
"""Test if chunking gives same results as full batching."""

import torch
from signal_lam_batched import joint_star_ig_batched
from lam import load_image_and_model
from utilss import get_device

device = get_device()
print("Loading model...")
model, x, baseline, info = load_image_and_model(device, min_conf=0.70)

# CRITICAL: Set seed for reproducibility
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

print("\nTest with n_alternating=2 (same as validation)")
print("="*70)

# Run with chunk_size=None (full batching)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)
print("\n1. Full batching (chunk_size=None)...")
try:
    result_full = joint_star_ig_batched(
        model, x, baseline, N=50, G=16,
        n_alternating=2, lam=1.0, tau=0.01,
        chunk_size=None
    )
    print(f"   Q = {result_full.Q:.10f}")
    print(f"   Var_ν = {result_full.Var_nu:.10e}")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("   OOM (expected on this GPU)")
        result_full = None
    else:
        raise

# Run with chunk_size=4
torch.manual_seed(12345)  # SAME SEED!
torch.cuda.manual_seed(12345)
torch.cuda.empty_cache()
print("\n2. Chunked (chunk_size=4)...")
result_chunk4 = joint_star_ig_batched(
    model, x, baseline, N=50, G=16,
    n_alternating=2, lam=1.0, tau=0.01,
    chunk_size=4
)
print(f"   Q = {result_chunk4.Q:.10f}")
print(f"   Var_ν = {result_chunk4.Var_nu:.10e}")

# Run with chunk_size=8
torch.manual_seed(12345)  # SAME SEED!
torch.cuda.manual_seed(12345)
torch.cuda.empty_cache()
print("\n3. Chunked (chunk_size=8)...")
result_chunk8 = joint_star_ig_batched(
    model, x, baseline, N=50, G=16,
    n_alternating=2, lam=1.0, tau=0.01,
    chunk_size=8
)
print(f"   Q = {result_chunk8.Q:.10f}")
print(f"   Var_ν = {result_chunk8.Var_nu:.10e}")

# Run with chunk_size=16 (same as full)
torch.manual_seed(12345)  # SAME SEED!
torch.cuda.manual_seed(12345)
torch.cuda.empty_cache()
print("\n4. Chunked (chunk_size=16, should equal None)...")
result_chunk16 = joint_star_ig_batched(
    model, x, baseline, N=50, G=16,
    n_alternating=2, lam=1.0, tau=0.01,
    chunk_size=16
)
print(f"   Q = {result_chunk16.Q:.10f}")
print(f"   Var_ν = {result_chunk16.Var_nu:.10e}")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)

if result_full is not None:
    print(f"\nFull vs chunk=4:")
    print(f"  Q diff: {abs(result_full.Q - result_chunk4.Q):.2e}")
    print(f"  Var_ν diff: {abs(result_full.Var_nu - result_chunk4.Var_nu):.2e}")
    attr_diff = (result_full.attributions - result_chunk4.attributions).abs().max().item()
    print(f"  Max attr diff: {attr_diff:.2e}")

print(f"\nchunk=4 vs chunk=8:")
print(f"  Q diff: {abs(result_chunk4.Q - result_chunk8.Q):.2e}")
print(f"  Var_ν diff: {abs(result_chunk4.Var_nu - result_chunk8.Var_nu):.2e}")
attr_diff = (result_chunk4.attributions - result_chunk8.attributions).abs().max().item()
print(f"  Max attr diff: {attr_diff:.2e}")

print(f"\nchunk=8 vs chunk=16:")
print(f"  Q diff: {abs(result_chunk8.Q - result_chunk16.Q):.2e}")
print(f"  Var_ν diff: {abs(result_chunk8.Var_nu - result_chunk16.Var_nu):.2e}")
attr_diff = (result_chunk8.attributions - result_chunk16.attributions).abs().max().item()
print(f"  Max attr diff: {attr_diff:.2e}")

if result_full is not None:
    print(f"\nFull (None) vs chunk=16 (should be identical):")
    print(f"  Q diff: {abs(result_full.Q - result_chunk16.Q):.2e}")
    print(f"  Var_ν diff: {abs(result_full.Var_nu - result_chunk16.Var_nu):.2e}")
    attr_diff = (result_full.attributions - result_chunk16.attributions).abs().max().item()
    print(f"  Max attr diff: {attr_diff:.2e}")
