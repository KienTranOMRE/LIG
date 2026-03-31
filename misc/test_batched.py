#!/usr/bin/env python3
"""
Quick test script for signal_lam_batched with chunk_size to avoid OOM.

Usage:
    python test_batched.py
"""

import torch
from signal_lam_batched import joint_star_ig_batched
from lam import load_image_and_model
from utilss import get_device

# Load model and image
device = get_device()
print("Loading model and image...")
model, x, baseline, info = load_image_and_model(device, min_conf=0.70)

print(f"\nModel: {info['model']}")
print(f"Class: {info['target_class']} (conf={info['confidence']:.4f})")

# Check GPU memory
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"\nGPU Memory: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")

# Clear cache
torch.cuda.empty_cache()

# Run with chunk_size=4 to avoid OOM
print("\nRunning Joint* with chunk_size=4 (memory-safe)...")
result = joint_star_ig_batched(
    model, x, baseline,
    N=50,
    G=16,
    n_alternating=2,
    lam=1.0,
    tau=0.01,
    mu_iter=300,
    path_iter=10,
    chunk_size=4  # Use chunk_size=4 for 23GB GPU
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Q:        {result.Q:.6f}")
print(f"Var_ν:    {result.Var_nu:.8f}")
print(f"CV²:      {result.CV2:.6f}")
print(f"Time:     {result.elapsed_s:.1f}s")
print("="*60)

print(f"\n✓ Success! No OOM error.")
print(f"Attribution shape: {result.attributions.shape}")
