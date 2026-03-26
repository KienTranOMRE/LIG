"""
unified_ig.py — Unified Integrated Gradients Framework (v3-batched)
=====================================================================

All IG variants minimise the same objective — the Snell's law conservation
violation:

    min_{γ,μ}  Var_ν(φ)  =  Σ_k ν_k (φ_k − φ̄_ν)²

where:
    φ_k = d_k / Δf_k         step fidelity
    ν_k = μ_k Δf_k² / Σ …   effective measure

Derived quantities reported alongside:
    CV²(φ)  = Var_ν(φ) / φ̄²         (scale-free)
    𝒬       = 1 / (1 + CV²(φ))      (quality score, 1 = perfect)

Methods and what they optimise:
    IG           — nothing           (straight line, uniform μ)
    IDGI         — μ heuristic       (straight line, μ_k ∝ |Δf_k|)
    Guided IG    — γ heuristic       (low-grad-first path, uniform μ)
    μ-Optimised  — μ optimal         (straight line, min Var_ν(φ))
    Joint        — γ + μ optimal     (alternating minimisation)

Perf changes vs v3:
    FIX 1 — _straight_line_pass batches all N interpolation points into one
            forward and one backward call (was N sequential calls).
    FIX 2 — optimize_path._var_of batches all N path points into one forward
            and one backward call (was N sequential calls per _var_of,
            called ~(1 + N) times per iteration).

Usage:
    python unified_ig.py                        # single run
    python unified_ig.py --viz --viz-fidelity   # with plots
    python unified_ig.py --json results.json    # export

Requirements: torch >= 2.0, torchvision
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from utilss import (
    get_device, AttributionResult, StepInfo, InsDelScores,
    compute_Var_nu, compute_CV2, compute_Q, compute_all_metrics,
    compute_insertion_deletion, run_insertion_deletion,
    visualize_step_fidelity, visualize_insertion_deletion,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  MODEL WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

class ClassLogitModel(nn.Module):
    """Wrap a classifier → scalar logit for a target class.  Shape: (B,)."""

    def __init__(self, backbone: nn.Module, target_class: int):
        super().__init__()
        self.backbone = backbone
        self.target_class = target_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, self.target_class]


# ═════════════════════════════════════════════════════════════════════════════
# §2  GRADIENT UTILITIES  (fused forward + backward)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_scalar(model: nn.Module, x: torch.Tensor) -> float:
    return float(model(x).squeeze())


@torch.no_grad()
def _forward_batch(model: nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    """f(x) for a batch.  Returns (B,) tensor on same device."""
    return model(x_batch)


def _forward_and_gradient(model: nn.Module, x: torch.Tensor
                          ) -> tuple[float, torch.Tensor]:
    """f(x) and ∇f(x) in ONE backward pass."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        out = model(x_in).sum()
        f_val = float(out)
        out.backward()
    return f_val, x_in.grad.detach()


def _forward_and_gradient_batch(model: nn.Module, x_batch: torch.Tensor
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched f(x) and ∇_x f(x).

    Args:
        x_batch: (B, C, H, W)

    Returns:
        f_vals: (B,) tensor of scalar outputs
        grads:  (B, C, H, W) tensor of per-sample gradients

    Uses torch.vmap-style trick: we sum all outputs and backward once,
    but because each output depends only on its own input row the
    cross-gradients are zero and x_in.grad gives per-sample gradients.
    """
    B = x_batch.shape[0]
    with torch.enable_grad():
        x_in = x_batch.detach().clone().requires_grad_(True)
        model.zero_grad()
        # model returns (B,) — one scalar per sample
        outs = model(x_in)          # (B,)
        f_vals = outs.detach()      # (B,)
        outs.sum().backward()
    return f_vals, x_in.grad.detach()


def _gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """∇f(x) only (when f value already known)."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        model(x_in).sum().backward()
    return x_in.grad.detach()


def _gradient_batch(model: nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    """Batched ∇f(x).  Returns (B, C, H, W)."""
    with torch.enable_grad():
        x_in = x_batch.detach().clone().requires_grad_(True)
        model.zero_grad()
        model(x_in).sum().backward()
    return x_in.grad.detach()


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


# ═════════════════════════════════════════════════════════════════════════════
# §3  STEP DIAGNOSTICS BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _rescale(attr: torch.Tensor, target: float) -> torch.Tensor:
    s = attr.sum().item()
    return attr * (target / s) if abs(s) > 1e-12 else attr


def _build_steps(d_list, df_list, f_vals, gnorms, mu, N) -> list[StepInfo]:
    steps = []
    for k in range(N):
        dk, dfk = d_list[k], df_list[k]
        rk = dfk - dk
        phik = dk / dfk if abs(dfk) > 1e-12 else 1.0
        steps.append(StepInfo(
            t=k / N, f=f_vals[k], d_k=dk, delta_f_k=dfk,
            r_k=rk, phi_k=phik, grad_norm=gnorms[k], mu_k=float(mu[k]),
        ))
    return steps


def _pack_result(name, attr, d_list, df_list, f_vals, gnorms, mu, N,
                 t0, Q_history=None) -> AttributionResult:
    """Build AttributionResult with all three metrics in one pass."""
    device = attr.device
    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    var_nu, cv2, Q = compute_all_metrics(d_arr, df_arr, mu)
    steps = _build_steps(d_list, df_list, f_vals, gnorms, mu, N)
    return AttributionResult(
        name=name, attributions=attr, Q=Q, CV2=cv2, Var_nu=var_nu,
        steps=steps, Q_history=Q_history or [], elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §4  STRAIGHT-LINE PASS  (shared by IG, IDGI, μ-Optimised)
#
#     FIX 1: batch all N interpolation points into ONE forward + ONE backward.
#     Old cost:  N sequential forward+backward = 2N model calls
#     New cost:  1 batched forward+backward     = 2 model calls  (+ 2 scalar)
# ═════════════════════════════════════════════════════════════════════════════

def _straight_line_pass(model: nn.Module, x: torch.Tensor,
                        baseline: torch.Tensor, N: int,
                        fwd_batch_size: int = 0):
    """
    Evaluate f and ∇f at N uniformly-spaced points along the straight line.

    FIX 1: All N points are stacked into a single (N, C, H, W) batch and
    processed in one forward+backward call (or chunked if fwd_batch_size > 0
    to limit GPU memory).

    Returns: (delta_x, target, grads, d_list, df_list, f_vals, gnorms)
        grads   : list of N gradient tensors  (each (1, C, H, W))
        d_list  : list of N floats (d_k = ∇f·Δγ_k)
        df_list : list of N floats (Δf_k)
        f_vals  : list of N+1 floats (f at each γ point, plus f(x))
        gnorms  : list of N floats (‖∇f‖)
    """
    delta_x = x - baseline
    step = delta_x / N                # (1, C, H, W)

    # Endpoints — scalar, cheap
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    # Build batch of N interpolation points: γ_k = baseline + (k/N) * delta_x
    # alphas shape (N, 1, 1, 1) for broadcasting
    alphas = torch.arange(N, device=x.device, dtype=x.dtype).view(N, 1, 1, 1) / N
    gamma_batch = baseline + alphas * delta_x       # (N, C, H, W)

    # ── Batched forward + backward ──
    if fwd_batch_size <= 0 or fwd_batch_size >= N:
        # Single shot
        f_batch, grad_batch = _forward_and_gradient_batch(model, gamma_batch)
    else:
        # Chunked to limit VRAM
        f_chunks, g_chunks = [], []
        for i0 in range(0, N, fwd_batch_size):
            i1 = min(i0 + fwd_batch_size, N)
            fb, gb = _forward_and_gradient_batch(model, gamma_batch[i0:i1])
            f_chunks.append(fb)
            g_chunks.append(gb)
        f_batch = torch.cat(f_chunks, dim=0)        # (N,)
        grad_batch = torch.cat(g_chunks, dim=0)      # (N, C, H, W)

    # ── Unpack results ──
    # gamma_batch has N points: γ_0=baseline, γ_1, ..., γ_{N-1}
    # We need f at N+1 points: γ_0, γ_1, ..., γ_{N-1}, γ_N=x
    # f_batch gives f(γ_0) through f(γ_{N-1}), we add f(x) at the end
    f_vals_inner = f_batch.tolist()                  # N floats: f(γ_0)..f(γ_{N-1})
    f_vals = f_vals_inner + [f_x]                    # N+1 floats: f(γ_0)..f(γ_N=x)

    # d_k = ∇f(γ_k) · step,  for each k = 0..N-1
    # This is the gradient-predicted change for the step γ_k → γ_{k+1}
    d_tensor = (grad_batch * step).view(N, -1).sum(dim=1)   # (N,)
    d_list = d_tensor.tolist()

    # Δf_k = f(γ_{k+1}) - f(γ_k),  for each k = 0..N-1
    # Now properly aligned: d[k] and df[k] both refer to the step γ_k → γ_{k+1}
    df_list = [f_vals[k + 1] - f_vals[k] for k in range(N)]

    # grad norms
    gnorms = grad_batch.view(N, -1).norm(dim=1).tolist()     # N floats

    # grads as list of (1, C, H, W) — clone to avoid shared-memory bugs
    grads = [grad_batch[k:k+1].clone() for k in range(N)]

    return delta_x, target, grads, d_list, df_list, f_vals, gnorms


# ═════════════════════════════════════════════════════════════════════════════
# §5  STANDARD IG
# ═════════════════════════════════════════════════════════════════════════════

def standard_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                N: int = 50) -> AttributionResult:
    """Standard IG (Sundararajan et al., 2017).  No optimisation."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    # grads[k] is (1, C, H, W) — stack and mean
    grad_sum = torch.cat(grads, dim=0).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = delta_x * grad_sum / N
    mu = torch.full((N,), 1.0 / N, device=x.device)

    return _pack_result("IG", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §6  IDGI
# ═════════════════════════════════════════════════════════════════════════════

def idgi(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
         N: int = 50) -> AttributionResult:
    """IDGI (Sikdar et al., 2021).  μ_k ∝ |Δf_k| on straight line."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    df_arr = torch.tensor(df_list, device=x.device)
    weights = df_arr.abs()
    w_sum = weights.sum()
    mu = weights / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0/N, device=x.device)

    # Weighted gradient sum — grads[k] is (1, C, H, W)
    grad_stack = torch.cat(grads, dim=0)               # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)                         # (N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = _rescale(delta_x * wg, target)

    return _pack_result("IDGI", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §7  GUIDED IG
# ═════════════════════════════════════════════════════════════════════════════

def guided_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
              N: int = 50) -> AttributionResult:
    """Guided IG (Kapishnikov et al., 2021).  Low-grad-first path, uniform μ.

    NOTE: This method is inherently sequential — each step depends on the
    previous one — so it cannot be batched like the straight-line methods.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    remaining = delta_x.clone()
    current = baseline.clone()
    gamma_pts = [current.clone()]
    grad_list, d_list, df_list, gnorms = [], [], [], []
    f_vals = [f_bl]

    for k in range(N):
        f_k, grad_k = _forward_and_gradient(model, current)
        grad_list.append(grad_k)
        gnorms.append(float(grad_k.norm()))

        # Inverse-gradient weighting: small |grad| → move more
        abs_g = grad_k.abs() + 1e-8
        inv_w = 1.0 / abs_g
        frac = inv_w / inv_w.sum()
        remaining_steps = N - k

        raw_step = remaining.abs() * frac * remaining_steps * remaining.numel()
        step = remaining.sign() * torch.minimum(raw_step, remaining.abs())

        next_pt = current + step
        f_k1 = _forward_scalar(model, next_pt)

        d_list.append(_dot(grad_k, step))
        df_list.append(f_k1 - f_k)
        f_vals.append(f_k1)

        remaining = remaining - step
        current = next_pt
        gamma_pts.append(current.clone())

    # Attribution
    attr = torch.zeros_like(x)
    for k in range(N):
        attr += grad_list[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    mu = torch.full((N,), 1.0 / N, device=device)
    return _pack_result("Guided IG", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §8  μ-OPTIMISATION  (Phase 1)
#
#     Objective aligned with paper:  min  Var_ν(φ) + τ·H(μ)
#     φ and Δf² are constants → hoisted out of Adam loop
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu(d: torch.Tensor, delta_f: torch.Tensor,
                tau: float = 0.01, n_iter: int = 200,
                lr: float = 0.05) -> torch.Tensor:
    """
    Find μ minimising  CV²(φ) + τ·Σ μ_k log μ_k.

    Uses CV²(φ) = Var_ν(φ) / φ̄² as objective (scale-aware).
    Adam on softmax logits → unconstrained optimisation on the simplex.
    """
    device = d.device
    N = d.shape[0]

    # Constants w.r.t. μ — hoist out of loop
    valid = delta_f.abs() > 1e-12
    safe = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe, torch.ones_like(d)).detach()
    df2 = (delta_f ** 2).detach()

    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mu = torch.softmax(logits, dim=0)

        # Effective measure ν_k = μ_k Δf_k², normalised
        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        w = nu / nu_sum

        mean_phi = (w * phi).sum()
        var_phi = (w * (phi - mean_phi) ** 2).sum()

        # CV²(φ) = Var_ν(φ) / φ̄² — scale-aware, prevents φ̄ → 0
        cv2 = var_phi / (mean_phi ** 2 + 1e-15)

        entropy = (mu * torch.log(mu + 1e-15)).sum()
        loss = cv2 + tau * entropy

        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §9  μ-OPTIMISED IG
# ═════════════════════════════════════════════════════════════════════════════

def mu_optimized_ig(model: nn.Module, x: torch.Tensor,
                    baseline: torch.Tensor, N: int = 50,
                    tau: float = 0.005, n_iter: int = 300
                    ) -> AttributionResult:
    """Straight line + optimal μ.  Cost = standard IG + O(N) arithmetic."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)
    mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=n_iter)

    # Weighted gradient sum
    grad_stack = torch.cat(grads, dim=0)               # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = _rescale(delta_x * wg, target)

    return _pack_result("μ-Optimized", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §10  PATH OPTIMISATION  (Phase 2)
#
#     Spatial grouping for images + softmax velocity scheduling.
#     Objective: Var_ν(φ), consistent with §8.
#
#     FIX 2: _var_of now batches all N path points into one forward +
#            one backward call.
#     Old cost per _var_of:  2N sequential model calls
#     New cost per _var_of:  2 batched model calls
# ═════════════════════════════════════════════════════════════════════════════

_group_cache: dict = {}   # memoised spatial groups


def _build_spatial_groups(model, x, baseline, G=16, patch_size=14):
    """Assign pixels to G groups by gradient importance.  Cached."""
    key = (x.data_ptr(), baseline.data_ptr(), G, patch_size)
    if key in _group_cache:
        return _group_cache[key]

    device = x.device
    _, C, H, W = x.shape
    delta_x = x - baseline
    mid = baseline + 0.5 * delta_x
    grad_mid = _gradient(model, mid)
    importance = (grad_mid * delta_x).abs().sum(dim=1, keepdim=True)

    n_rows = (H + patch_size - 1) // patch_size
    n_cols = (W + patch_size - 1) // patch_size
    n_patches = n_rows * n_cols

    patch_imp = torch.zeros(n_patches, device=device)
    patch_map = torch.zeros(1, 1, H, W, dtype=torch.long, device=device)

    for r in range(n_rows):
        for c in range(n_cols):
            pid = r * n_cols + c
            r0, r1 = r * patch_size, min((r + 1) * patch_size, H)
            c0, c1 = c * patch_size, min((c + 1) * patch_size, W)
            patch_map[0, 0, r0:r1, c0:c1] = pid
            patch_imp[pid] = importance[0, 0, r0:r1, c0:c1].mean()

    order = torch.argsort(patch_imp)
    p2g = torch.zeros(n_patches, dtype=torch.long, device=device)
    per_grp = n_patches // G
    for g in range(G):
        lo = g * per_grp
        hi = (g + 1) * per_grp if g < G - 1 else n_patches
        p2g[order[lo:hi]] = g

    gmap = p2g[patch_map.flatten()].view(1, 1, H, W)
    _group_cache[key] = gmap
    return gmap


def _build_path_2d(baseline, delta_x, V, group_map, N):
    """Path from grouped velocity schedule V (G, N)."""
    G = V.shape[0]
    gamma = [baseline.clone()]
    v_sums = V.sum(dim=1, keepdim=True).clamp(min=1e-12)
    for k in range(N):
        step = torch.zeros_like(baseline)
        for g in range(G):
            mask = (group_map == g).expand_as(baseline)
            step[mask] = delta_x[mask] * (V[g, k] / v_sums[g, 0])
        gamma.append(gamma[-1] + step)
    return gamma


def _eval_path_batched(model, gamma_pts, N, device):
    """
    FIX 2 — core helper: evaluate d_k, Δf_k for a path in batched calls.

    Given gamma_pts (list of N+1 tensors, each (1, C, H, W)):
      1. Batch all N+1 points → one forward call  → f_vals (N+1,)
      2. Batch the first N points → one backward call → grads (N, C, H, W)
      3. Compute d_k = grad_k · (γ_{k+1} - γ_k) and Δf_k = f_{k+1} - f_k

    Returns: (d_vec, df_vec)  both (N,) tensors
    """
    # Stack all N+1 points → (N+1, C, H, W)
    all_pts = torch.cat(gamma_pts, dim=0)               # (N+1, C, H, W)

    # Forward all points in one call
    with torch.no_grad():
        f_all = model(all_pts)                           # (N+1,)

    # Backward only the first N points (we need grads at γ_0 .. γ_{N-1})
    pts_N = all_pts[:N]                                  # (N, C, H, W)
    grads_N = _gradient_batch(model, pts_N)              # (N, C, H, W)

    # Steps: Δγ_k = γ_{k+1} - γ_k
    steps = all_pts[1:] - all_pts[:N]                    # (N, C, H, W)

    # d_k = ∇f(γ_k) · Δγ_k   (per-sample dot product)
    d_vec = (grads_N * steps).view(N, -1).sum(dim=1)     # (N,)

    # Δf_k = f(γ_{k+1}) - f(γ_k)
    df_vec = f_all[1:] - f_all[:N]                       # (N,)

    return d_vec, df_vec


def _compute_path_obj(d_v: torch.Tensor, df_v: torch.Tensor,
                      mu: torch.Tensor) -> float:
    """
    Path objective: MSE_ν(φ, 1) = Σ ν_k (φ_k - 1)².

    This equals Var_ν(φ) + (φ̄_ν - 1)² — variance plus bias-squared.

    Why not just Var_ν?
        Path changes both d_k and Δf_k, so Var_ν = 0 is achievable by making
        all φ_k = c for any constant c (even c = -0.6).

    Why not Q (Cauchy-Schwarz)?
        Q measures proportionality:  Q = 1 whenever d = c·Δf for any c ≠ 0.
        A path with φ_k = -0.6 everywhere has Q = 1 but is terrible.

    MSE around φ = 1 cannot be gamed: the only minimum is φ_k = 1 for all
    steps with ν_k > 0, which is exactly the conservation law (§4.2).

    For μ-optimisation (§8) the bias term is constant (path is fixed) so
    minimising Var_ν is equivalent.  For path optimisation the bias varies,
    so we need the full MSE.
    """
    valid = df_v.abs() > 1e-12
    safe_df = torch.where(valid, df_v, torch.ones_like(df_v))
    phi = torch.where(valid, d_v / safe_df, torch.ones_like(d_v))

    nu = mu * df_v ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return 0.0

    nu = nu / nu_sum
    # MSE_ν(φ, 1) = Σ ν_k (φ_k - 1)²
    mse = float((nu * (phi - 1.0) ** 2).sum())
    return mse


def optimize_path(model, x, baseline, mu, N=50, G=16, patch_size=14,
                  n_iter=15, lr=0.08):
    """
    Optimise path via grouped spatial velocity scheduling.

    Matches the proven v2 approach:
      - V is clamped (≥ 0.01), not softmax-parameterised
      - Objective is CV²(φ) (scale-aware, proven stable)
      - Stochastic FD: one random time step per group per iteration
        (reduces cost from O(G·N) to O(G) evaluations per iteration)

    FIX 2: _eval_of uses _eval_path_batched (2 model calls instead of 2N).
    """
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    # Initialise: uniform velocity = straight line
    V = torch.ones(G, N, device=device)
    best_cv2 = float("inf")
    best_V = V.clone()

    def _cv2_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return compute_CV2(d_v, df_v, mu)

    eps = 0.05
    for it in range(n_iter):
        cv2 = _cv2_of(V)
        if cv2 < best_cv2:
            best_cv2 = cv2
            best_V = V.clone()

        # Stochastic FD: perturb one random time step per group
        grad_V = torch.zeros_like(V)
        for g in range(G):
            k = torch.randint(0, N, (1,)).item()
            V[g, k] += eps
            cv2_plus = _cv2_of(V)
            grad_V[g, k] = (cv2_plus - cv2) / eps
            V[g, k] -= eps

        V = V - lr * grad_V
        V = torch.clamp(V, min=0.01)

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)


# ═════════════════════════════════════════════════════════════════════════════
# §11  JOINT OPTIMISATION  (alternating γ + μ)
# ═════════════════════════════════════════════════════════════════════════════

def joint_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
             N: int = 50, G: int = 16, patch_size: int = 14,
             n_alternating: int = 2, tau: float = 0.005,
             mu_iter: int = 300, path_iter: int = 10,
             ) -> AttributionResult:
    """
    Joint optimisation: alternating min of CV²(φ) over (γ, μ).

    Regression guard: path optimisation only takes effect if it
    improves Q over the current best.  This guarantees Joint ≥ μ-Optimized.

    Uses batched evaluation for path assessment.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    # Initialise: straight line, uniform μ
    gamma_pts = [baseline + (k / N) * delta_x for k in range(N + 1)]
    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history = []

    # Track the best state seen so far
    best_Q = -1.0
    best_gamma_pts = gamma_pts
    best_mu = mu
    best_d_list = []
    best_df_list = []
    best_f_vals = []
    best_gnorms = []
    best_grads = []

    def _evaluate_path(gp, mu_vec):
        """Batched evaluation of a path. Returns (d_list, df_list, f_vals, gnorms, grads, d_arr, df_arr)."""
        ap = torch.cat(gp, dim=0)
        with torch.no_grad():
            fa = model(ap)
        fv = fa.tolist()
        pn = ap[:N]
        gb = _gradient_batch(model, pn)
        sb = ap[1:] - ap[:N]
        dt = (gb * sb).view(N, -1).sum(dim=1)
        dl = dt.tolist()
        dfl = (fa[1:] - fa[:N]).tolist()
        gn = gb.view(N, -1).norm(dim=1).tolist()
        # Clone each slice so it doesn't share memory with gb
        gr = [gb[k:k+1].clone() for k in range(N)]
        da = torch.tensor(dl, device=device)
        dfa = torch.tensor(dfl, device=device)
        return dl, dfl, fv, gn, gr, da, dfa

    for s in range(n_alternating):
        # ── Evaluate current path ──
        d_list, df_list, f_vals, gnorms, grads, d_arr, df_arr = \
            _evaluate_path(gamma_pts, mu)

        # Phase 1: optimise μ
        mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=mu_iter)
        var_mu, cv2_mu, Q_mu = compute_all_metrics(d_arr, df_arr, mu)

        # Update best if improved
        if Q_mu > best_Q:
            best_Q = Q_mu
            best_gamma_pts = gamma_pts
            best_mu = mu.clone()
            best_d_list, best_df_list = d_list, df_list
            best_f_vals, best_gnorms, best_grads = f_vals, gnorms, grads

        # Phase 2: optimise path (skip on last iteration)
        Q_path = Q_mu
        if s < n_alternating - 1:
            new_gamma_pts = optimize_path(
                model, x, baseline, mu, N=N, G=G,
                patch_size=patch_size, n_iter=path_iter)

            # Evaluate the new path
            new_d_list, new_df_list, new_f_vals, new_gnorms, new_grads, \
                new_d_arr, new_df_arr = _evaluate_path(new_gamma_pts, mu)
            _, _, Q_new_path = compute_all_metrics(new_d_arr, new_df_arr, mu)
            Q_path = Q_new_path

            # Regression guard: only accept if better
            if Q_new_path > best_Q:
                gamma_pts = new_gamma_pts
                d_list, df_list = new_d_list, new_df_list
                f_vals, gnorms, grads = new_f_vals, new_gnorms, new_grads
                d_arr, df_arr = new_d_arr, new_df_arr

                best_Q = Q_new_path
                best_gamma_pts = new_gamma_pts
                best_mu = mu.clone()
                best_d_list, best_df_list = new_d_list, new_df_list
                best_f_vals, best_gnorms, best_grads = new_f_vals, new_gnorms, new_grads
            else:
                # Revert to best known state
                gamma_pts = best_gamma_pts
                mu = best_mu.clone()

        Q_history.append({
            "iteration": s,
            "Q_after_mu": float(Q_mu),
            "Q_after_path": float(Q_path),
            "best_Q": float(best_Q),
        })

    # Use the best state for final attributions
    gamma_pts = best_gamma_pts
    mu = best_mu
    grads = best_grads

    attr = torch.zeros_like(x)
    for k in range(N):
        attr += mu[k] * grads[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    return _pack_result("Joint", attr, best_d_list, best_df_list,
                        best_f_vals, best_gnorms, mu, N, t0, Q_history)


# ═════════════════════════════════════════════════════════════════════════════
# §12  IMAGE LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_image_and_model(device: torch.device, min_conf: float = 0.70):
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x, pc, cf = None, None, None
    source = "none"

    # Try local images
    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                       os.path.expanduser("~/sample_imagenet1k")]:
        if not os.path.isdir(sample_dir):
            continue
        try:
            from PIL import Image
            import random
            jpegs = sorted([f for f in os.listdir(sample_dir)
                            if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            random.shuffle(jpegs)
            print(f"Found {sample_dir} ({len(jpegs)} images)")
            for fname in jpegs:
                try:
                    img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
                except Exception:
                    continue
                xc = tf(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(backbone(xc), dim=-1)
                    c, pr = p[0].max(0)
                if c.item() >= min_conf:
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"{sample_dir}/{fname}"
                    print(f"  ✓ {fname} → class={pc}, conf={cf:.4f}")
                    break
        except Exception as e:
            print(f"  Error: {e}")
        if x is not None:
            break

    # Fallback: CIFAR-10
    if x is None:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],
                                         [0.229,0.224,0.225])])
            ds = CIFAR10("./data", train=False, download=True, transform=ctf)
            for i in range(500):
                im, _ = ds[i]
                xc = im.unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(backbone(xc), dim=-1)
                    c, pr = p[0].max(0)
                if c.item() >= min_conf:
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"CIFAR-10 idx={i}"
                    break
        except Exception:
            pass

    # Fallback: synthetic
    if x is None:
        print("Using synthetic image fallback")
        m = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        s = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
        torch.manual_seed(42)
        raw = (torch.randn(1,3,224,224, device=device)*0.2+0.5).clamp(0,1)
        x = (raw - m) / s
        with torch.no_grad():
            p = F.softmax(backbone(x), dim=-1)
            c, pr = p[0].max(0)
            pc, cf = pr.item(), c.item()
        source = "synthetic"

    model = ClassLogitModel(backbone, target_class=pc).to(device).eval()
    baseline = torch.zeros_like(x)
    info = {"source": source, "target_class": pc, "confidence": cf,
            "model": "ResNet-50 (ImageNet pretrained)"}
    return model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §13  HEATMAP VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def visualize_attributions(x, methods, info, save_path="attribution_heatmaps.png",
                           delta_f=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(x.device)
    img = ((x*std+mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype("uint8")
    img_dark = (img.astype(float)*0.4).astype("uint8")

    cmap = LinearSegmentedColormap.from_list("heat", [
        (0,(0,0,0,0)), (0.3,(0.97,0.45,0.02,0.4)),
        (0.6,(0.97,0.71,0.22,0.7)), (1,(1,1,1,1))])
    colors = {"IG":"#6B7280","IDGI":"#8B5CF6","Guided IG":"#06B6D4",
              "μ-Optimized":"#F59E0B","Joint":"#EF4444"}

    n = len(methods)
    fig, axes = plt.subplots(2, n+1, figsize=(3.6*(n+1), 7.5), facecolor="#0D0D0D")

    axes[0,0].imshow(img); axes[0,0].set_title("Original", color="#E8E4DF",
        fontsize=10, fontfamily="monospace"); axes[0,0].axis("off")

    for i, m in enumerate(methods):
        sal = m.attributions[0].abs().sum(0).cpu().numpy()
        vmax = max(np.percentile(sal, 99), 1e-12)
        sal = (sal/vmax).clip(0,1)
        ax = axes[0, i+1]
        ax.imshow(img_dark); ax.imshow(sal, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
        c = colors.get(m.name, "#F7B538")
        ax.set_title(f"{m.name}\n𝒬={m.Q:.4f}  Var={m.Var_nu:.4f}",
                     color=c, fontsize=9, fontfamily="monospace", linespacing=1.4)
        ax.axis("off")

    # Bottom row: Q bar chart + signed heatmaps
    ax_bar = axes[1, 0]; ax_bar.set_facecolor("#0D0D0D")
    qs = [m.Q for m in methods]; names = [m.name for m in methods]
    cs = [colors.get(n, "#F7B538") for n in names]
    bars = ax_bar.barh(range(n), qs, color=cs, height=0.6)
    for bar, q in zip(bars, qs):
        ax_bar.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                    f"{q:.4f}", va="center", color="#E8E4DF", fontsize=8,
                    fontfamily="monospace")
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(names, fontsize=8, fontfamily="monospace", color="#E8E4DF")
    ax_bar.set_xlim(0, 1.15); ax_bar.invert_yaxis()
    ax_bar.set_title("𝒬 Score", color="#E8E4DF", fontsize=10, fontfamily="monospace")
    ax_bar.tick_params(colors="#888", labelsize=7)
    for sp in ax_bar.spines.values(): sp.set_color("#333")

    cmap_div = LinearSegmentedColormap.from_list("div", [
        (0,(0.15,0.35,0.85,0.9)), (0.5,(0,0,0,0)), (1,(0.95,0.2,0.1,0.9))])
    for i, m in enumerate(methods):
        sal = m.attributions[0].sum(0).cpu().numpy()
        vmax = max(np.percentile(np.abs(sal), 99), 1e-12)
        ax = axes[1, i+1]; ax.imshow(img_dark)
        ax.imshow((sal/vmax).clip(-1,1), cmap=cmap_div, vmin=-1, vmax=1, alpha=0.85)
        ax.set_title(f"Signed · {m.name}", color=colors.get(m.name,"#F7B538"),
                     fontsize=9, fontfamily="monospace"); ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="#0D0D0D",
                bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"✓ Heatmap → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# §14  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N=50, device=None, min_conf=0.70):
    if device is None:
        device = get_device()

    print("Loading ResNet-50 and image...")
    model, x, baseline, info = load_image_and_model(device, min_conf)

    f_x = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl

    print(f"\nModel : {info['model']}")
    print(f"Source: {info['source']}")
    print(f"Class : {info['target_class']} (conf={info['confidence']:.4f})")
    print(f"f(x) = {f_x:.4f},  f(bl) = {f_bl:.4f},  Δf = {delta_f:.4f}")
    print(f"N = {N}\n")
    print(f"{'Method':<16} {'Var_ν':>10} {'CV²':>8} {'𝒬':>8} {'Time':>8}")
    print("─" * 56)

    G = 16
    methods = [
        standard_ig(model, x, baseline, N),
        idgi(model, x, baseline, N),
        guided_ig(model, x, baseline, N),
        mu_optimized_ig(model, x, baseline, N, tau=0.005, n_iter=300),
        joint_ig(model, x, baseline, N, G=G, n_alternating=2,
                 tau=0.005, mu_iter=300, path_iter=G),
    ]

    for m in methods:
        print(f"{m.name:<16} {m.Var_nu:>10.6f} {m.CV2:>8.4f} "
              f"{m.Q:>8.4f} {m.elapsed_s:>7.1f}s")

    results = {
        "image_info": info,
        "model_info": {"f_x": f_x, "f_baseline": f_bl,
                       "delta_f": delta_f, "N": N, "device": str(device)},
        "methods": {m.name: m.to_dict() for m in methods},
    }
    return results, methods, model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §15  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified IG v3-batched (PyTorch)")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-conf", type=float, default=0.70)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-path", type=str, default="attribution_heatmaps.png")
    parser.add_argument("--viz-fidelity", action="store_true")
    parser.add_argument("--insdel", action="store_true")
    parser.add_argument("--insdel-steps", type=int, default=100)
    parser.add_argument("--viz-insdel", action="store_true")
    args = parser.parse_args()

    device = get_device(force=args.device)
    results, methods, model, x, baseline, info = run_experiment(
        N=args.steps, device=device, min_conf=args.min_conf)

    if args.insdel or args.viz_insdel:
        run_insertion_deletion(model, x, baseline, methods,
                               n_steps=args.insdel_steps)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults → {args.json}")

    if args.viz:
        visualize_attributions(x, methods, info, save_path=args.viz_path,
                               delta_f=results["model_info"]["delta_f"])

    if args.viz_fidelity:
        fpath = args.viz_path.replace(".png", "_fidelity.png")
        visualize_step_fidelity(methods, save_path=fpath)

    if args.viz_insdel:
        ipath = args.viz_path.replace(".png", "_insdel.png")
        visualize_insertion_deletion(methods, save_path=ipath)