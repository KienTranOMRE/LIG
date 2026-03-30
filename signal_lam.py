"""
signal_harvesting.py — Signal-Harvesting Action for Integrated Gradients
=========================================================================

Implements the unified variational objective from the paper (Eq. 20):

    min_{γ,μ}  Var_ν(φ)  −  λ Σ_k μ_k |d_k|  +  (τ/2) ‖μ‖²₂

where:
    Term 1: Var_ν(φ)           — linearisation distortion (Fermat/Snell)
    Term 2: −λ Σ_k μ_k |d_k|  — signal harvested (transition concentration)
    Term 3: (τ/2) ‖μ‖²₂       — L2 admissibility (prevents spike degeneracy)

Special cases (Table 1 in paper):
    λ=0, τ→∞    : Standard IG      (uniform μ, straight line)
    λ>0, τ→0    : IDGI             (μ_k ∝ |Δf_k|, straight line)
    λ>0, uniform : Guided IG        (heuristic path, uniform μ)
    λ=0, τ>0    : μ-Optimised      (min Var_ν only)
    λ>0, τ>0    : Joint*           (full signal-harvesting solution)

Drop-in replacements for optimize_mu, mu_optimized_ig, joint_ig in
unified_ig.py.

Usage:
    from signal_harvesting import (
        optimize_mu_signal_harvesting,
        mu_star_closed_form,
        mu_optimized_ig,
        joint_star_ig,
        compute_signal_harvesting_objective,
    )
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import shared infrastructure from the existing codebase
from utilss import (
    AttributionResult, StepInfo,
    compute_Var_nu, compute_CV2, compute_Q, compute_all_metrics,
)

# Import path/gradient utilities from existing unified_ig
from lam import (
    _forward_scalar, _forward_batch, _forward_and_gradient,
    _forward_and_gradient_batch, _gradient, _gradient_batch,
    _dot, _rescale, _build_steps, _straight_line_pass,
    _build_spatial_groups, _build_path_2d, _eval_path_batched,
    _group_cache,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  SIGNAL-HARVESTING OBJECTIVE  (Eq. 20)
# ═════════════════════════════════════════════════════════════════════════════

def compute_signal_harvesting_objective(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mu: torch.Tensor,
    lam: float = 1.0,
    tau: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Evaluate the full signal-harvesting objective (Eq. 20):

        L(γ,μ) = Var_ν(φ)  −  λ Σ_k μ_k |d_k|  +  (τ/2) ‖μ‖²₂

    Args:
        d:       (N,) tensor of d_k = ∇f(γ_k) · Δγ_k
        delta_f: (N,) tensor of Δf_k = f(γ_{k+1}) − f(γ_k)
        mu:      (N,) probability measure over steps
        lam:     signal-harvesting strength λ
        tau:     L2 admissibility multiplier τ

    Returns:
        (total_objective, var_nu_term, signal_term, l2_term)
    """
    # Term 1: Var_ν(φ)
    var_nu = compute_Var_nu(d, delta_f, mu)

    # Term 2: −λ Σ_k μ_k |d_k|
    signal = float((mu * d.abs()).sum())

    # Term 3: (τ/2) ‖μ‖²₂
    l2 = float((mu ** 2).sum())

    total = var_nu - lam * signal + (tau / 2.0) * l2

    return total, var_nu, signal, l2


# ═════════════════════════════════════════════════════════════════════════════
# §2  CLOSED-FORM μ* (KKT stationary point, Eq. 15/21)
# ═════════════════════════════════════════════════════════════════════════════

def mu_star_closed_form(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mode: str = "d",
) -> torch.Tensor:
    """
    Closed-form KKT stationary measure (Eq. 14-15):

        μ*_k ∝ |d_k|  ≈  |Δf_k|

    This is the exact stationary point of the signal-harvesting action
    in the limit τ → 0⁺, λ > 0, at Var_ν = 0.

    Args:
        d:       (N,) tensor of d_k
        delta_f: (N,) tensor of Δf_k
        mode:    "d" uses |d_k| (exact KKT), "df" uses |Δf_k| (IDGI)

    Returns:
        (N,) normalised probability measure
    """
    if mode == "d":
        weights = d.abs()
    elif mode == "df":
        weights = delta_f.abs()
    else:
        raise ValueError(f"mode must be 'd' or 'df', got '{mode}'")

    w_sum = weights.sum()
    if w_sum < 1e-12:
        return torch.full_like(weights, 1.0 / len(weights))
    return weights / w_sum


# ═════════════════════════════════════════════════════════════════════════════
# §3  μ-OPTIMISATION WITH SIGNAL HARVESTING  (Phase 1, Eq. 24)
#
#     Objective:  min  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ‖μ‖²₂
#
#     Key differences from original optimize_mu:
#       1. L2 penalty replaces entropy  (linear stationary condition → IDGI)
#       2. Signal-harvesting term −λΣμ_k|d_k| added
#       3. Softmax parameterisation on simplex (μ ≥ 0, Σμ = 1)
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu_signal_harvesting(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    lam: float = 1.0,
    tau: float = 0.01,
    n_iter: int = 300,
    lr: float = 0.05,
) -> torch.Tensor:
    """
    Find μ minimising the signal-harvesting objective (Eq. 24):

        min_{μ∈P_N}  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ‖μ‖²₂

    Uses Adam on softmax logits for unconstrained optimisation on the simplex.

    The L2 penalty (τ/2)‖μ‖² replaces the entropy penalty τΣμ_k log μ_k
    from the original code, because:
      - L2 yields a LINEAR stationary condition: μ*_k ∝ |d_k|/τ
      - Entropy yields an EXPONENTIAL condition: μ*_k ∝ exp(|d_k|/τ)
      - Only the linear form recovers IDGI exactly (Proposition 8)

    Limiting behaviour (Eq. 25):
        λ → 0 :  μ* → arg min Var_ν(φ)          (original LAM)
        λ → ∞ :  μ* → |d_k| / Σ_j |d_j|         (IDGI)

    Cost: O(N) arithmetic per iteration, zero additional model evaluations.
    """
    device = d.device
    N = d.shape[0]

    # ── Constants w.r.t. μ — hoist out of optimisation loop ──
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d)).detach()
    df2 = (delta_f ** 2).detach()
    abs_d = d.abs().detach()          # |d_k| for signal-harvesting term

    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mu = torch.softmax(logits, dim=0)

        # ── Term 1: Var_ν(φ) ──
        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        w = nu / nu_sum

        mean_phi = (w * phi).sum()
        var_phi = (w * (phi - mean_phi) ** 2).sum()

        # ── Term 2: −λ Σ_k μ_k |d_k| ──
        signal_term = (mu * abs_d).sum()

        # ── Term 3: (τ/2) ‖μ‖²₂ ──
        l2_term = (mu ** 2).sum()

        # ── Full objective (Eq. 20) ──
        loss = var_phi - lam * signal_term + (tau / 2.0) * l2_term

        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §4  μ-OPTIMISED IG WITH SIGNAL HARVESTING
#
#     Straight line + optimal μ from Eq. 24.
#     This is the most practical contribution: high Q and good faithfulness
#     at ZERO additional model evaluations beyond standard IG.
# ═════════════════════════════════════════════════════════════════════════════

def _pack_result(name, attr, d_list, df_list, f_vals, gnorms, mu, N,
                 t0, Q_history=None) -> AttributionResult:
    """Build AttributionResult with all metrics.  Matches unified_ig._pack_result."""
    device = attr.device
    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    var_nu, cv2, Q = compute_all_metrics(d_arr, df_arr, mu)
    steps = _build_steps(d_list, df_list, f_vals, gnorms, mu, N)
    return AttributionResult(
        name=name, attributions=attr, Q=Q, CV2=cv2, Var_nu=var_nu,
        steps=steps, Q_history=Q_history or [], elapsed_s=time.time() - t0,
    )


def mu_optimized_ig(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    lam: float = 1.0,
    tau: float = 0.01,
    n_iter: int = 300,
) -> AttributionResult:
    """
    Straight line + optimal μ under the signal-harvesting objective.

    This is the "μ-Optimised" row in Table 1 with λ > 0: it combines the
    conservation-law objective (Var_ν) with signal harvesting (−λΣμ|d|)
    and L2 admissibility.

    Cost: standard IG + O(N) arithmetic.  Zero extra model evaluations.

    Special cases:
        lam=0 : pure conservation (original LAM μ-optimisation)
        lam→∞ : recovers IDGI (μ_k ∝ |d_k|)
    """
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)

    mu = optimize_mu_signal_harvesting(
        d_arr, df_arr, lam=lam, tau=tau, n_iter=n_iter)

    # Weighted gradient sum
    grad_stack = torch.cat(grads, dim=0)                 # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)   # (1, C, H, W)
    attr = _rescale(delta_x * wg, target)

    return _pack_result("μ-Optimized*", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §5  SIGNAL-HARVESTING PATH OBJECTIVE
#
#     For path optimisation, we use the full objective (Eq. 20) evaluated
#     on the candidate path, not just CV²(φ).
#
#     The path changes both d_k and Δf_k, so we use
#       MSE_ν(φ,1) − λ Σ μ_k |d_k|
#     as the path sub-objective.  MSE_ν = Var_ν + (φ̄−1)² prevents the
#     degenerate basin where φ_k = c ≠ 1.
# ═════════════════════════════════════════════════════════════════════════════

def _signal_harvesting_path_obj(
    d_v: torch.Tensor,
    df_v: torch.Tensor,
    mu: torch.Tensor,
    lam: float = 1.0,
) -> float:
    """
    Path sub-objective: MSE_ν(φ,1) − λ Σ_k μ_k |d_k|

    MSE_ν(φ,1) = Σ ν_k (φ_k − 1)² = Var_ν(φ) + (φ̄_ν − 1)²

    The signal-harvesting term forces the path toward regions where
    |∇f · Δγ| is large — the output-transition region (Eq. 16).
    """
    valid = df_v.abs() > 1e-12
    safe_df = torch.where(valid, df_v, torch.ones_like(df_v))
    phi = torch.where(valid, d_v / safe_df, torch.ones_like(d_v))

    nu = mu * df_v ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return 0.0
    nu = nu / nu_sum

    # MSE_ν(φ, 1) = Σ ν_k (φ_k − 1)²
    mse = float((nu * (phi - 1.0) ** 2).sum())

    # Signal harvested: Σ μ_k |d_k|
    signal = float((mu * d_v.abs()).sum())

    return mse - lam * signal


# ═════════════════════════════════════════════════════════════════════════════
# §6  PATH OPTIMISATION WITH SIGNAL HARVESTING  (Phase 2, Eq. 16)
#
#     The Euler-Lagrange equation (Eq. 16) acquires a forcing term
#     from the signal-harvesting action that pushes γ toward regions
#     where H_f γ' is large in the direction of ∇f — the transition region.
#
#     We approximate this variationally via grouped velocity scheduling
#     with the signal-harvesting path objective.
# ═════════════════════════════════════════════════════════════════════════════

# def optimize_path_signal_harvesting(
#     model: nn.Module,
#     x: torch.Tensor,
#     baseline: torch.Tensor,
#     mu: torch.Tensor,
#     N: int = 50,
#     G: int = 16,
#     patch_size: int = 14,
#     n_iter: int = 15,
#     lr: float = 0.08,
#     lam: float = 1.0,
# ):
#     """
#     Optimise path via grouped spatial velocity scheduling under the
#     signal-harvesting objective.

#     Objective per probe:
#         MSE_ν(φ,1) − λ Σ_k μ_k |d_k|

#     The −λΣμ|d| term creates a forcing that biases the path toward
#     concentrating output change into steps where μ is large — matching
#     the Euler-Lagrange forcing term in Eq. 16.

#     Stochastic FD: one random time step per group per iteration.
#     Cost: O(G) batched model evaluations per iteration.
#     """
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     # Initialise: uniform velocity = straight line
#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     for it in range(n_iter):
#         obj = _obj_of(V)
#         if obj < best_obj:
#             best_obj = obj
#             best_V = V.clone()

#         # Stochastic FD: perturb one random time step per group
#         grad_V = torch.zeros_like(V)
#         for g in range(G):
#             k = torch.randint(0, N, (1,)).item()
#             V[g, k] += eps
#             obj_plus = _obj_of(V)
#             grad_V[g, k] = (obj_plus - obj) / eps
#             V[g, k] -= eps

#         V = V - lr * grad_V
#         V = torch.clamp(V, min=0.01)

#     return _build_path_2d(baseline, delta_x, best_V, gmap, N)
# def optimize_path_signal_harvesting(
#     model: nn.Module,
#     x: torch.Tensor,
#     baseline: torch.Tensor,
#     mu: torch.Tensor,
#     N: int = 50,
#     G: int = 16,
#     patch_size: int = 14,
#     n_iter: int = 25,
#     lr: float = 0.08,
#     lam: float = 1.0,
#     early_stop_patience: int = 10,
#     early_stop_rtol: float = 0.01,
#     verbose: bool = True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []
#     block_size = 5
#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(V)
#         improved = False

#         if obj < best_obj:
#             best_obj = obj
#             best_V = V.clone()
#             improved = True

#         # Stochastic FD: perturb one random time step per group
#         grad_V = torch.zeros_like(V)
#         grad_norms_per_group = []
#         # for g in range(G):
#         #     k = torch.randint(0, N, (1,)).item()
#         #     V[g, k] += eps
#         #     obj_plus = _obj_of(V)
#         #     grad_V[g, k] = (obj_plus - obj) / eps
#         #     V[g, k] -= eps
#         #     grad_norms_per_group.append(abs(grad_V[g, k].item()))
#         for g in range(G):
#                 # Random block start
#                 k0 = torch.randint(0, N - block_size + 1, (1,)).item()
#                 k1 = k0 + block_size
                
#                 # Perturb the block with a structured pattern
#                 z = torch.randn(block_size, device=device)
#                 z = z / z.norm() * block_size**0.5  # scale so per-element magnitude ~ eps
                
#                 V[g, k0:k1] += eps * z
#                 obj_plus = _obj_of(V)
#                 V[g, k0:k1] -= eps * z
#                 grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
#                 grad_norms_per_group.append(float(grad_V[g].norm()))


#         V = V - lr * grad_V
#         V = torch.clamp(V, min=0.01)

#         dt = time.time() - t_it
#         grad_norm = float(grad_V.norm())
#         obj_history.append(best_obj)

#         if verbose:
#             mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
#             max_g = max(grad_norms_per_group)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇V|={grad_norm:.4f}  "
#                   f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         # Early stopping: check if best_obj has stagnated
#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")
#   return _build_path_2d(baseline, delta_x, best_V, gmap, N)
def optimize_path_signal_harvesting(
    model, x, baseline, mu, N=50, G=16, patch_size=14,
    n_iter=15, lr=0.08, lam=1.0,
    early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
):
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    V = torch.ones(G, N, device=device, requires_grad=False)
    best_obj = float("inf")
    best_V = V.clone()

    # Adam state per V[g, k]
    m_V = torch.zeros_like(V)  # first moment
    v_V = torch.zeros_like(V)  # second moment
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    def _obj_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

    eps = 0.05
    block_size = max(N // 10, 3)
    stale_count = 0
    prev_best = float("inf")
    obj_history = []

    for it in range(n_iter):
        t_it = time.time()
        obj = _obj_of(V)
        improved = obj < best_obj
        if improved:
            best_obj = obj
            best_V = V.clone()

        # Block FD gradient estimation
        grad_V = torch.zeros_like(V)
        grad_norms_per_group = []
        for g in range(G):
            k0 = torch.randint(0, N - block_size + 1, (1,)).item()
            k1 = k0 + block_size
            z = torch.randn(block_size, device=device)
            z = z / z.norm() * block_size**0.5

            V[g, k0:k1] += eps * z
            obj_plus = _obj_of(V)
            V[g, k0:k1] -= eps * z

            grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
            grad_norms_per_group.append(float(grad_V[g].norm()))

        # Adam update
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        m_hat = m_V / (1 - beta1 ** (it + 1))
        v_hat = v_V / (1 - beta2 ** (it + 1))

        V = V - lr * m_hat / (v_hat.sqrt() + adam_eps)
        V = torch.clamp(V, min=0.01)

        dt = time.time() - t_it
        obj_history.append(best_obj)

        if verbose:
            mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
            max_g = max(grad_norms_per_group)
            print(f"  path_opt iter {it:2d}/{n_iter}  "
                  f"obj={obj:+.6f}  best={best_obj:+.6f}  "
                  f"|∇V|={float(grad_V.norm()):.4f}  "
                  f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
                  f"{'✓' if improved else ' '}  {dt:.2f}s")

        # Early stopping
        if abs(prev_best) > 1e-12:
            rel_change = abs(prev_best - best_obj) / abs(prev_best)
        else:
            rel_change = abs(prev_best - best_obj)

        if rel_change < early_stop_rtol:
            stale_count += 1
        else:
            stale_count = 0
        prev_best = best_obj

        if stale_count >= early_stop_patience:
            if verbose:
                print(f"  ⚡ Early stop at iter {it}: "
                      f"no improvement > {early_stop_rtol:.1%} "
                      f"for {early_stop_patience} iters")
            break

    if verbose:
        print(f"  path_opt done: {len(obj_history)} iters, "
              f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
              f"(Δ={obj_history[0] - best_obj:+.4f})")

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)

# def optimize_path_signal_harvesting_no(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=15, lr=0.08, lam=1.0,
#     momentum=0.7, block_size=None,
#     early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()
#     vel_V = torch.zeros_like(V)

#     if block_size is None:
#         block_size = max(N // 10, 3)

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []

#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(V)
#         improved = obj < best_obj
#         if improved:
#             best_obj = obj
#             best_V = V.clone()

#         grad_V = torch.zeros_like(V)
#         grad_norms_per_group = []
#         for g in range(G):
#             k0 = torch.randint(0, N - block_size + 1, (1,)).item()
#             k1 = k0 + block_size
#             z = torch.randn(block_size, device=device)
#             z = z / z.norm() * block_size**0.5

#             V[g, k0:k1] += eps * z
#             obj_plus = _obj_of(V)
#             V[g, k0:k1] -= eps * z

#             grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
#             grad_norms_per_group.append(float(grad_V[g].norm()))

#         # SGD with momentum
#         vel_V = momentum * vel_V + grad_V
#         V = V - lr * vel_V
#         V = torch.clamp(V, min=0.01)

#         dt = time.time() - t_it
#         obj_history.append(best_obj)

#         if verbose:
#             mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
#             max_g = max(grad_norms_per_group)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇V|={float(grad_V.norm()):.4f}  "
#                   f"|vel|={float(vel_V.norm()):.4f}  "
#                   f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         # Early stopping
#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")

#     return _build_path_2d(baseline, delta_x, best_V, gmap, N)

# def optimize_path_signal_harvesting_no(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=15, lr=0.08, lam=1.0,
#     momentum=0.7, block_size=None,
#     early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()
#     vel_V = torch.zeros_like(V)

#     if block_size is None:
#         block_size = max(N // 10, 3)

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []
#     restarted = False

#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(V)
#         improved = obj < best_obj
#         if improved:
#             best_obj = obj
#             best_V = V.clone()

#         grad_V = torch.zeros_like(V)
#         grad_norms_per_group = []
#         for g in range(G):
#             k0 = torch.randint(0, N - block_size + 1, (1,)).item()
#             k1 = k0 + block_size
#             z = torch.randn(block_size, device=device)
#             z = z / z.norm() * block_size**0.5

#             V[g, k0:k1] += eps * z
#             obj_plus = _obj_of(V)
#             V[g, k0:k1] -= eps * z

#             grad_V[g, k0:k1] = ((obj_plus - obj) / eps) * z
#             grad_norms_per_group.append(float(grad_V[g].norm()))

#         # SGD with momentum
#         vel_V = momentum * vel_V + grad_V
#         V = V - lr * vel_V
#         V = torch.clamp(V, min=0.01)

#         dt = time.time() - t_it
#         obj_history.append(best_obj)

#         if verbose:
#             mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
#             max_g = max(grad_norms_per_group)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇V|={float(grad_V.norm()):.4f}  "
#                   f"|vel|={float(vel_V.norm()):.4f}  "
#                   f"mean/max_g={mean_g:.4f}/{max_g:.4f}  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         # Early stopping
#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         # Restart from best_V halfway through patience
#         if stale_count == early_stop_patience // 2 and not restarted:
#             if verbose:
#                 print(f"  🔄 Restart from best_V at iter {it}")
#             V = best_V.clone()
#             vel_V = torch.zeros_like(V)
#             stale_count = 0
#             restarted = True
#             continue

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")
#
#    return _build_path_2d(baseline, delta_x, best_V, gmap, N)

# def optimize_path_signal_harvesting(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=15, lr=0.002, lam=1.0,
#     momentum=0.5, n_basis=15,
#     early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     basis = torch.stack([
#         torch.cos(torch.arange(N, device=device, dtype=torch.float32) * j * 3.14159 / N)
#         for j in range(n_basis)
#     ])
#     basis = basis / basis.norm(dim=1, keepdim=True)

#     A = torch.zeros(G, n_basis, device=device)
#     A[:, 0] = basis[0].sum()

#     best_obj = float("inf")
#     best_A = A.clone()
#     vel_A = torch.zeros_like(A)

#     def _V_from_A(Am):
#         return torch.clamp(Am @ basis, min=0.01)

#     def _obj_of(Am):
#         V = _V_from_A(Am)
#         gp = _build_path_2d(baseline, delta_x, V, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.01
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []
#     restarted = False

#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(A)
#         improved = obj < best_obj
#         if improved:
#             best_obj = obj
#             best_A = A.clone()

#         grad_A = torch.zeros_like(A)
#         grad_norms_per_group = []
#         for g in range(G):
#             j = torch.randint(0, n_basis, (1,)).item()
#             A[g, j] += eps
#             obj_plus = _obj_of(A)
#             grad_A[g, j] = (obj_plus - obj) / eps
#             A[g, j] -= eps
#             grad_norms_per_group.append(float(grad_A[g].norm()))

#         # Normalize gradient to have unit norm, then scale by lr
#         grad_norm = grad_A.norm()
#         if grad_norm > 1e-8:
#             grad_A = grad_A / grad_norm

#         vel_A = momentum * vel_A + grad_A
#         A = A - lr * vel_A

#         dt = time.time() - t_it
#         obj_history.append(best_obj)

#         if verbose:
#             V_now = _V_from_A(A)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇A|={float(grad_norm):.4f}  "
#                   f"|vel|={float(vel_A.norm()):.4f}  "
#                   f"V=[{float(V_now.min()):.2f},{float(V_now.max()):.2f}]  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         if stale_count == early_stop_patience // 2 and not restarted:
#             if verbose:
#                 print(f"  🔄 Restart from best_A at iter {it}")
#             A = best_A.clone()
#             vel_A = torch.zeros_like(A)
#             stale_count = 0
#             restarted = True
#             continue

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         n_params = G * n_basis
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"{n_params} params (G={G} × basis={n_basis}), "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")

#     V = _V_from_A(best_A)
#     return _build_path_2d(baseline, delta_x, V, gmap, N)
# def optimize_path_signal_harvesting(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=15, lr=0.08, lam=1.0,
#     momentum=0.7, bump_width=None,
#     early_stop_patience=10, early_stop_rtol=0.01, verbose=True,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

#     V = torch.ones(G, N, device=device)
#     best_obj = float("inf")
#     best_V = V.clone()
#     vel_V = torch.zeros_like(V)

#     if bump_width is None:
#         bump_width = max(N // 10, 3)

#     # Precompute timestep indices
#     t_idx = torch.arange(N, device=device, dtype=torch.float32)

#     def _obj_of(Vm):
#         gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
#         d_v, df_v = _eval_path_batched(model, gp, N, device)
#         return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

#     eps = 0.05
#     stale_count = 0
#     prev_best = float("inf")
#     obj_history = []
#     restarted = False

#     for it in range(n_iter):
#         t_it = time.time()
#         obj = _obj_of(V)
#         improved = obj < best_obj
#         if improved:
#             best_obj = obj
#             best_V = V.clone()

#         grad_V = torch.zeros_like(V)
#         grad_norms_per_group = []
#         for g in range(G):
#             # Random Gaussian bump
#             center = torch.randint(0, N, (1,)).item()
#             bump = torch.exp(-0.5 * ((t_idx - center) / bump_width) ** 2)
#             bump = bump / bump.norm()

#             V[g] += eps * bump
#             obj_plus = _obj_of(V)
#             V[g] -= eps * bump

#             grad_V[g] = ((obj_plus - obj) / eps) * bump
#             grad_norms_per_group.append(float(grad_V[g].norm()))

#         # SGD with momentum
#         vel_V = momentum * vel_V + grad_V
#         V = V - lr * vel_V
#         V = torch.clamp(V, min=0.01)

#         dt = time.time() - t_it
#         obj_history.append(best_obj)

#         if verbose:
#             mean_g = sum(grad_norms_per_group) / len(grad_norms_per_group)
#             max_g = max(grad_norms_per_group)
#             print(f"  path_opt iter {it:2d}/{n_iter}  "
#                   f"obj={obj:+.6f}  best={best_obj:+.6f}  "
#                   f"|∇V|={float(grad_V.norm()):.4f}  "
#                   f"|vel|={float(vel_V.norm()):.4f}  "
#                   f"V=[{float(V.min()):.2f},{float(V.max()):.2f}]  "
#                   f"{'✓' if improved else ' '}  {dt:.2f}s")

#         # Early stopping
#         if abs(prev_best) > 1e-12:
#             rel_change = abs(prev_best - best_obj) / abs(prev_best)
#         else:
#             rel_change = abs(prev_best - best_obj)

#         if rel_change < early_stop_rtol:
#             stale_count += 1
#         else:
#             stale_count = 0
#         prev_best = best_obj

#         # Restart from best halfway through patience
#         if stale_count == early_stop_patience // 2 and not restarted:
#             if verbose:
#                 print(f"  🔄 Restart from best_V at iter {it}")
#             V = best_V.clone()
#             vel_V = torch.zeros_like(V)
#             stale_count = 0
#             restarted = True
#             continue

#         if stale_count >= early_stop_patience:
#             if verbose:
#                 print(f"  ⚡ Early stop at iter {it}: "
#                       f"no improvement > {early_stop_rtol:.1%} "
#                       f"for {early_stop_patience} iters")
#             break

#     if verbose:
#         print(f"  path_opt done: {len(obj_history)} iters, "
#               f"bump_width={bump_width}, "
#               f"obj {obj_history[0]:+.4f} → {best_obj:+.4f}  "
#               f"(Δ={obj_history[0] - best_obj:+.4f})")

#     return _build_path_2d(baseline, delta_x, best_V, gmap, N)

# def optimize_path_signal_harvesting(
#     model, x, baseline, mu, N=50, G=16, patch_size=14,
#     n_iter=30, lr=0.02, lam=1.0,
# ):
#     device = x.device
#     delta_x = x - baseline
#     gmap = _build_spatial_groups(model, x, baseline, G, patch_size)
#     gmap_flat = gmap.flatten()
#     _, C, H, W = baseline.shape

#     # Learnable parameter
#     V_logits = torch.zeros(G, N, device=device, requires_grad=True)
#     optimizer = torch.optim.Adam([V_logits], lr=lr)

#     best_obj = float("inf")
#     best_V = None

#     for it in range(n_iter):
#         optimizer.zero_grad()

#         # Softplus ensures V > 0 without hard clamping (differentiable)
#         V = F.softplus(V_logits)
#         v_sums = V.sum(dim=1, keepdim=True).clamp(min=1e-12)
#         W_norm = V / v_sums                          # (G, N)

#         pixel_weights = W_norm.T[:, gmap_flat]        # (N, H*W)
#         pixel_weights = pixel_weights.view(N, 1, H, W)

#         steps = delta_x * pixel_weights               # (N, C, H, W)
#         cum = torch.cumsum(steps, dim=0)
#         gamma_stack = torch.cat([baseline, baseline + cum], dim=0)  # (N+1, C, H, W)

#         # Forward through model — IN the graph
#         with torch.enable_grad():
#             f_all = model(gamma_stack)                 # (N+1,)

#         # Gradients at first N points
#         pts_N = gamma_stack[:N]
#         grads_N = torch.autograd.grad(
#             f_all[:N].sum(), gamma_stack,
#             create_graph=True
#         )[0][:N]                                       # (N, C, H, W)

#         step_vecs = gamma_stack[1:] - gamma_stack[:N]
#         d_v = (grads_N * step_vecs).view(N, -1).sum(dim=1)

#         f_ext = torch.cat([f_all[0:1], f_all])
#         df_v = f_ext[1:N+1] - f_ext[:N]

#         # Objective: MSE_ν(φ,1) - λ Σ μ_k |d_k|
#         valid = df_v.abs() > 1e-12
#         safe_df = torch.where(valid, df_v, torch.ones_like(df_v))
#         phi = torch.where(valid, d_v / safe_df, torch.ones_like(d_v))

#         nu = mu * df_v ** 2
#         nu = nu / nu.sum().clamp(min=1e-15)
#         mse = (nu * (phi - 1.0) ** 2).sum()
#         signal = (mu * d_v.abs()).sum()
#         loss = mse - lam * signal

#         loss.backward()
#         optimizer.step()

#         with torch.no_grad():
#             if loss.item() < best_obj:
#                 best_obj = loss.item()
#                 best_V = F.softplus(V_logits).detach().clone()

#     return _build_path_2d(baseline, delta_x, best_V, gmap, N)
# ═════════════════════════════════════════════════════════════════════════════
# §7  JOINT* OPTIMISATION  (Algorithm 1 — Full Signal-Harvesting Solution)
#
#     Alternating minimisation of (Eq. 20):
#       Phase 1 (measure): fix γ, optimise μ via Eq. 24
#       Phase 2 (path):    fix μ, optimise γ via velocity scheduling
#       Each phase monotonically decreases the objective.
# ═════════════════════════════════════════════════════════════════════════════

def joint_star_ig(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    G: int = 16,
    patch_size: int = 14,
    n_alternating: int = 2,
    lam: float = 1.0,
    tau: float = 0.01,
    mu_iter: int = 300,
    path_iter: int = 10,
    init_path: Optional[list[torch.Tensor]] = None,
) -> AttributionResult:
    """
    Joint* — the complete signal-harvesting solution (Table 1, last row).

    Alternating minimisation of the full objective (Eq. 20):
        min_{γ,μ}  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ‖μ‖²₂

    Phase 1 (measure): fix γ, optimise μ via projected gradient descent
        on the signal-harvesting objective (Eq. 24).
        Limiting: λ→0 gives original LAM, λ→∞ gives IDGI.

    Phase 2 (path): fix μ*, optimise γ via grouped velocity scheduling
        with objective MSE_ν(φ,1) − λ Σ_k μ_k |d_k| (Eq. 16 approximation).
        The signal-harvesting term forces γ toward the transition region.

    Parameters
    ----------
    lam : float
        Signal-harvesting strength λ (Eq. 20). Controls interpolation:
            λ = 0   : pure conservation (original LAM)
            λ → ∞   : pure signal harvesting (IDGI limit)
            λ ∈ [0.5, 2.0] : recommended range (Section 7.3)
    tau : float
        L2 admissibility multiplier (Eq. 19). Prevents μ from collapsing
        to a Dirac spike. Recommended: τ ∈ [0.005, 0.01].
    init_path : optional
        List of N+1 tensors for initial path. Pass gamma_pts from Guided IG
        to warm-start from a better baseline.

    Regression guard: path optimisation only accepted if it improves the
    signal-harvesting objective. Guarantees Joint* ≥ init.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    # ── Initialise path ──
    if init_path is not None:
        assert len(init_path) == N + 1, \
            f"init_path must have N+1={N+1} points, got {len(init_path)}"
        gamma_pts = [p.clone() for p in init_path]
    else:
        gamma_pts = [baseline + (k / N) * delta_x for k in range(N + 1)]

    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history = []

    # ── Track best state (by signal-harvesting objective) ──
    best_obj = float("inf")
    best_Q = -1.0
    best_gamma_pts = gamma_pts
    best_mu = mu
    best_d_list: list[float] = []
    best_df_list: list[float] = []
    best_f_vals: list[float] = []
    best_gnorms: list[float] = []
    best_grads: list[torch.Tensor] = []

    def _evaluate_path(gp, mu_vec):
        """Batched evaluation of a path.  Returns diagnostics."""
        ap = torch.cat(gp, dim=0)                          # (N+1, C, H, W)
        with torch.no_grad():
            fa = model(ap)                                  # (N+1,)
        pn = ap[:N]
        gb = _gradient_batch(model, pn)                     # (N, C, H, W)
        sb = ap[1:] - ap[:N]                                # steps
        dt = (gb * sb).view(N, -1).sum(dim=1)               # d_k
        dl = dt.tolist()

        f0 = fa[0].item()
        fv = [f0] + fa.tolist()                             # N+2 entries
        dfl = [fv[k + 1] - fv[k] for k in range(N)]

        gn = gb.view(N, -1).norm(dim=1).tolist()
        gr = [gb[k:k+1].clone() for k in range(N)]
        da = torch.tensor(dl, device=device)
        dfa = torch.tensor(dfl, device=device)
        return dl, dfl, fv, gn, gr, da, dfa

    for s in range(n_alternating):
        # ── Evaluate current path ──
        d_list, df_list, f_vals, gnorms, grads, d_arr, df_arr = \
            _evaluate_path(gamma_pts, mu)

        # ── Phase 1: optimise μ (Eq. 24) ──
        mu = optimize_mu_signal_harvesting(
            d_arr, df_arr, lam=lam, tau=tau, n_iter=mu_iter)

        # Evaluate quality and objective
        var_mu, cv2_mu, Q_mu = compute_all_metrics(d_arr, df_arr, mu)
        obj_mu, _, _, _ = compute_signal_harvesting_objective(
            d_arr, df_arr, mu, lam=lam, tau=tau)

        # Update best if improved (by objective, with Q as tiebreaker)
        if obj_mu < best_obj or (abs(obj_mu - best_obj) < 1e-8 and Q_mu > best_Q):
            best_obj = obj_mu
            best_Q = Q_mu
            best_gamma_pts = gamma_pts
            best_mu = mu.clone()
            best_d_list, best_df_list = d_list, df_list
            best_f_vals, best_gnorms, best_grads = f_vals, gnorms, grads

        # ── Phase 2: optimise path (Eq. 16 approximation) ──
        Q_path = Q_mu
        obj_path = obj_mu
        if s < n_alternating - 1:
            new_gamma_pts = optimize_path_signal_harvesting(
                model, x, baseline, mu, N=N, G=G,
                patch_size=patch_size, n_iter=path_iter, lr=0.08,
                lam=lam)

            # Evaluate the new path
            new_d_list, new_df_list, new_f_vals, new_gnorms, new_grads, \
                new_d_arr, new_df_arr = _evaluate_path(new_gamma_pts, mu)
            _, _, Q_new = compute_all_metrics(new_d_arr, new_df_arr, mu)
            obj_new, _, _, _ = compute_signal_harvesting_objective(
                new_d_arr, new_df_arr, mu, lam=lam, tau=tau)

            Q_path = Q_new
            obj_path = obj_new

            # Regression guard: only accept if objective improved
            if obj_new < best_obj:
                gamma_pts = new_gamma_pts
                d_list, df_list = new_d_list, new_df_list
                f_vals, gnorms, grads = new_f_vals, new_gnorms, new_grads
                d_arr, df_arr = new_d_arr, new_df_arr

                best_obj = obj_new
                best_Q = Q_new
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
            "obj_after_mu": float(obj_mu),
            "obj_after_path": float(obj_path),
            "best_Q": float(best_Q),
            "best_obj": float(best_obj),
        })

    # ── Use best state for final attributions ──
    gamma_pts = best_gamma_pts
    mu = best_mu
    grads = best_grads

    attr = torch.zeros_like(x)
    for k in range(N):
        attr += mu[k] * grads[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    return _pack_result("Joint*", attr, best_d_list, best_df_list,
                        best_f_vals, best_gnorms, mu, N, t0, Q_history)


# ═════════════════════════════════════════════════════════════════════════════
# §8  CONVENIENCE: run_all_methods
#
#     Runs all six methods from Table 1 and returns them in order.
# ═════════════════════════════════════════════════════════════════════════════

def run_all_methods(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    lam: float = 1.0,
    tau: float = 0.01,
    G: int = 16,
    patch_size: int = 14,
    n_alternating: int = 2,
    mu_iter: int = 300,
    path_iter: int = 15,
    guided_init: bool = False,
) -> list[AttributionResult]:
    """
    Run all six IG variants from Table 1 of the paper.

    Returns list: [IG, IDGI, Guided IG, μ-Optimized*, Joint(λ=0), Joint*]
    """
    from lam import standard_ig, idgi, guided_ig, joint_ig

    results = []

    # 1. Standard IG  (λ=0, τ→∞, uniform μ)
    results.append(standard_ig(model, x, baseline, N))

    # 2. IDGI  (λ>0, τ→0, μ_k ∝ |Δf_k|)
    results.append(idgi(model, x, baseline, N))

    # 3. Guided IG  (heuristic path, uniform μ)
    gig = guided_ig(model, x, baseline, N)
    results.append(gig)

    # 4. μ-Optimized*  (straight line, signal-harvesting μ)
    results.append(mu_optimized_ig(
        model, x, baseline, N, lam=lam, tau=tau, n_iter=mu_iter))

    # # 5. Joint (λ=0)  — original LAM, no signal harvesting
    # init_path = gig.gamma_pts if guided_init else None
    # results.append(joint_ig(
    #     model, x, baseline, N, G=G, n_alternating=n_alternating,
    #     tau=0.005, mu_iter=mu_iter, path_iter=path_iter,
    #     init_path=init_path))

    # 6. Joint*  (λ>0) — full signal-harvesting solution
    init_path_star = gig.gamma_pts if guided_init else None
    results.append(joint_star_ig(
        model, x, baseline, N, G=G, patch_size=patch_size,
        n_alternating=n_alternating, lam=lam, tau=tau,
        mu_iter=mu_iter, path_iter=path_iter,
        init_path=init_path_star))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# §9  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N=50, device=None, min_conf=0.70, guided_init=False,
                   lam=1.0, tau=0.01, skip=0):
    """Full experiment: load model/image, run all 6 methods, print table."""
    from lam import load_image_and_model
    from utilss import get_device

    if device is None:
        device = get_device()

    print("Loading ResNet-50 and image...")
    model, x, baseline, info = load_image_and_model(device, min_conf, skip=skip)

    f_x = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl

    print(f"\nModel : {info['model']}")
    print(f"Source: {info['source']}")
    print(f"Class : {info['target_class']} (conf={info['confidence']:.4f})")
    print(f"f(x) = {f_x:.4f},  f(bl) = {f_bl:.4f},  Δf = {delta_f:.4f}")
    print(f"N = {N},  λ = {lam},  τ = {tau}\n")

    methods = run_all_methods(
        model, x, baseline, N=N,
        lam=lam, tau=tau,
        guided_init=guided_init)

    # ── Print table ──
    hdr = (f"{'Method':<16} {'Var_ν':>10} {'CV²':>8} {'𝒬':>8} "
           f"{'Obj':>10} {'Time':>8}")
    print(hdr)
    print("─" * len(hdr))

    for m in methods:
        d_arr = torch.tensor([s.d_k for s in m.steps], device=device)
        df_arr = torch.tensor([s.delta_f_k for s in m.steps], device=device)
        mu_arr = torch.tensor([s.mu_k for s in m.steps], device=device)
        obj, _, _, _ = compute_signal_harvesting_objective(
            d_arr, df_arr, mu_arr, lam=lam, tau=tau)
        print(f"{m.name:<16} {m.Var_nu:>10.6f} {m.CV2:>8.4f} "
              f"{m.Q:>8.4f} {obj:>10.4f} {m.elapsed_s:>7.1f}s")

    results = {
        "config": {"N": N, "lam": lam, "tau": tau},
        "image_info": info,
        "model_info": {"f_x": f_x, "f_baseline": f_bl,
                       "delta_f": delta_f, "N": N, "device": str(device)},
        "methods": {m.name: m.to_dict() for m in methods},
    }
    return results, methods, model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §10  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json
    from utilss import set_seed
    parser = argparse.ArgumentParser(
        description="Signal-Harvesting IG — unified variational framework")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of interpolation steps N")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Signal-harvesting strength λ (0 = original LAM)")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="L2 admissibility multiplier τ")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-conf", type=float, default=0.70)
    parser.add_argument("--guided-init", action="store_true",
                        help="Initialise Joint/Joint* from Guided IG path")
    # ── Visualisation flags ──
    parser.add_argument("--viz", action="store_true",
                        help="Generate attribution heatmap plot")
    parser.add_argument("--viz-path", type=str,
                        default="attribution_heatmaps.png",
                        help="Output path for heatmap plot")
    parser.add_argument("--viz-fidelity", action="store_true",
                        help="Generate step-fidelity φ_k plot")
    # ── Insertion / Deletion ──
    parser.add_argument("--insdel", action="store_true",
                        help="Compute pixel-based insertion/deletion AUC")
    parser.add_argument("--insdel-steps", type=int, default=100,
                        help="Number of steps for ins/del evaluation")
    parser.add_argument("--viz-insdel", action="store_true",
                        help="Generate insertion/deletion curve plot")
    # ── Region-based Insertion / Deletion ──
    parser.add_argument("--region-insdel", action="store_true",
                        help="Compute region-based ins/del (SIC-style)")
    parser.add_argument("--viz-region-insdel", action="store_true",
                        help="Generate region-based ins/del curve plot")
    parser.add_argument("--patch-size", type=int, default=14,
                        help="Grid patch size for region ins/del")
    parser.add_argument("--no-slic", action="store_true",
                        help="Use grid patches instead of SLIC superpixels")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip", type=int, default=0)
              
    
    args = parser.parse_args()
    set_seed(args.seed)

    from utilss import (
        get_device, run_insertion_deletion, run_region_insertion_deletion,
        visualize_step_fidelity, visualize_insertion_deletion,
    )
    from lam import visualize_attributions

    device = get_device(force=args.device)
    results, methods, model, x, baseline, info = run_experiment(
        N=args.steps, device=device, min_conf=args.min_conf,
        guided_init=args.guided_init, lam=args.lam, tau=args.tau, skip=args.skip)

    # ── Insertion / Deletion ──
    if args.insdel or args.viz_insdel:
        run_insertion_deletion(model, x, baseline, methods,
                               n_steps=args.insdel_steps)

    if args.region_insdel or args.viz_region_insdel:
        run_region_insertion_deletion(
            model, x, baseline, methods,
            patch_size=args.patch_size,
            use_slic=not args.no_slic)

    # ── JSON export ──
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults → {args.json}")

    # ── Visualisation ──
    if args.viz:
        visualize_attributions(x, methods, info, save_path=args.viz_path,
                               delta_f=results["model_info"]["delta_f"])

    if args.viz_fidelity:
        fpath = args.viz_path.replace(".png", "_fidelity.png")
        visualize_step_fidelity(methods, save_path=fpath)

    if args.viz_insdel:
        ipath = args.viz_path.replace(".png", "_insdel.png")
        visualize_insertion_deletion(methods, save_path=ipath)

    if args.viz_region_insdel:
        rpath = args.viz_path.replace(".png", "_region_insdel.png")
        visualize_insertion_deletion(methods, save_path=rpath,
                                     use_region=True)