"""
lamp.py — LAMP: Least Action Movement with Per-step completeness
================================================================

LAMP replaces the CV²(φ) objective in Joint IG with MSE_ν(φ,1):

    MSE_ν(φ,1) = Σ_k ν_k (φ_k − 1)²
               = Var_ν(φ) + (φ̄_ν − 1)²

Motivation
----------
The original LAM objective min Var_ν(φ) enforces φ_k = const (conservation
law / Snell's law analogy). At finite N, the constant is not guaranteed to
equal 1, so μ-optimisation can find degenerate basins where φ_k ≈ c for
some c ≠ 1, achieving high Q but poor faithfulness.

MSE_ν(φ,1) is the natural penalty relaxation of:

    min  Var_ν(φ)   s.t.   φ̄_ν = 1

The constraint φ̄_ν = 1 is the finite-N weighted completeness condition —
it holds automatically as N → ∞ (where Σ d_k → Σ Δf_k by the fundamental
theorem of calculus), but must be enforced explicitly at finite N.

This keeps the LAM theoretical grounding intact (primary objective is still
Var_ν(φ)) while adding a finite-N correction that anchors φ̄_ν to 1.

Methods exported
----------------
    optimize_mu_lamp(d, delta_f, tau, n_iter, lr)  — μ-phase with MSE_ν
    lamp(model, x, baseline, N, ...)               — full LAMP attribution

Usage (drop-in alongside unified_ig.py)
---------------------------------------
    from lamp import lamp
    result = lamp(model, x, baseline, N=50)

All helpers (_straight_line_pass, _build_spatial_groups, _build_path_2d,
_eval_path_batched, _pack_result, etc.) are imported from unified_ig.py to
avoid code duplication and ensure identical experimental conditions.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn

# ── Import shared infrastructure from unified_ig ──────────────────────────
from lam import (
    _straight_line_pass,
    _build_spatial_groups,
    _build_path_2d,
    _eval_path_batched,
    _forward_scalar,
    _gradient_batch,
    _rescale,
    _pack_result,
)
from utilss import (
    AttributionResult,
    compute_all_metrics,
    compute_CV2,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  CORE OBJECTIVE:  MSE_ν(φ, 1)
# ═════════════════════════════════════════════════════════════════════════════

def _mse_nu(d: torch.Tensor, delta_f: torch.Tensor,
            mu: torch.Tensor) -> torch.Tensor:
    """
    Differentiable MSE_ν(φ, 1) = Σ_k ν_k (φ_k − 1)²

    where:
        φ_k  = d_k / Δf_k          (step fidelity)
        ν_k  = μ_k Δf_k² / Σ_j μ_j Δf_j²   (effective measure)

    Decomposition:
        MSE_ν(φ,1) = Var_ν(φ) + (φ̄_ν − 1)²

    The bias² term (φ̄_ν − 1)² is the finite-N completeness correction:
    it penalises solutions where gradients systematically under/over-predict
    output changes on average, which is the failure mode of CV²-based
    optimisation.

    Args:
        d       : (N,) gradient-predicted output changes  d_k = ∇f·Δγ_k
        delta_f : (N,) actual output changes  Δf_k
        mu      : (N,) attribution measure (need not sum to 1 here;
                  softmax is applied outside)

    Returns:
        scalar tensor (differentiable w.r.t. mu)
    """
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))  # φ_k

    # Effective measure ν_k = μ_k Δf_k², normalised
    nu = mu * delta_f ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return torch.tensor(0.0, device=d.device)
    nu = nu / nu_sum

    # MSE_ν(φ, 1) = Σ ν_k (φ_k − 1)²
    mse = (nu * (phi - 1.0) ** 2).sum()
    return mse


def _mse_nu_float(d: torch.Tensor, delta_f: torch.Tensor,
                  mu: torch.Tensor) -> float:
    """Non-differentiable float version for path optimisation comparisons."""
    with torch.no_grad():
        return float(_mse_nu(d, delta_f, mu))


# ═════════════════════════════════════════════════════════════════════════════
# §2  PHASE 1: μ-OPTIMISATION WITH MSE_ν(φ,1)
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu_lamp(d: torch.Tensor, delta_f: torch.Tensor,
                     tau: float = 0.01, n_iter: int = 200,
                     lr: float = 0.05) -> torch.Tensor:
    """
    Find μ minimising  MSE_ν(φ,1) + τ·H(μ).

    Difference from optimize_mu (unified_ig.py):
        unified_ig  →  minimises CV²(φ) = Var_ν(φ) / φ̄²
                        φ̄_ν is free → degenerate basin at φ_k = c ≠ 1
        lamp        →  minimises MSE_ν(φ,1) = Var_ν(φ) + (φ̄_ν − 1)²
                        forces φ̄_ν → 1 (finite-N completeness correction)

    Adam on softmax logits → unconstrained optimisation on the simplex.

    Args:
        d       : (N,) tensor, gradient dot products
        delta_f : (N,) tensor, actual output changes
        tau     : entropy regularisation (prevents μ collapsing to one step)
        n_iter  : Adam iterations
        lr      : Adam learning rate

    Returns:
        mu : (N,) tensor, optimised measure summing to 1
    """
    device = d.device
    N = d.shape[0]

    # Precompute constants w.r.t. μ — hoist out of loop
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d)).detach()  # (N,)
    df2 = (delta_f ** 2).detach()                                        # (N,)

    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mu = torch.softmax(logits, dim=0)

        # ν_k = μ_k Δf_k² / Σ_j μ_j Δf_j²
        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        nu_norm = nu / nu_sum

        # MSE_ν(φ,1) = Σ ν_k (φ_k − 1)²
        #            = Var_ν(φ) + (φ̄_ν − 1)²
        mse = (nu_norm * (phi - 1.0) ** 2).sum()

        # Entropy regularisation: prevents μ collapsing to a single step
        entropy = (mu * torch.log(mu + 1e-15)).sum()

        loss = mse + tau * entropy
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §3  PHASE 2: PATH OPTIMISATION WITH MSE_ν(φ,1)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_path_obj_lamp(d_v: torch.Tensor, df_v: torch.Tensor,
                           mu: torch.Tensor) -> float:
    """
    Path objective: MSE_ν(φ,1) = Σ ν_k (φ_k − 1)²

    This is identical to the path objective already used in unified_ig.py's
    _compute_path_obj — confirming that the path phase was already correct.
    The bug was only in the μ-phase using CV²(φ) instead of MSE_ν(φ,1).

    Kept here as an explicit function for clarity and to allow future
    extension (e.g. adding path-specific regularisation).
    """
    return _mse_nu_float(d_v, df_v, mu)


def optimize_path_lamp(model: nn.Module, x: torch.Tensor,
                       baseline: torch.Tensor, mu: torch.Tensor,
                       N: int = 50, G: int = 16, patch_size: int = 14,
                       n_iter: int = 15, lr: float = 0.08) -> list:
    """
    Path optimisation with MSE_ν(φ,1) objective.

    Identical structure to optimize_path in unified_ig.py but uses
    _compute_path_obj_lamp (MSE_ν) instead of compute_CV2.

    The path parameterisation (grouped velocity scheduling) is unchanged:
        V ∈ R^{G×N}  controls when each spatial group's displacement
        is delivered along the interpolation.

    Stochastic FD: one random time step per group per iteration.
    Cost: 2 batched model calls per FD probe × G probes per iteration.
    """
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    # Initialise: uniform velocity = straight line
    V = torch.ones(G, N, device=device)
    best_mse = float("inf")
    best_V = V.clone()

    def _mse_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _compute_path_obj_lamp(d_v, df_v, mu)

    eps = 0.05
    for it in range(n_iter):
        mse = _mse_of(V)
        if mse < best_mse:
            best_mse = mse
            best_V = V.clone()

        # Stochastic FD: perturb one random time step per group
        grad_V = torch.zeros_like(V)
        for g in range(G):
            k = torch.randint(0, N, (1,)).item()
            V[g, k] += eps
            mse_plus = _mse_of(V)
            grad_V[g, k] = (mse_plus - mse) / eps
            V[g, k] -= eps

        V = V - lr * grad_V
        V = torch.clamp(V, min=0.01)

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)


# ═════════════════════════════════════════════════════════════════════════════
# §4  LAMP: FULL ALTERNATING MINIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def lamp(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
         N: int = 50, G: int = 16, patch_size: int = 14,
         n_alternating: int = 2, tau: float = 0.005,
         mu_iter: int = 300, path_iter: int = 10,
         init_path: Optional[list] = None,
         ) -> AttributionResult:
    """
    LAMP: Least Action Movement with Per-step completeness.

    Alternating minimisation of MSE_ν(φ,1) over (γ, μ):

        Phase 1 (μ): fix γ, minimise MSE_ν(φ,1) + τ·H(μ) via Adam
        Phase 2 (γ): fix μ, minimise MSE_ν(φ,1) via velocity scheduling

    Difference from joint_ig (unified_ig.py):
    ┌─────────────────┬──────────────────┬──────────────────┐
    │                 │ Joint            │ LAMP             │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ μ objective     │ CV²(φ)           │ MSE_ν(φ,1)       │
    │ path objective  │ MSE_ν(φ,1) ✓    │ MSE_ν(φ,1) ✓    │
    │ regression guard│ Q-based          │ MSE-based        │
    │ theory          │ LAM (Var_ν only) │ LAM + completeness│
    └─────────────────┴──────────────────┴──────────────────┘

    Note: the path phase was already using MSE_ν(φ,1) in unified_ig.py
    (_compute_path_obj). LAMP makes the μ-phase consistent with the path
    phase — a single unified objective throughout.

    Parameters
    ----------
    model        : ClassLogitModel wrapping backbone
    x            : (1,C,H,W) input tensor
    baseline     : (1,C,H,W) baseline tensor (typically zeros)
    N            : number of interpolation steps
    G            : number of spatial groups for path optimisation
    patch_size   : patch size for spatial grouping
    n_alternating: number of alternating iterations
    tau          : entropy regularisation for μ-optimisation
    mu_iter      : Adam iterations for μ-phase
    path_iter    : FD iterations for path-phase
    init_path    : optional list of N+1 tensors as initial path
                   (e.g. gamma_pts from guided_ig for warm start)

    Returns
    -------
    AttributionResult with name="LAMP"
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    # ── Initialise path ───────────────────────────────────────────────────
    if init_path is not None:
        assert len(init_path) == N + 1, \
            f"init_path must have N+1={N+1} points, got {len(init_path)}"
        gamma_pts = [p.clone() for p in init_path]
    else:
        # Straight line: γ_k = x' + (k/N)(x − x')
        gamma_pts = [baseline + (k / N) * delta_x for k in range(N + 1)]

    mu = torch.full((N,), 1.0 / N, device=device)

    # ── Best-state tracking (regression guard) ────────────────────────────
    best_mse = float("inf")
    best_gamma_pts = gamma_pts
    best_mu = mu.clone()
    best_d_list, best_df_list = [], []
    best_f_vals, best_gnorms, best_grads = [], [], []
    Q_history = []

    def _evaluate_path(gp, mu_vec):
        """
        Batched evaluation matching joint_ig._evaluate_path convention:
          - f_vals: [f(γ_0)] + [f(γ_0)..f(γ_{N-1})] + [f(γ_N)]  (N+2 entries)
          - d[k] = grad(γ_k) · (γ_{k+1} − γ_k)
          - df[k] = f_vals[k+1] − f_vals[k]  (backward-looking)
        """
        ap = torch.cat(gp, dim=0)                           # (N+1, C, H, W)
        with torch.no_grad():
            fa = model(ap)                                   # (N+1,)
        pn = ap[:N]                                          # γ_0..γ_{N-1}
        gb = _gradient_batch(model, pn)                      # (N, C, H, W)
        sb = ap[1:] - ap[:N]                                 # steps
        dt = (gb * sb).view(N, -1).sum(dim=1)               # d_k  (N,)
        dl = dt.tolist()

        f0 = fa[0].item()
        fv = [f0] + fa.tolist()                              # N+2 entries
        dfl = [fv[k + 1] - fv[k] for k in range(N)]

        gn = gb.view(N, -1).norm(dim=1).tolist()
        gr = [gb[k:k+1].clone() for k in range(N)]
        da = torch.tensor(dl, device=device)
        dfa = torch.tensor(dfl, device=device)
        return dl, dfl, fv, gn, gr, da, dfa

    # ── Alternating minimisation ──────────────────────────────────────────
    for s in range(n_alternating):

        # Evaluate current path
        d_list, df_list, f_vals, gnorms, grads, d_arr, df_arr = \
            _evaluate_path(gamma_pts, mu)

        # ── Phase 1: μ-optimisation with MSE_ν(φ,1) ──────────────────────
        mu = optimize_mu_lamp(d_arr, df_arr, tau=tau, n_iter=mu_iter)

        # Compute MSE_ν and standard metrics for tracking
        mse_mu = _mse_nu_float(d_arr, df_arr, mu)
        _, _, Q_mu = compute_all_metrics(d_arr, df_arr, mu)

        # Regression guard: keep best state by MSE_ν (lower = better)
        if mse_mu < best_mse:
            best_mse = mse_mu
            best_gamma_pts = gamma_pts
            best_mu = mu.clone()
            best_d_list, best_df_list = d_list, df_list
            best_f_vals, best_gnorms, best_grads = f_vals, gnorms, grads

        # ── Phase 2: path optimisation with MSE_ν(φ,1) ───────────────────
        mse_path = mse_mu
        if s < n_alternating - 1:
            new_gamma_pts = optimize_path_lamp(
                model, x, baseline, mu, N=N, G=G,
                patch_size=patch_size, n_iter=path_iter)

            new_d_list, new_df_list, new_f_vals, new_gnorms, new_grads, \
                new_d_arr, new_df_arr = _evaluate_path(new_gamma_pts, mu)

            mse_new = _mse_nu_float(new_d_arr, new_df_arr, mu)
            _, _, Q_new = compute_all_metrics(new_d_arr, new_df_arr, mu)
            mse_path = mse_new

            # Accept only if MSE_ν improves (regression guard)
            if mse_new < best_mse:
                gamma_pts = new_gamma_pts
                d_list, df_list = new_d_list, new_df_list
                f_vals, gnorms, grads = new_f_vals, new_gnorms, new_grads
                d_arr, df_arr = new_d_arr, new_df_arr

                best_mse = mse_new
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
            "MSE_after_mu": float(mse_mu),
            "MSE_after_path": float(mse_path),
            "Q_after_mu": float(Q_mu),
            "best_MSE": float(best_mse),
        })

    # ── Final attribution using best state ────────────────────────────────
    gamma_pts = best_gamma_pts
    mu = best_mu
    grads = best_grads

    attr = torch.zeros_like(x)
    for k in range(N):
        attr += mu[k] * grads[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    return _pack_result(
        "LAMP", attr,
        best_d_list, best_df_list, best_f_vals, best_gnorms,
        mu, N, t0, Q_history,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §5  μ-LAMP: STRAIGHT LINE + MSE_ν(φ,1) MEASURE ONLY
#     (ablation: isolates the μ-objective change from path optimisation)
# ═════════════════════════════════════════════════════════════════════════════

def mu_lamp(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
            N: int = 50, tau: float = 0.005, n_iter: int = 300,
            ) -> AttributionResult:
    """
    Straight line + MSE_ν(φ,1)-optimal μ.

    Ablation counterpart to mu_optimized_ig (unified_ig.py):
        mu_optimized_ig  →  straight line + CV²(φ)-optimal μ
        mu_lamp          →  straight line + MSE_ν(φ,1)-optimal μ

    This isolates the effect of the objective change (CV² → MSE_ν)
    from the effect of path optimisation. If mu_lamp > mu_optimized_ig
    on faithfulness metrics, the μ-objective is the bottleneck.
    If mu_lamp ≈ mu_optimized_ig, the path is the bottleneck.

    Cost: identical to mu_optimized_ig (zero additional model evaluations
    beyond _straight_line_pass).
    """
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)

    # MSE_ν(φ,1) optimisation instead of CV²
    mu = optimize_mu_lamp(d_arr, df_arr, tau=tau, n_iter=n_iter)

    # Weighted gradient sum
    grad_stack = torch.cat(grads, dim=0)               # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = _rescale(delta_x * wg, target)

    return _pack_result("μ-LAMP", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §6  DIAGNOSTIC: decompose MSE_ν into Var_ν + bias²
# ═════════════════════════════════════════════════════════════════════════════

def decompose_mse(d: torch.Tensor, delta_f: torch.Tensor,
                  mu: torch.Tensor) -> dict:
    """
    Decompose MSE_ν(φ,1) = Var_ν(φ) + (φ̄_ν − 1)²

    Useful for diagnosing whether faithfulness gap is due to:
        - high variance (path is noisy) → Var_ν dominates
        - high bias (gradients systematically mis-predict) → bias² dominates

    Returns dict with keys: mse, var_nu, bias_sq, phi_bar, Q
    """
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))

    nu = mu * delta_f ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return {"mse": 0.0, "var_nu": 0.0, "bias_sq": 0.0,
                "phi_bar": 1.0, "Q": 1.0}
    nu = nu / nu_sum

    phi_bar = float((nu * phi).sum())
    var_nu = float((nu * (phi - phi_bar) ** 2).sum())
    bias_sq = (phi_bar - 1.0) ** 2
    mse = var_nu + bias_sq

    cv2 = var_nu / (phi_bar ** 2 + 1e-15)
    Q = 1.0 / (1.0 + cv2)

    return {
        "mse": mse,
        "var_nu": var_nu,
        "bias_sq": bias_sq,
        "phi_bar": phi_bar,
        "Q": Q,
    }


def print_decomposition(methods: list[AttributionResult],
                        device: torch.device) -> None:
    """
    Print MSE_ν decomposition for a list of AttributionResult objects.
    Useful for diagnosing why a method has poor faithfulness.
    """
    print(f"\n{'Method':<16} {'MSE_ν':>8} {'Var_ν':>8} "
          f"{'bias²':>8} {'φ̄_ν':>8} {'Q':>8}")
    print("─" * 60)
    for m in methods:
        d_arr = torch.tensor([s.d_k for s in m.steps], device=device)
        df_arr = torch.tensor([s.delta_f_k for s in m.steps], device=device)
        mu_arr = torch.tensor([s.mu_k for s in m.steps], device=device)
        dec = decompose_mse(d_arr, df_arr, mu_arr)
        print(f"{m.name:<16} {dec['mse']:>8.4f} {dec['var_nu']:>8.4f} "
              f"{dec['bias_sq']:>8.4f} {dec['phi_bar']:>8.4f} "
              f"{dec['Q']:>8.4f}")