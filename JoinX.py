"""
unified_ig.py — Unified Integrated Gradients Framework (PyTorch)
=================================================================

A unified framework for Integrated Gradients (IG) attribution methods,
showing that IG, IDGI, and Guided IG all optimise the same objective:

    𝒬(γ, μ) = 1 / (1 + CV²(φ))

where φ_k = d_k / Δf_k is the step fidelity ratio between the gradient-
predicted output change d_k and the actual output change Δf_k at each
interpolation step k.  The quality metric 𝒬 equals the squared cosine
similarity between d and Δf under the μ-weighted inner product:

    𝒬 = (Σ_k μ_k d_k Δf_k)² / [(Σ_k μ_k d_k²)(Σ_k μ_k Δf_k²)]

Methods differ only in which degree of freedom they optimise:
    ┌─────────────┬──────────┬────────┬───────────────────────────────┐
    │ Method      │ Path γ   │ Meas μ │ Strategy                      │
    ├─────────────┼──────────┼────────┼───────────────────────────────┤
    │ Standard IG │ fixed    │ fixed  │ straight line, uniform        │
    │ IDGI        │ fixed    │ heur.  │ straight line, μ_k ∝ |Δf_k|  │
    │ Guided IG   │ heur.    │ fixed  │ low-grad-first path, uniform  │
    │ μ-Optimised │ fixed    │ opt.   │ straight line, min CV²(φ)     │
    │ Joint       │ opt.     │ opt.   │ alternating minimisation       │
    └─────────────┴──────────┴────────┴───────────────────────────────┘

Usage
-----
    python unified_ig.py                        # single-seed results
    python unified_ig.py --multi-seed           # aggregate over 8 seeds
    python unified_ig.py --json results.json    # export JSON

Requirements: torch >= 2.0

References
----------
    Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
    Kapishnikov et al., "Guided Integrated Gradients" (NeurIPS 2021)
    Sikdar et al., "Integrated Directional Gradients" (ACL 2021)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# §1  DEVICE SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """Auto-select best available device: CUDA → MPS → CPU."""
    return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] MPS (Apple Silicon)")
    else:
        dev = torch.device("cpu")
        print("[device] CPU")
    return dev


# ═════════════════════════════════════════════════════════════════════════════
# §2  MODEL
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    3-layer network: f(x) = Linear(ReLU(Linear(Tanh(Linear(x)))))

    Architecture: input_dim → hidden1 (tanh) → hidden2 (relu) → 1 (linear)

    The tanh→ReLU composition creates genuine nonlinearity, unlike pure-ReLU
    networks which are piecewise linear and give deceptively high fidelity.
    """

    def __init__(self, input_dim: int = 10, hidden1: int = 20,
                 hidden2: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1, bias=False)

        # Controlled initialisation for reproducible behaviour
        nn.init.normal_(self.fc1.weight, std=0.6)
        nn.init.normal_(self.fc1.bias, std=0.2)
        nn.init.normal_(self.fc2.weight, std=0.5)
        nn.init.normal_(self.fc2.bias, std=0.1)
        nn.init.normal_(self.fc3.weight, std=0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (..., input_dim) — supports batched and single inputs.

        Returns
        -------
        Scalar (or batch of scalars) output.
        """
        h = torch.tanh(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# §3  DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StepInfo:
    """Per-step diagnostics along the interpolation path."""
    t: float            # interpolation parameter ∈ [0, 1)
    f: float            # f(γ(t_k))
    d_k: float          # gradient prediction:  ∇f(γ(t_k)) · Δγ_k
    delta_f_k: float    # actual output change:  f(γ(t_{k+1})) − f(γ(t_k))
    r_k: float          # linearisation residual: Δf_k − d_k
    phi_k: float        # step fidelity: d_k / Δf_k
    grad_norm: float    # ‖∇f(γ(t_k))‖₂
    mu_k: float         # attribution weight at this step


@dataclass
class AttributionResult:
    """Complete output of an attribution method."""
    name: str
    attributions: torch.Tensor          # (n,) per-feature attributions
    Q: float                            # quality metric 𝒬
    CV2: float                          # CV²(φ) under effective measure
    steps: list[StepInfo]               # per-step diagnostics
    Q_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "attributions": self.attributions.cpu().tolist(),
            "Q": self.Q,
            "CV2": self.CV2,
            "steps": [asdict(s) for s in self.steps],
            "Q_history": self.Q_history,
        }


# ═════════════════════════════════════════════════════════════════════════════
# §4  QUALITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_Q(d: torch.Tensor, delta_f: torch.Tensor,
              mu: torch.Tensor) -> float:
    """
    Quality metric: squared cosine similarity between d and Δf under μ.

        𝒬 = (Σ μ_k d_k Δf_k)² / [(Σ μ_k d_k²)(Σ μ_k Δf_k²)]

    Returns 1.0 when d_k = α·Δf_k for all k with μ_k > 0.
    """
    num = (mu * d * delta_f).sum() ** 2
    den1 = (mu * d ** 2).sum()
    den2 = (mu * delta_f ** 2).sum()
    if den1 < 1e-15 or den2 < 1e-15:
        return 0.0
    return float(num / (den1 * den2))


def compute_CV2(d: torch.Tensor, delta_f: torch.Tensor,
                mu: torch.Tensor) -> float:
    """
    Squared coefficient of variation of step fidelity φ_k = d_k / Δf_k
    under effective measure ν_k ∝ μ_k Δf_k².

    Satisfies: 𝒬 = 1 / (1 + CV²(φ)).
    """
    valid = delta_f.abs() > 1e-12
    if valid.sum() < 2:
        return 0.0
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))
    nu = mu * delta_f ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return 0.0
    w = nu / nu_sum
    mean_phi = (w * phi).sum()
    var_phi = (w * (phi - mean_phi) ** 2).sum()
    if mean_phi.abs() < 1e-12:
        return float("inf")
    return float(var_phi / mean_phi ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# §5  GRADIENT UTILITY
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_scalar(model: nn.Module, x: torch.Tensor) -> float:
    """Evaluate f(x) as a Python float (no grad tracking)."""
    return float(model(x))


def _gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇_x f(x) for a scalar-output model.

    Returns a detached gradient tensor on the same device as x.
    """
    x_in = x.detach().requires_grad_(True)
    out = model(x_in)
    out.backward()
    grad = x_in.grad.detach().clone()
    x_in.requires_grad_(False)
    return grad


# ═════════════════════════════════════════════════════════════════════════════
# §6  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _evaluate_path(
    model: nn.Module,
    gamma: torch.Tensor,        # (N+1, n)
    mu: torch.Tensor,           # (N,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[StepInfo]]:
    """
    Evaluate a path: compute d_k, Δf_k, attributions, and diagnostics.

    Returns (attributions, d_vals, delta_f_vals, steps_info).
    """
    N = mu.shape[0]
    n = gamma.shape[1]
    device = gamma.device

    d_vals = torch.zeros(N, device=device)
    df_vals = torch.zeros(N, device=device)
    attr = torch.zeros(n, device=device)
    steps: list[StepInfo] = []

    for k in range(N):
        f_k = _forward_scalar(model, gamma[k])
        f_k1 = _forward_scalar(model, gamma[k + 1])
        grad_k = _gradient(model, gamma[k])
        step_k = gamma[k + 1] - gamma[k]

        d_k = float(grad_k @ step_k)
        delta_f_k = f_k1 - f_k
        r_k = delta_f_k - d_k
        phi_k = d_k / delta_f_k if abs(delta_f_k) > 1e-12 else 1.0

        d_vals[k] = d_k
        df_vals[k] = delta_f_k
        attr += mu[k] * grad_k * step_k

        steps.append(StepInfo(
            t=k / N, f=f_k, d_k=d_k, delta_f_k=delta_f_k,
            r_k=r_k, phi_k=phi_k,
            grad_norm=float(grad_k.norm()), mu_k=float(mu[k]),
        ))

    return attr, d_vals, df_vals, steps


def _rescale_for_completeness(attr: torch.Tensor, target: float) -> torch.Tensor:
    """Scale attributions so Σ A_i = f(x) − f(x') (completeness axiom)."""
    s = attr.sum().item()
    if abs(s) > 1e-12:
        return attr * (target / s)
    return attr


# ═════════════════════════════════════════════════════════════════════════════
# §7  STANDARD IG
# ═════════════════════════════════════════════════════════════════════════════

def standard_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                N: int = 50) -> AttributionResult:
    """
    Standard Integrated Gradients (Sundararajan et al., 2017).

    Straight-line path γ(t) = x' + t(x − x'), uniform measure μ_k = 1/N.

        A_i = Δx_i · (1/N) Σ_k ∂f/∂x_i(γ(t_k))
    """
    device = x.device
    delta_x = x - baseline
    mu = torch.full((N,), 1.0 / N, device=device)

    grad_sum = torch.zeros_like(x)
    steps: list[StepInfo] = []

    for k in range(N):
        t_k = k / N
        gamma_k = baseline + t_k * delta_x
        gamma_k1 = baseline + (t_k + 1.0 / N) * delta_x

        f_k = _forward_scalar(model, gamma_k)
        f_k1 = _forward_scalar(model, gamma_k1)
        grad_k = _gradient(model, gamma_k)

        d_k = float(grad_k @ (delta_x / N))
        df_k = f_k1 - f_k
        r_k = df_k - d_k
        phi_k = d_k / df_k if abs(df_k) > 1e-12 else 1.0

        grad_sum += grad_k
        steps.append(StepInfo(
            t=t_k, f=f_k, d_k=d_k, delta_f_k=df_k,
            r_k=r_k, phi_k=phi_k,
            grad_norm=float(grad_k.norm()), mu_k=1.0 / N,
        ))

    attr = delta_x * grad_sum / N
    d = torch.tensor([s.d_k for s in steps], device=device)
    df = torch.tensor([s.delta_f_k for s in steps], device=device)

    return AttributionResult(
        name="IG", attributions=attr,
        Q=compute_Q(d, df, mu), CV2=compute_CV2(d, df, mu), steps=steps,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §8  IDGI
# ═════════════════════════════════════════════════════════════════════════════

def idgi(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
         N: int = 50) -> AttributionResult:
    """
    Integrated Directional Gradients (Sikdar et al., 2021).

    Straight-line path, measure μ_k ∝ |Δf_k|.

        A_i = Δx_i · Σ_k μ_k ∂f/∂x_i(γ(t_k))
    """
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    grads: list[torch.Tensor] = []
    d_list: list[float] = []
    df_list: list[float] = []

    for k in range(N):
        t_k = k / N
        gk = baseline + t_k * delta_x
        gk1 = baseline + (t_k + 1.0 / N) * delta_x
        grad_k = _gradient(model, gk)
        d_k = float(grad_k @ (delta_x / N))
        df_k = _forward_scalar(model, gk1) - _forward_scalar(model, gk)
        grads.append(grad_k)
        d_list.append(d_k)
        df_list.append(df_k)

    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)

    # IDGI weighting: μ_k ∝ |Δf_k|
    weights = df_arr.abs()
    w_sum = weights.sum()
    mu = weights / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0 / N, device=device)

    # Attribution
    wg = sum(mu[k].item() * grads[k] for k in range(N))
    attr = _rescale_for_completeness(delta_x * wg, target)

    # Diagnostics
    steps: list[StepInfo] = []
    for k in range(N):
        r_k = df_list[k] - d_list[k]
        phi_k = d_list[k] / df_list[k] if abs(df_list[k]) > 1e-12 else 1.0
        steps.append(StepInfo(
            t=k / N,
            f=_forward_scalar(model, baseline + (k / N) * delta_x),
            d_k=d_list[k], delta_f_k=df_list[k], r_k=r_k, phi_k=phi_k,
            grad_norm=float(grads[k].norm()), mu_k=float(mu[k]),
        ))

    return AttributionResult(
        name="IDGI", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §9  GUIDED IG
# ═════════════════════════════════════════════════════════════════════════════

def guided_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
              N: int = 50) -> AttributionResult:
    """
    Guided Integrated Gradients (Kapishnikov et al., 2021).

    Heuristic path: move low-gradient dimensions first, deferring high-
    gradient dimensions to later steps. Uniform measure.
    """
    device = x.device
    n = x.shape[0]
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    remaining = delta_x.clone()
    current = baseline.clone()
    gamma_pts: list[torch.Tensor] = [current.clone()]
    grad_list: list[torch.Tensor] = []
    steps: list[StepInfo] = []

    for k in range(N):
        f_k = _forward_scalar(model, current)
        grad_k = _gradient(model, current)
        grad_list.append(grad_k)

        # Inverse-gradient weighting: move low-|grad| dims faster
        abs_g = grad_k.abs() + 1e-8
        inv_w = 1.0 / abs_g
        frac = inv_w / inv_w.sum()
        remaining_steps = N - k

        # Per-dimension step (don't overshoot)
        raw_step = remaining.abs() * frac * remaining_steps
        step = remaining.sign() * torch.minimum(raw_step, remaining.abs())

        next_pt = current + step
        f_k1 = _forward_scalar(model, next_pt)

        d_k = float(grad_k @ step)
        df_k = f_k1 - f_k
        r_k = df_k - d_k
        phi_k = d_k / df_k if abs(df_k) > 1e-12 else 1.0

        steps.append(StepInfo(
            t=k / N, f=f_k, d_k=d_k, delta_f_k=df_k,
            r_k=r_k, phi_k=phi_k,
            grad_norm=float(grad_k.norm()), mu_k=1.0 / N,
        ))

        remaining = remaining - step
        current = next_pt
        gamma_pts.append(current.clone())

    # Attribution: Σ_k grad_k ⊙ Δγ_k  (uniform μ)
    attr = torch.zeros(n, device=device)
    for k in range(N):
        attr += grad_list[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale_for_completeness(attr, target)

    d_arr = torch.tensor([s.d_k for s in steps], device=device)
    df_arr = torch.tensor([s.delta_f_k for s in steps], device=device)
    mu = torch.full((N,), 1.0 / N, device=device)

    return AttributionResult(
        name="Guided IG", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §10  μ-OPTIMISATION (Phase 1)
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu(d: torch.Tensor, delta_f: torch.Tensor,
                tau: float = 0.01, n_iter: int = 200,
                lr: float = 0.05) -> torch.Tensor:
    """
    Find μ ∈ Δ_N minimising Var_ν(φ) + τ·H(μ)  (entropy regularised).

    Uses softmax reparametrisation for unconstrained gradient descent.

    Parameters
    ----------
    d       : (N,) gradient-predicted step changes.
    delta_f : (N,) actual step changes.
    tau     : entropy regularisation (larger → more uniform μ).
    n_iter  : gradient-descent iterations.
    lr      : learning rate.

    Returns
    -------
    mu : (N,) optimised measure on the probability simplex.
    """
    device = d.device
    N = d.shape[0]

    # Pre-compute φ_k (fixed for a given path)
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))
    df2 = delta_f ** 2

    # Softmax logits (learnable)
    logits = torch.zeros(N, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        optimiser.zero_grad()
        mu = torch.softmax(logits, dim=0)

        # Effective measure ν_k = μ_k Δf_k², normalised
        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        w = nu / nu_sum

        mean_phi = (w * phi).sum()
        var_phi = (w * (phi - mean_phi) ** 2).sum()

        # Entropy regularisation (negative entropy = Σ μ log μ)
        entropy = (mu * torch.log(mu + 1e-15)).sum()

        loss = var_phi + tau * entropy
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §11  μ-ONLY ATTRIBUTION (Shortcut: straight line + optimised μ)
# ═════════════════════════════════════════════════════════════════════════════

def mu_optimized_ig(model: nn.Module, x: torch.Tensor,
                    baseline: torch.Tensor, N: int = 50,
                    tau: float = 0.005, n_iter: int = 300) -> AttributionResult:
    """
    Optimised-μ IG: straight-line path with μ minimising CV²(φ).

    Computational cost identical to standard IG (reuses the same gradients).
    The μ-optimisation is pure arithmetic — a free improvement.
    """
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    grads: list[torch.Tensor] = []
    d_list: list[float] = []
    df_list: list[float] = []

    for k in range(N):
        t_k = k / N
        gk = baseline + t_k * delta_x
        gk1 = baseline + (t_k + 1.0 / N) * delta_x
        grad_k = _gradient(model, gk)
        grads.append(grad_k)
        d_list.append(float(grad_k @ (delta_x / N)))
        df_list.append(_forward_scalar(model, gk1) - _forward_scalar(model, gk))

    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)

    mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=n_iter)

    # Attribution
    wg = sum(mu[k].item() * grads[k] for k in range(N))
    attr = _rescale_for_completeness(delta_x * wg, target)

    steps: list[StepInfo] = []
    for k in range(N):
        r_k = df_list[k] - d_list[k]
        phi_k = d_list[k] / df_list[k] if abs(df_list[k]) > 1e-12 else 1.0
        steps.append(StepInfo(
            t=k / N,
            f=_forward_scalar(model, baseline + (k / N) * delta_x),
            d_k=d_list[k], delta_f_k=df_list[k], r_k=r_k, phi_k=phi_k,
            grad_norm=float(grads[k].norm()), mu_k=float(mu[k]),
        ))

    return AttributionResult(
        name="μ-Optimized", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §12  PATH OPTIMISATION (Phase 2)
# ═════════════════════════════════════════════════════════════════════════════

def _build_path_from_velocity(
    baseline: torch.Tensor, delta_x: torch.Tensor,
    V: torch.Tensor, group_idx: torch.Tensor, N: int,
) -> torch.Tensor:
    """
    Construct (N+1, n) path from grouped velocity schedule V (G, N).

    V[g, k] controls how much of group g's displacement is delivered at
    step k. The total displacement per group is guaranteed by normalisation:
    Δγ_{k,i} = Δx_i · V[g(i), k] / Σ_j V[g(i), j].
    """
    n = baseline.shape[0]
    device = baseline.device
    gamma = torch.zeros(N + 1, n, device=device)
    gamma[0] = baseline.clone()

    G = V.shape[0]
    for k in range(N):
        step = torch.zeros(n, device=device)
        for g in range(G):
            mask = group_idx == g
            v_sum = V[g].sum()
            if v_sum > 1e-12:
                step[mask] = delta_x[mask] * V[g, k] / v_sum
        gamma[k + 1] = gamma[k] + step
    return gamma


def optimize_path(
    model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
    mu: torch.Tensor, N: int = 50, G: int = 5,
    n_iter: int = 30, lr: float = 0.1,
) -> torch.Tensor:
    """
    Phase 2: optimise path via grouped velocity scheduling.

    Features are partitioned into G groups by gradient-weighted importance
    at the midpoint. The velocity schedule V ∈ ℝ^{G×N} determines when
    each group's displacement is delivered.

    Optimisation uses finite-difference gradients of CV²(φ) w.r.t. V.

    Returns
    -------
    gamma : (N+1, n) optimised path from baseline to x.
    """
    device = x.device
    n = x.shape[0]
    delta_x = x - baseline

    # ── Group features by midpoint importance ──
    mid = baseline + 0.5 * delta_x
    grad_mid = _gradient(model, mid)
    importance = (grad_mid * delta_x).abs()
    order = torch.argsort(importance)
    group_idx = torch.zeros(n, dtype=torch.long, device=device)
    group_size = n // G
    for g in range(G):
        lo = g * group_size
        hi = (g + 1) * group_size if g < G - 1 else n
        group_idx[order[lo:hi]] = g

    # ── Initialise velocity: uniform ──
    V = torch.ones(G, N, device=device)
    best_cv = float("inf")
    best_V = V.clone()

    def _cv_of(Vmat: torch.Tensor) -> float:
        gp = _build_path_from_velocity(baseline, delta_x, Vmat, group_idx, N)
        d_v = torch.zeros(N, device=device)
        df_v = torch.zeros(N, device=device)
        for kk in range(N):
            grd = _gradient(model, gp[kk])
            d_v[kk] = grd @ (gp[kk + 1] - gp[kk])
            df_v[kk] = _forward_scalar(model, gp[kk + 1]) - _forward_scalar(model, gp[kk])
        return compute_CV2(d_v, df_v, mu)

    # ── Finite-difference optimisation ──
    eps = 0.05
    for _ in range(n_iter):
        cv2 = _cv_of(V)
        if cv2 < best_cv:
            best_cv = cv2
            best_V = V.clone()

        grad_V = torch.zeros_like(V)
        for g in range(G):
            for k in range(N):
                V[g, k] += eps
                grad_V[g, k] = (_cv_of(V) - cv2) / eps
                V[g, k] -= eps

        V = V - lr * grad_V
        V = torch.clamp(V, min=0.01)

    return _build_path_from_velocity(baseline, delta_x, best_V, group_idx, N)


# ═════════════════════════════════════════════════════════════════════════════
# §13  JOINT OPTIMISATION (Path + Measure)
# ═════════════════════════════════════════════════════════════════════════════

def joint_ig(
    model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
    N: int = 50, G: int = 5, n_alternating: int = 3,
    tau: float = 0.005, mu_iter: int = 300, path_iter: int = 15,
) -> AttributionResult:
    """
    Joint optimisation of path γ and measure μ via alternating minimisation
    of CV²(φ).

    Algorithm
    ---------
    1. Initialise γ = straight line, μ = uniform.
    2. For s = 1 … n_alternating:
       a. Phase 1 — optimise μ given γ  (gradient descent on simplex)
       b. Phase 2 — optimise γ given μ  (grouped velocity scheduling)
    3. Compute attributions with final (γ, μ).

    Computational cost: ~(2·n_alternating + 1) × standard IG.
    """
    device = x.device
    n = x.shape[0]
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    # Initialise
    alphas = torch.linspace(0, 1, N + 1, device=device).unsqueeze(1)  # (N+1, 1)
    gamma = baseline.unsqueeze(0) + alphas * delta_x.unsqueeze(0)      # (N+1, n)
    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history: list[dict] = []

    for s in range(n_alternating):
        _, d_vals, df_vals, _ = _evaluate_path(model, gamma, mu)

        # Phase 1: optimise μ
        mu = optimize_mu(d_vals, df_vals, tau=tau, n_iter=mu_iter)
        Q_mu = compute_Q(d_vals, df_vals, mu)
        cv2_mu = compute_CV2(d_vals, df_vals, mu)

        # Phase 2: optimise path (skip on final iteration)
        if s < n_alternating - 1:
            gamma = optimize_path(model, x, baseline, mu, N=N, G=G,
                                  n_iter=path_iter)
            _, d_new, df_new, _ = _evaluate_path(model, gamma, mu)
            Q_path = compute_Q(d_new, df_new, mu)
        else:
            Q_path = Q_mu

        Q_history.append({
            "iteration": s,
            "Q_after_mu": float(Q_mu),
            "Q_after_path": float(Q_path),
            "CV2_after_mu": float(cv2_mu),
        })

    # Final attributions
    attr, d_final, df_final, steps = _evaluate_path(model, gamma, mu)
    attr = _rescale_for_completeness(attr, target)

    return AttributionResult(
        name="Joint", attributions=attr,
        Q=compute_Q(d_final, df_final, mu),
        CV2=compute_CV2(d_final, df_final, mu),
        steps=steps, Q_history=Q_history,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §14  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(input_dim: int = 10, N: int = 20,
                   seed: int = 42, device: Optional[torch.device] = None,
                   ) -> dict:
    """
    Run all five methods on a synthetic example.

    Parameters
    ----------
    input_dim : number of input features.
    N         : number of interpolation steps.
    seed      : random seed.
    device    : torch device (auto-detected if None).

    Returns
    -------
    results : dict suitable for JSON serialisation.
    """
    if device is None:
        device = get_device()

    torch.manual_seed(seed)
    model = MLP(input_dim=input_dim, hidden1=20, hidden2=16).to(device).eval()

    torch.manual_seed(seed + 1000)   # separate seed for input
    x = (torch.randn(input_dim, device=device) * 2.0)
    baseline = torch.zeros(input_dim, device=device)

    f_x = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl

    print(f"Model : MLP(tanh→ReLU→linear), {input_dim}→20→16→1")
    print(f"Device: {device}")
    print(f"f(x) = {f_x:.4f},  f(baseline) = {f_bl:.4f},  Δf = {delta_f:.4f}")
    print(f"N = {N} interpolation steps\n")
    print(f"{'Method':<16} {'𝒬':>8} {'CV²(φ)':>10} {'Σ Aᵢ':>10}")
    print("─" * 48)

    methods = [
        standard_ig(model, x, baseline, N),
        idgi(model, x, baseline, N),
        guided_ig(model, x, baseline, N),
        mu_optimized_ig(model, x, baseline, N, tau=0.005, n_iter=300),
        joint_ig(model, x, baseline, N, G=5, n_alternating=3,
                 tau=0.005, mu_iter=300, path_iter=15),
    ]

    for m in methods:
        sa = m.attributions.sum().item()
        print(f"{m.name:<16} {m.Q:>8.4f} {m.CV2:>10.4f} {sa:>10.4f}")

    results = {
        "model_info": {
            "input_dim": input_dim,
            "architecture": f"{input_dim}→20(tanh)→16(relu)→1",
            "f_x": f_x, "f_baseline": f_bl, "delta_f": delta_f, "N": N,
            "x": x.cpu().tolist(), "baseline": baseline.cpu().tolist(),
            "device": str(device),
        },
        "feature_names": [f"x{i}" for i in range(input_dim)],
        "methods": {m.name: m.to_dict() for m in methods},
    }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# §15  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified IG Framework (PyTorch)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--dim", type=int, default=50,
                        help="Input dimension (default: 10)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Interpolation steps N (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Aggregate experiment over 8 seeds")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cuda, mps, or cpu")
    args = parser.parse_args()

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
        print(f"[device] {device} (forced)")
    else:
        device = get_device()

    if args.multi_seed:
        # ── Aggregate experiment ──
        seeds = [42, 123, 456, 789, 1024, 2048, 3333, 7777]
        names = ["IG", "IDGI", "Guided IG", "μ-Optimized", "Joint"]
        qs: dict[str, list[float]] = {m: [] for m in names}

        print(f"\nRunning {len(seeds)} seeds, dim={args.dim}, N={args.steps}\n")
        print(f"{'Seed':<7}" + "".join(f"{m:<14}" for m in names))
        print("─" * 77)

        for seed in seeds:
            torch.manual_seed(seed)
            model = MLP(input_dim=args.dim).to(device).eval()
            torch.manual_seed(seed + 1000)
            x = torch.randn(args.dim, device=device) * 2.0
            bl = torch.zeros(args.dim, device=device)

            rs = [
                standard_ig(model, x, bl, args.steps),
                idgi(model, x, bl, args.steps),
                guided_ig(model, x, bl, args.steps),
                mu_optimized_ig(model, x, bl, args.steps),
                joint_ig(model, x, bl, args.steps, n_alternating=3,
                         path_iter=15),
            ]
            line = f"{seed:<7}"
            for r in rs:
                qs[r.name].append(r.Q)
                line += f"{r.Q:<14.4f}"
            print(line)

        print("─" * 77)
        print(f"\n{'Stat':<12}" + "".join(f"{m:<14}" for m in names))
        print("─" * 77)
        for label, fn in [("Mean 𝒬", lambda v: sum(v) / len(v)),
                           ("Std 𝒬", lambda v: (sum((x - sum(v)/len(v))**2 for x in v)/len(v))**0.5),
                           ("Min 𝒬", min), ("Max 𝒬", max)]:
            line = f"{label:<12}"
            for m in names:
                line += f"{fn(qs[m]):<14.4f}"
            print(line)

    else:
        # ── Single experiment ──
        results = run_experiment(input_dim=args.dim, N=args.steps,
                                seed=args.seed, device=device)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.json}")