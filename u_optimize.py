"""
u_optimize.py — μ-Optimized IG with Signal Harvesting
=====================================================

Implements μ-optimization for Integrated Gradients with signal-harvesting objective:

    min_{μ}  Var_ν(φ) +  (τ/2) ‖μ‖²₂

Compares three methods:
    1. Standard IG      — uniform μ, straight line
    2. IDGI             — μ_k ∝ |Δf_k|, straight line
    3. μ-Optimized      — optimized μ via signal-harvesting objective

Usage:
    from u_optimize import mu_optimized_ig, run_all_methods, run_experiment
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn

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
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  SIGNAL-HARVESTING OBJECTIVE
# ═════════════════════════════════════════════════════════════════════════════

def compute_signal_harvesting_objective(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mu: torch.Tensor,
    lam: float = 1.0,
    tau: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Evaluate the signal-harvesting objective:

        L(μ) = Var_ν(φ)  +  (τ/2) ‖μ‖²₂

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
# §2  CLOSED-FORM μ* (KKT stationary point — IDGI)
# ═════════════════════════════════════════════════════════════════════════════

def mu_star_closed_form(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mode: str = "d",
) -> torch.Tensor:
    """
    Closed-form KKT stationary measure:

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
# §3  μ-OPTIMISATION WITH SIGNAL HARVESTING
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
    Find μ minimising the signal-harvesting objective:

        min_{μ∈P_N}  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ‖μ‖²₂

    Uses Adam on softmax logits for unconstrained optimisation on the simplex.

    The L2 penalty (τ/2)‖μ‖² replaces the entropy penalty τΣμ_k log μ_k
    from the original code, because:
      - L2 yields a LINEAR stationary condition: μ*_k ∝ |d_k|/τ
      - Entropy yields an EXPONENTIAL condition: μ*_k ∝ exp(|d_k|/τ)
      - Only the linear form recovers IDGI exactly

    Limiting behaviour:
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

        # ── Full objective ──
        loss = var_phi - lam * signal_term + (tau / 2.0) * l2_term

        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §4  μ-OPTIMISED IG WITH SIGNAL HARVESTING
# ═════════════════════════════════════════════════════════════════════════════

def _pack_result(name, attr, d_list, df_list, f_vals, gnorms, mu, N,
                 t0, Q_history=None) -> AttributionResult:
    """Build AttributionResult with all metrics."""
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

    This combines the conservation-law objective (Var_ν) with signal
    harvesting (−λΣμ|d|) and L2 admissibility.

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
# §5  RUN ALL METHODS (IG, IDGI, μ-Optimized only)
# ═════════════════════════════════════════════════════════════════════════════

def run_all_methods(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
    lam: float = 1.0,
    tau: float = 0.01,
    mu_iter: int = 300,
) -> list[AttributionResult]:
    """
    Run three IG variants: IG, IDGI, and μ-Optimized*.

    Returns list: [IG, IDGI, μ-Optimized*]
    """
    from lam import standard_ig, idgi

    results = []

    # 1. Standard IG  (uniform μ, straight line)
    results.append(standard_ig(model, x, baseline, N))

    # 2. IDGI  (μ_k ∝ |Δf_k|, straight line)
    results.append(idgi(model, x, baseline, N))

    # 3. μ-Optimized*  (straight line, signal-harvesting μ)
    results.append(mu_optimized_ig(
        model, x, baseline, N, lam=lam, tau=tau, n_iter=mu_iter))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# §6  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N=50, device=None, min_conf=0.70, lam=1.0, tau=0.01, skip=0):
    """Run experiment: load model/image, run 3 methods (IG, IDGI, μ-Optimized), print table."""
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
        lam=lam, tau=tau)

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
# §7  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json
    from utilss import set_seed
    parser = argparse.ArgumentParser(
        description="μ-Optimized IG — compare IG, IDGI, and μ-Optimized")
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
        lam=args.lam, tau=args.tau, skip=args.skip)

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
