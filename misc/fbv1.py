"""
Free Boundary Path IG — v6
============================
Find the path γ from baseline x' to input x where the smallest
subset of interpolation steps accounts for the full output change Δf.

Theory: free boundary variational problem
  min |S|  s.t.  Σ_{S} d(α) = Δf,  d(α)≈0 outside S

  where d(α) = ∇f(γ(α))·γ'(α) is the attribution density.

Optimality conditions (Euler-Lagrange):
  Inside S:   γ' = c · ∇f / ‖∇f‖²   (aligned, constant density c = Δf/|S|)
  Outside S:  γ' ⊥ ∇f               (orthogonal, zero density)

Algorithm: alternating optimization with closed-form updates.
  No Hessians, no HVP, no neural networks. Just gradient projections.

Comparison: vanilla IG, IDGI, Guided IG, Free Boundary IG.
Metric: concentration ratio = smallest fraction of path carrying 90% of Δf.

Requirements: pip install torch torchvision numpy matplotlib
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# 1. MODEL GRADIENT UTILITIES
# =============================================================================

def model_grad(model_f, z, target_class):
    """∇_z f(z) — single point, detached."""
    z_leaf = z.detach().requires_grad_(True)
    logits = model_f(z_leaf)
    f_val = logits[:, target_class]
    g = torch.autograd.grad(f_val.sum(), z_leaf, create_graph=False)[0]
    return g.detach(), f_val.detach()


def model_grads_batched(model_f, points, target_class, chunk=8):
    """Batched ∇f at multiple points, detached."""
    if isinstance(points, list):
        points = torch.stack(points, 0)
    M, B, C, H, W = points.shape
    out = torch.empty_like(points)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        flat = points[s:e].reshape((e-s)*B, C, H, W).detach().requires_grad_(True)
        logits = model_f(flat)
        g = torch.autograd.grad(logits[:, target_class].sum(), flat)[0]
        out[s:e] = g.reshape(e-s, B, C, H, W)
    return out.detach()


# =============================================================================
# 2. VANILLA IG
# =============================================================================

def compute_vanilla_ig(model_f, x, x_prime, target_class, N=50, chunk=16):
    """Standard IG along straight line."""
    device = x.device
    direction = x - x_prime
    alphas = torch.linspace(1/N, 1.0, N, device=device)
    waypoints = torch.stack([x_prime + a * direction for a in alphas], 0)
    grads = model_grads_batched(model_f, waypoints, target_class, chunk=chunk)
    attr = grads.mean(0) * direction

    densities = []
    for i in range(N):
        d = (grads[i] * direction).flatten(1).sum(1)
        densities.append(d.item())

    path = [x_prime + (i/N) * direction for i in range(N + 1)]
    return attr, densities, path


# =============================================================================
# 3. IDGI
# =============================================================================

def compute_idgi(model_f, x, x_prime, target_class, N=50, chunk=16):
    """
    IDGI: normalize gradients to unit vectors, weight by |Δlogit|,
    scale by Δf. Guarantees completeness by construction.
    """
    device = x.device
    B = x.shape[0]
    direction = x - x_prime
    alphas = torch.linspace(0, 1.0, N + 1, device=device)

    # Logits at all points
    logit_vals = []
    with torch.no_grad():
        for a in alphas:
            z = x_prime + a * direction
            logit_vals.append(model_f(z)[:, target_class])

    delta_f = logit_vals[-1] - logit_vals[0]

    # Importance weights: |Δlogit|
    weights = []
    for i in range(1, N + 1):
        weights.append((logit_vals[i] - logit_vals[i-1]).abs())
    w_stack = torch.stack(weights, 0)  # (N, B)
    w_sum = w_stack.sum(0, keepdim=True).clamp(min=1e-8)
    w_norm = w_stack / w_sum  # (N, B), sums to 1

    # Gradients at interpolation points
    step_alphas = torch.linspace(1/N, 1.0, N, device=device)
    waypoints = torch.stack([x_prime + a * direction for a in step_alphas], 0)
    grads = model_grads_batched(model_f, waypoints, target_class, chunk=chunk)

    # Normalize gradients to unit direction
    g_flat = grads.reshape(N, B, -1)
    g_norm = g_flat.norm(dim=2, keepdim=True).clamp(min=1e-10)
    g_unit = g_flat / g_norm

    # Weighted unit directions
    w_exp = w_norm.unsqueeze(-1)  # (N, B, 1)
    weighted = (w_exp * g_unit).sum(dim=0)  # (B, D)

    # Scale by Δf, normalize for completeness
    d_sum = weighted.sum(dim=1, keepdim=True).clamp(min=1e-10)
    attr = (delta_f.unsqueeze(-1) * weighted / d_sum).view_as(x)

    # Densities for profiling
    densities = []
    for i in range(N):
        d_flat = delta_f.unsqueeze(-1) * w_exp[i] * g_unit[i] / d_sum
        densities.append(d_flat.sum(1).item())

    path = [x_prime + a * direction for a in alphas]
    return attr, densities, path


# =============================================================================
# 4. GUIDED IG (corrected: low-gradient dimensions first)
# =============================================================================

def compute_guided_ig(model_f, x, x_prime, target_class, N=50):
    """
    Guided IG: advance all dimensions, weighted inversely to |∇f|.
    Low-gradient dims take larger steps (cleared early).
    High-gradient dims take smaller steps (arrive late → clean signal).

    Key: clamp inverse weights to prevent explosion near baseline
    where |∇f| ≈ 0, and use remaining distance (not global direction).
    """
    device = x.device
    B = x.shape[0]

    z = x_prime.clone()
    attr = torch.zeros_like(x)
    waypoints = [z.clone()]
    densities = []

    for i in range(N):
        grad_f, _ = model_grad(model_f, z, target_class)
        remaining = x - z  # (B, C, H, W)

        # Inverse gradient weighting per dimension
        grad_abs = grad_f.abs().flatten(1)  # (B, D)

        # Clamp: don't let weight exceed 10× the median
        # This prevents explosion where |∇f| ≈ 0
        median_g = grad_abs.median(dim=1, keepdim=True).values.clamp(min=1e-8)
        inv_weight = 1.0 / (grad_abs + median_g)  # bounded: max = 1/median_g

        # Normalize so mean weight = 1 (average step = 1/N of remaining)
        inv_weight = inv_weight / inv_weight.mean(dim=1, keepdim=True).clamp(min=1e-10)

        # Step: advance each dim proportionally
        rem_flat = remaining.flatten(1)  # (B, D)
        step_flat = (1.0 / N) * inv_weight * rem_flat

        # Safety: clamp step to not exceed remaining (no overshoot)
        step_flat = step_flat.clamp(min=rem_flat.clamp(max=0), max=rem_flat.clamp(min=0))

        step = step_flat.view_as(z)

        attr += grad_f * step
        d = (grad_f * step).flatten(1).sum(1)
        densities.append(d.item())

        z = z + step
        waypoints.append(z.clone())

    return attr, waypoints, densities


# =============================================================================
# 5. FREE BOUNDARY PATH IG
# =============================================================================

@dataclass
class FBConfig:
    n_outer: int = 10         # alternating optimization iterations
    tau: float = 0.9          # fraction of |Δf| the active set must cover
    boundary_correction: bool = True  # rescale to hit endpoint
    log_every: int = 1


def free_boundary_ig(model_f, x, x_prime, target_class, N=50,
                     cfg=FBConfig(), chunk=16):
    """
    Free Boundary Path IG.

    Alternating optimization:
      1. Compute attribution density dᵢ at each step
      2. Find active set S = smallest subset with Σ_S |dᵢ| ≥ τ·|Δf|
      3. Inside S: set velocity aligned with ∇f, constant density
         Outside S: set velocity orthogonal to ∇f
      4. Reintegrate path from updated velocities
      5. Repeat until |S| stabilizes

    All updates are closed-form. No optimization loops.
    Cost: ~n_outer × N gradient evaluations.
    """
    device = x.device
    B = x.shape[0]

    with torch.no_grad():
        f_x = model_f(x)[:, target_class].item()
        f_xp = model_f(x_prime)[:, target_class].item()
    delta_f = f_x - f_xp

    # Initialize: straight-line path
    waypoints = [x_prime + (i / N) * (x - x_prime) for i in range(N + 1)]
    # velocities[i] = N * (waypoints[i+1] - waypoints[i])
    velocities = [(x - x_prime).clone() for _ in range(N)]

    history = []
    prev_S_size = N
    # Running average of densities for stable active set selection
    avg_densities = None
    momentum = 0.5  # blend factor: new = momentum * current + (1-momentum) * previous

    for outer in range(cfg.n_outer):
        # ===== Step 1: Compute densities and gradients =====
        grads = []
        densities = []
        for i in range(N):
            g, _ = model_grad(model_f, waypoints[i], target_class)
            grads.append(g)
            v_i = velocities[i]
            d_i = (g * v_i).flatten(1).sum(1).item()
            densities.append(d_i)

        # Smooth densities with running average to prevent oscillation
        if avg_densities is None:
            avg_densities = [d for d in densities]
        else:
            avg_densities = [momentum * d + (1 - momentum) * a
                             for d, a in zip(densities, avg_densities)]

        # ===== Step 2: Find active set S (from smoothed densities) =====
        abs_dens = [abs(d) for d in avg_densities]
        sorted_idx = sorted(range(N), key=lambda i: abs_dens[i], reverse=True)

        cumsum = 0.0
        target = cfg.tau * abs(delta_f)
        S = set()
        for idx in sorted_idx:
            S.add(idx)
            cumsum += abs_dens[idx]
            if cumsum >= target:
                break

        S_size = len(S)
        conc_ratio = S_size / N

        # ===== Step 3: Update velocities with momentum =====
        if S_size > 0 and abs(delta_f) > 1e-10:
            target_density = delta_f / S_size
        else:
            target_density = 0.0

        straight = (x - x_prime).flatten(1)

        for i in range(N):
            g = grads[i]
            g_flat = g.flatten(1)
            g_norm_sq = (g_flat ** 2).sum(1, keepdim=True).clamp(min=1e-12)

            if i in S:
                # Active: gradient-aligned + straight-line blend
                v_grad = (target_density * g_flat / g_norm_sq)
                v_target = 0.7 * v_grad + 0.3 * straight
            else:
                # Inactive: straight-line velocity
                v_target = straight

            # Momentum update: blend with previous velocity
            v_old = velocities[i].flatten(1)
            v_new = momentum * v_target + (1 - momentum) * v_old
            velocities[i] = v_new.view_as(x)

        # ===== Step 4: Reintegrate path =====
        waypoints = [x_prime.clone()]
        for i in range(N):
            next_wp = waypoints[-1] + (1.0 / N) * velocities[i]
            waypoints.append(next_wp)

        # ===== Step 5: Boundary correction =====
        if cfg.boundary_correction:
            endpoint_err = (waypoints[-1] - x).flatten(1).norm(1).mean().item()
            if endpoint_err > 1e-6:
                # Distribute the correction across the last few steps
                correction = x - waypoints[-1]
                n_correct = max(1, N // 5)  # spread over last 20% of steps
                for j in range(N - n_correct, N):
                    frac = (j - (N - n_correct) + 1) / n_correct
                    waypoints[j + 1] = waypoints[j + 1] + frac * correction
                # Also adjust velocities for the corrected steps
                for j in range(N - n_correct, N):
                    velocities[j] = N * (waypoints[j + 1] - waypoints[j])

        # ===== Logging =====
        endpoint_l2 = (waypoints[-1] - x).flatten(1).norm(1).mean().item()

        # Recompute densities after integration for accurate logging
        actual_densities = []
        for i in range(N):
            g, _ = model_grad(model_f, waypoints[i], target_class)
            step = waypoints[i+1] - waypoints[i]
            d = (g * step).flatten(1).sum(1).item() * N  # density = N * (∇f · step)
            actual_densities.append(d)

        actual_attr_sum = sum(actual_densities) / N
        compl_err = abs(actual_attr_sum - delta_f)

        if outer % cfg.log_every == 0 or outer == cfg.n_outer - 1:
            print(f"  [{outer:2d}/{cfg.n_outer}] "
                  f"|S|={S_size:3d}/{N} ({conc_ratio:.1%}) "
                  f"compl={compl_err:.3f} "
                  f"endpt={endpoint_l2:.4f} "
                  f"Σattr={actual_attr_sum:.3f} Δf={delta_f:.3f}")
            history.append({
                "iter": outer, "S_size": S_size, "conc_ratio": conc_ratio,
                "compl_err": compl_err, "endpoint_l2": endpoint_l2,
            })

        # Check convergence
        if S_size == prev_S_size and outer > 0:
            # S stabilized — could stop early
            pass
        prev_S_size = S_size

    # ===== Compute final attributions =====
    attr = torch.zeros_like(x)
    final_densities = []
    for i in range(N):
        g, _ = model_grad(model_f, waypoints[i], target_class)
        step = waypoints[i+1] - waypoints[i]
        contribution = g * step * N  # scale by N since step = (1/N)*v
        attr += contribution / N     # average over steps
        final_densities.append((g * step).flatten(1).sum(1).item() * N)

    return attr, waypoints, final_densities, S, history


# =============================================================================
# 6. CONCENTRATION RATIO (evaluation metric)
# =============================================================================

def concentration_ratio(densities, delta_f, tau=0.9):
    """
    What fraction of path steps carries τ of |Δf|?

    Sort |dᵢ| descending, find smallest k with Σ top-k ≥ τ·|Δf|.
    Return k/N.
    """
    N = len(densities)
    abs_d = sorted([abs(d) for d in densities], reverse=True)
    target = tau * abs(delta_f)
    cumsum = 0.0
    for k, d in enumerate(abs_d, 1):
        cumsum += d
        if cumsum >= target:
            return k / N
    return 1.0  # all steps needed


def density_entropy(densities):
    """
    Entropy of the attribution density distribution.
    Lower = more concentrated. Higher = more uniform.
    """
    import numpy as np
    abs_d = np.array([abs(d) for d in densities])
    total = abs_d.sum()
    if total < 1e-10:
        return 0.0
    p = abs_d / total
    p = p[p > 0]
    return -np.sum(p * np.log(p + 1e-15))


# =============================================================================
# 7. LOGIT PROFILE
# =============================================================================

def compute_logit_profile(model_f, path, target_class):
    logits = []
    with torch.no_grad():
        for z in path:
            logits.append(model_f(z)[:, target_class].item())
    return logits


# =============================================================================
# 8. PATH ENERGY
# =============================================================================

def compute_path_energy(model_f, waypoints, target_class):
    N = len(waypoints)
    energy = 0.0
    for z in waypoints:
        g, _ = model_grad(model_f, z, target_class)
        energy += (g ** 2).sum().item()
    return energy / N


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def plot_comparison(x, attrs, info, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    img = x[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    def heatmap(a):
        h = a[0].cpu().abs().sum(0).numpy()
        p99 = np.percentile(h, 99)
        return np.clip(h / max(p99, 1e-10), 0, 1)

    names = list(attrs.keys())
    n = len(names) + 1  # +1 for input image
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
    for i, name in enumerate(names):
        axes[i+1].imshow(heatmap(attrs[name]), cmap="hot", vmin=0, vmax=1)
        axes[i+1].set_title(name); axes[i+1].axis("off")

    title_parts = [f"{name}:{info['compl'][name]:.2f}" for name in names]
    fig.suptitle(f"Completeness — {', '.join(title_parts)}  (Δf={info['delta_f']:.2f})\n"
                 f"Per-image normalized (99th pctl)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_profiles(profiles, delta_f, save_path=None):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for name, logits, color in profiles["logits"]:
        a = [i / (len(logits)-1) for i in range(len(logits))]
        ax1.plot(a, logits, color=color, label=name, alpha=0.8)
    ax1.set_xlabel("α"); ax1.set_ylabel("f(γ(α))"); ax1.set_title("Logit profile")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for name, dens, color in profiles["densities"]:
        a = [(i+0.5)/len(dens) for i in range(len(dens))]
        ax2.plot(a, dens, color=color, label=name, alpha=0.8)
    ax2.axhline(delta_f, color="gray", ls="--", alpha=0.5, label=f"Δf={delta_f:.2f}")
    ax2.set_xlabel("α"); ax2.set_ylabel("∇f·γ' (density)"); ax2.set_title("Attribution density")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_concentration(all_densities, delta_f, save_path=None):
    """Lorenz-style concentration curve for each method."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, dens, color in all_densities:
        N = len(dens)
        abs_d = sorted([abs(d) for d in dens], reverse=True)
        cumsum = np.cumsum(abs_d) / (abs(delta_f) + 1e-10)
        fracs = np.arange(1, N+1) / N
        ax.plot(fracs, cumsum, color=color, label=name, alpha=0.8, linewidth=2)

        # Mark 90% threshold
        cr = concentration_ratio(dens, delta_f, tau=0.9)
        ax.plot(cr, 0.9, 'o', color=color, markersize=8)
        ax.annotate(f"{cr:.0%}", (cr, 0.9), textcoords="offset points",
                    xytext=(5, -15), fontsize=10, color=color)

    ax.axhline(0.9, color="gray", ls=":", alpha=0.4)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, label="Uniform (worst)")
    ax.set_xlabel("Fraction of path steps (sorted by |density|)")
    ax.set_ylabel("Cumulative fraction of |Δf|")
    ax.set_title("Attribution concentration curve (steeper = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 10. DEMO
# =============================================================================

def demo():
    import os

    try:
        import torchvision.models as models
        import torchvision.transforms as T
    except ImportError:
        print("pip install torchvision"); return

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    print("Loading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    # --- Find confident sample ---
    MIN_CONF = 0.70
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([.485,.456,.406], [.229,.224,.225])])

    loaded = False
    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                        os.path.expanduser("~/sample_imagenet1k")]:
        if os.path.isdir(sample_dir):
            try:
                from PIL import Image
                jpegs = sorted([f for f in os.listdir(sample_dir)
                                if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                import random
                random.shuffle(jpegs)
                print(f"Found {sample_dir} ({len(jpegs)} images)")
                for fname in jpegs:
                    try:
                        img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
                    except: continue
                    xc = tf(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = F.softmax(model(xc), -1)
                        c, pr = p[0].max(0)
                    if c.item() >= MIN_CONF:
                        x, pc, cf = xc, pr.item(), c.item()
                        print(f"  {fname} → class={pc}, conf={cf:.4f}")
                        loaded = True; break
            except Exception as e:
                print(f"  Error: {e}")
            if loaded: break

    if not loaded:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([.485,.456,.406],[.229,.224,.225])])
            ds = CIFAR10("./data", False, download=True, transform=ctf)
            for i in range(500):
                im, _ = ds[i]
                xc = im.unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(model(xc), -1)
                    c, pr = p[0].max(0)
                if c.item() >= MIN_CONF:
                    x, pc, cf = xc, pr.item(), c.item()
                    print(f"CIFAR-10 idx={i}, class={pc}, conf={cf:.4f}")
                    loaded = True; break
        except Exception as e:
            print(f"CIFAR-10: {e}")

    if not loaded:
        print("Synthetic fallback")
        m = torch.tensor([.485,.456,.406], device=device).view(1,3,1,1)
        s = torch.tensor([.229,.224,.225], device=device).view(1,3,1,1)
        torch.manual_seed(42)
        x = ((torch.randn(1,3,224,224,device=device)*.2+.5).clamp(0,1)-m)/s
        with torch.no_grad():
            p = F.softmax(model(x),-1); cf,pr=p[0].max(0); pc,cf=pr.item(),cf.item()

    x_prime = torch.zeros_like(x)
    with torch.no_grad():
        delta_f = (model(x)[:, pc] - model(x_prime)[:, pc]).item()
    print(f"Class: {pc}, conf: {cf:.4f}, Δf: {delta_f:.3f}")

    N = 50
    colors = {
        "IG": "#D85A30", "IDGI": "#7F77DD",
        "Guided IG": "#378ADD", "FB-IG": "#1D9E75"
    }

    # ==========================================
    # 1. VANILLA IG
    # ==========================================
    print(f"\n{'='*55}")
    print("1. Vanilla IG")
    print(f"{'='*55}")
    t0 = time.perf_counter()
    attr_ig, dens_ig, ig_path = compute_vanilla_ig(model, x, x_prime, pc, N=N)
    t_ig = time.perf_counter() - t0
    ig_logits = compute_logit_profile(model, ig_path, pc)
    ig_sum = attr_ig.flatten(1).sum(1).item()
    cr_ig = concentration_ratio(dens_ig, delta_f)
    print(f"  Time: {t_ig:.1f}s  Σattr: {ig_sum:.3f}  CR₉₀: {cr_ig:.1%}")

    # ==========================================
    # 2. IDGI
    # ==========================================
    print(f"\n{'='*55}")
    print("2. IDGI")
    print(f"{'='*55}")
    t0 = time.perf_counter()
    attr_idgi, dens_idgi, idgi_path = compute_idgi(model, x, x_prime, pc, N=N)
    t_idgi = time.perf_counter() - t0
    idgi_logits = compute_logit_profile(model, idgi_path, pc)
    idgi_sum = attr_idgi.flatten(1).sum(1).item()
    cr_idgi = concentration_ratio(dens_idgi, delta_f)
    print(f"  Time: {t_idgi:.1f}s  Σattr: {idgi_sum:.3f}  CR₉₀: {cr_idgi:.1%}")

    # ==========================================
    # 3. GUIDED IG
    # ==========================================
    print(f"\n{'='*55}")
    print("3. Guided IG")
    print(f"{'='*55}")
    t0 = time.perf_counter()
    attr_gig, gig_path, dens_gig = compute_guided_ig(model, x, x_prime, pc, N=N)
    t_gig = time.perf_counter() - t0
    gig_logits = compute_logit_profile(model, gig_path, pc)
    gig_sum = attr_gig.flatten(1).sum(1).item()
    cr_gig = concentration_ratio(dens_gig, delta_f)
    print(f"  Time: {t_gig:.1f}s  Σattr: {gig_sum:.3f}  CR₉₀: {cr_gig:.1%}")

    # ==========================================
    # 4. FREE BOUNDARY IG
    # ==========================================
    print(f"\n{'='*55}")
    print("4. Free Boundary IG")
    print(f"{'='*55}")
    t0 = time.perf_counter()
    attr_fb, fb_path, dens_fb, fb_S, fb_history = free_boundary_ig(
        model, x, x_prime, pc, N=N,
        cfg=FBConfig(n_outer=10, tau=0.9, log_every=1),
        chunk=16,
    )
    t_fb = time.perf_counter() - t0
    fb_logits = compute_logit_profile(model, fb_path, pc)
    fb_sum = attr_fb.flatten(1).sum(1).item()
    cr_fb = concentration_ratio(dens_fb, delta_f)
    print(f"  Time: {t_fb:.1f}s  Σattr: {fb_sum:.3f}  CR₉₀: {cr_fb:.1%}  |S|={len(fb_S)}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*80}")
    print(f"{'':22} {'IG':>8} {'IDGI':>8} {'GuidedIG':>10} {'FB-IG':>8}")
    print(f"{'-'*80}")
    print(f"  Time (s)            {t_ig:8.1f} {t_idgi:8.1f} {t_gig:10.1f} {t_fb:8.1f}")
    print(f"  Σ attr              {ig_sum:+8.3f} {idgi_sum:+8.3f} {gig_sum:+10.3f} {fb_sum:+8.3f}")
    print(f"  Compl. error        {abs(ig_sum-delta_f):8.3f} {abs(idgi_sum-delta_f):8.3f} {abs(gig_sum-delta_f):10.3f} {abs(fb_sum-delta_f):8.3f}")
    print(f"  Concentration CR₉₀  {cr_ig:7.1%} {cr_idgi:7.1%} {cr_gig:9.1%} {cr_fb:7.1%}")
    print(f"  Expected Δf         {delta_f:+8.3f}")
    print(f"{'='*80}")

    # ==========================================
    # PLOTS
    # ==========================================
    attrs = {"IG": attr_ig, "IDGI": attr_idgi, "Guided IG": attr_gig, "FB-IG": attr_fb}
    info = {
        "delta_f": delta_f,
        "compl": {
            "IG": abs(ig_sum - delta_f), "IDGI": abs(idgi_sum - delta_f),
            "Guided IG": abs(gig_sum - delta_f), "FB-IG": abs(fb_sum - delta_f),
        },
    }

    try:
        plot_comparison(x, attrs, info, "attr_v6.png")

        profiles = {
            "logits": [
                ("IG", ig_logits, colors["IG"]),
                ("IDGI", idgi_logits, colors["IDGI"]),
                ("Guided IG", gig_logits, colors["Guided IG"]),
                ("FB-IG", fb_logits, colors["FB-IG"]),
            ],
            "densities": [
                ("IG", dens_ig, colors["IG"]),
                ("IDGI", dens_idgi, colors["IDGI"]),
                ("Guided IG", dens_gig, colors["Guided IG"]),
                ("FB-IG", dens_fb, colors["FB-IG"]),
            ],
        }
        plot_profiles(profiles, delta_f, "profiles_v6.png")

        conc_data = [
            ("IG", dens_ig, colors["IG"]),
            ("IDGI", dens_idgi, colors["IDGI"]),
            ("Guided IG", dens_gig, colors["Guided IG"]),
            ("FB-IG", dens_fb, colors["FB-IG"]),
        ]
        plot_concentration(conc_data, delta_f, "concentration_v6.png")

        print("Saved: attr_v6.png, profiles_v6.png, concentration_v6.png")
    except ImportError:
        print("pip install matplotlib for plots")

    return attrs, info


if __name__ == "__main__":
    demo()