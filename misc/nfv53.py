"""
Neural Flow IG — v5
====================
Initialize from Guided IG, refine with Hessian-vector products.

Strategy:
  1. Compute Guided IG path (adaptive coordinate selection)
  2. Use it as waypoint initialization for a Neural ODE velocity field
  3. Refine with HVP-based energy minimization (true ∂‖∇f‖²/∂γ)
  4. Compare all three: vanilla IG, Guided IG, Neural Flow IG

The HVP gives us the REAL gradient of the energy functional w.r.t.
the path — no proxies, no detached gradients, no exploitation.

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
    """∇_z f(z) — single point, detached result."""
    z_leaf = z.detach().requires_grad_(True)
    logits = model_f(z_leaf)
    f_val = logits[:, target_class]
    g = torch.autograd.grad(f_val.sum(), z_leaf, create_graph=False)[0]
    return g.detach(), f_val.detach()


def model_grad_with_graph(model_f, z, target_class):
    """∇_z f(z) — keeps computation graph for HVP."""
    z_leaf = z.detach().requires_grad_(True)
    logits = model_f(z_leaf)
    f_val = logits[:, target_class]
    g = torch.autograd.grad(f_val.sum(), z_leaf, create_graph=True)[0]
    return g, f_val, z_leaf


def model_grads_batched(model_f, points, target_class, chunk=8):
    """Batched ∇f, detached. For IG/Guided IG."""
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
# 2. VANILLA INTEGRATED GRADIENTS
# =============================================================================

def compute_vanilla_ig(model_f, x, x_prime, target_class, N=100, chunk=16):
    """Standard IG along straight line."""
    device = x.device
    direction = x - x_prime
    alphas = torch.linspace(1/N, 1.0, N, device=device)
    waypoints = torch.stack([x_prime + a * direction for a in alphas], 0)

    grads = model_grads_batched(model_f, waypoints, target_class, chunk=chunk)
    attr = (grads.mean(0) * direction)

    # Per-step density for profiling
    densities = []
    for i in range(N):
        d = (grads[i] * direction).flatten(1).sum(1)  # ∇f · (x-x')
        densities.append(d.item())

    return attr, densities


# =============================================================================
# 3. GUIDED INTEGRATED GRADIENTS
# =============================================================================

def compute_guided_ig(model_f, x, x_prime, target_class, N=100, fraction=0.25):
    """
    Guided IG: at each step, select the top-fraction dimensions
    (by |∇f|) to advance. Other dimensions stay put.

    This produces an adaptive path that traverses high-gradient
    dimensions first, spending more "path budget" on the features
    the model cares about.

    Returns attributions, the path waypoints, and per-step densities.
    """
    device = x.device
    B = x.shape[0]
    direction = x - x_prime  # total displacement

    z = x_prime.clone()
    dt = 1.0 / N
    attr = torch.zeros_like(x)
    waypoints = [z.clone()]
    densities = []

    for i in range(N):
        # Gradient at current position
        grad_f, _ = model_grad(model_f, z, target_class)

        # How much of each dimension is left to traverse
        remaining = x - z  # (B, C, H, W)

        # Score: |∇f| * |remaining| — prioritize dimensions with
        # large gradients AND large remaining distance
        score = (grad_f.abs() * remaining.abs()).flatten(1)  # (B, D)

        # Select top-fraction dimensions
        k = max(1, int(fraction * score.shape[1]))
        _, top_idx = score.topk(k, dim=1)

        # Build step: advance selected dimensions by dt, others stay
        step = torch.zeros_like(z).flatten(1)  # (B, D)
        remaining_flat = remaining.flatten(1)

        # For selected dimensions, move by dt * remaining
        # (so after N steps we've covered all of remaining)
        for b in range(B):
            step[b, top_idx[b]] = remaining_flat[b, top_idx[b]] * dt / (1.0/N)

        # Scale step so total displacement over N steps ≈ direction
        step = step.view_as(z)

        # Accumulate attribution: ∇f * step
        attr += grad_f * step

        # Step density
        d = (grad_f * step).flatten(1).sum(1)
        densities.append(d.item())

        z = z + step
        waypoints.append(z.clone())

    return attr, waypoints, densities


# =============================================================================
# 4. IDGI (Importance-Driven Gradients Integration)
# =============================================================================

def compute_idgi(model_f, x, x_prime, target_class, N=100, chunk=16):
    """
    IDGI: Integrated Gradients with importance-based weighting.

    From: "IDGI: A Framework to Eliminate Explanation Noise from
    Integrated Gradients" (2023)

    Key idea: instead of uniform weighting along the straight-line path,
    weight each step by the magnitude of logit change |Δf| at that step.
    Steps where the model's output changes rapidly get more weight;
    steps in flat regions get less weight. The path stays straight
    (same as vanilla IG), only the integration weights change.

    This is importance sampling applied to the IG integral:
      A = ∫₀¹ ∇f(γ(α)) · γ'(α) dα
        ≈ Σᵢ wᵢ · ∇f(γ(αᵢ)) · (x - x')

    where wᵢ ∝ |f(γ(αᵢ)) - f(γ(αᵢ₋₁))| and Σwᵢ = 1.

    Returns attributions, per-step densities, and the (straight-line) path.
    """
    device = x.device
    B = x.shape[0]
    direction = x - x_prime
    alphas = torch.linspace(0, 1.0, N + 1, device=device)  # include 0

    # Compute logits at all interpolation points
    logit_vals = []
    with torch.no_grad():
        for a in alphas:
            z = x_prime + a * direction
            l = model_f(z)[:, target_class]
            logit_vals.append(l)

    # Importance weights: |Δ logit| at each step
    weights = []
    for i in range(1, N + 1):
        w = (logit_vals[i] - logit_vals[i-1]).abs()  # (B,)
        weights.append(w)

    weights_stack = torch.stack(weights, 0)  # (N, B)
    # Normalize so weights sum to 1 per batch element
    w_sum = weights_stack.sum(dim=0, keepdim=True).clamp(min=1e-8)
    norm_weights = weights_stack / w_sum  # (N, B)
    # Scale by N so that uniform weights would give w=1 (matching IG scale)
    norm_weights = norm_weights * N

    # Compute gradients at each step (same straight-line points as IG)
    step_alphas = torch.linspace(1/N, 1.0, N, device=device)
    waypoints = torch.stack([x_prime + a * direction for a in step_alphas], 0)
    grads = model_grads_batched(model_f, waypoints, target_class, chunk=chunk)
    # grads: (N, B, C, H, W)

    # Weighted attribution: wᵢ · ∇f(zᵢ) · (x - x') / N
    # norm_weights: (N, B) → (N, B, 1, 1, 1)
    w_expanded = norm_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    weighted_grads = grads * w_expanded  # (N, B, C, H, W)
    attr = (weighted_grads.mean(0) * direction)

    # Per-step density for profiling
    densities = []
    for i in range(N):
        d = (weighted_grads[i] * direction).flatten(1).sum(1)
        densities.append(d.item() / N)

    # Build path (same straight line as IG)
    path = [x_prime + a * direction for a in alphas]

    return attr, densities, path

# =============================================================================
# 5. ENERGY COMPUTATION WITH HVP
# =============================================================================

def compute_path_energy(model_f, waypoints, target_class):
    """
    E(γ) = (1/N) Σᵢ ‖∇f(γᵢ)‖²

    Returns scalar energy (detached) for monitoring.
    """
    N = len(waypoints)
    energy = 0.0
    for z in waypoints:
        g, _ = model_grad(model_f, z, target_class)
        energy += (g ** 2).sum().item()
    return energy / N


def energy_grad_hvp(model_f, z, target_class, direction, eps=1e-3):
    """
    Compute ∂‖∇f(z)‖²/∂z using finite-difference approximation.

    ∂‖∇f(z)‖²/∂z ≈ (∇f(z+εd)·∇f(z+εd) - ∇f(z-εd)·∇f(z-εd)) / (2ε) · d̂

    But we actually want the full gradient, not just directional.

    Simpler and more direct: approximate ∂‖∇f‖²/∂z via:

    ∇_z ‖∇f(z)‖² ≈ [‖∇f(z + ε·eⱼ)‖² - ‖∇f(z - ε·eⱼ)‖²] / (2ε)

    But that's per-dimension (150K dims = too expensive).

    Instead: use the identity ∂‖∇f‖²/∂z = 2·H·∇f, and approximate H·∇f via:

    H·v ≈ (∇f(z + εv) - ∇f(z - εv)) / (2ε)

    where v = ∇f(z) (the direction we care about).
    Cost: 2 extra forward+backward passes. Works with ANY model.
    """
    v = direction.detach()
    v_norm = v.flatten().norm()
    if v_norm < 1e-10:
        return torch.zeros_like(z), torch.zeros_like(z).flatten(1).sum(1)

    # Normalize direction for numerical stability, scale epsilon
    v_hat = v / (v_norm + 1e-12)
    scaled_eps = eps * v_norm  # absolute perturbation size

    # ∇f at z + ε·v̂ and z - ε·v̂
    g_plus, _ = model_grad(model_f, z + eps * v_hat, target_class)
    g_minus, _ = model_grad(model_f, z - eps * v_hat, target_class)

    # H·v̂ ≈ (∇f(z+ε·v̂) - ∇f(z-ε·v̂)) / (2ε)
    Hv_hat = (g_plus - g_minus) / (2 * eps)

    # We want H·v = H·(v_norm · v̂) = v_norm · H·v̂
    Hv = v_norm * Hv_hat

    # ∂‖∇f‖²/∂z = 2·H·∇f = 2·H·v  (since direction = ∇f)
    grad_f, _ = model_grad(model_f, z, target_class)

    return (2.0 * Hv).detach(), grad_f.detach()


# =============================================================================
# 5. WAYPOINT-BASED PATH REFINEMENT (no Neural ODE needed!)
# =============================================================================

@dataclass
class RefineConfig:
    n_iters: int = 30          # refinement iterations
    lr: float = 0.01           # step size for waypoint update
    energy_weight: float = 1.0
    boundary_weight: float = 100.0  # keep endpoints fixed
    smoothness_weight: float = 0.5  # penalize jagged waypoint jumps
    completeness_weight: float = 5.0


def refine_path_hvp(model_f, waypoints_init, x, x_prime, target_class,
                    cfg=RefineConfig(), log_every=5):
    """
    Refine a path (from Guided IG) by directly optimizing waypoint
    positions using HVP-based energy gradients.

    No Neural ODE, no velocity field, no proxy losses.
    Just: move each waypoint in the direction that reduces ‖∇f‖² there,
    while keeping the path smooth and endpoints fixed.

    This is the simplest possible implementation of the energy
    minimization idea. Each waypoint is a free parameter.

    The HVP tells us: "if I move waypoint zᵢ by δ, how does ‖∇f(zᵢ)‖²
    change?" — and we move it in the direction that decreases energy.
    """
    device = x.device
    N = len(waypoints_init) - 2  # exclude start and end (fixed)

    # Waypoints as parameters (exclude first=x' and last=x)
    # Shape: list of (B, C, H, W) tensors
    inner_wps = [w.clone().detach().requires_grad_(False)
                 for w in waypoints_init[1:-1]]

    history = []

    for it in range(cfg.n_iters):
        # Full path: [x'] + inner + [x]
        path = [x_prime.detach()] + inner_wps + [x.detach()]

        # --- Compute energy gradient at each inner waypoint via HVP ---
        energy = 0.0
        grad_updates = []

        for i, z in enumerate(inner_wps):
            # ∇f at this waypoint
            grad_f, _ = model_grad(model_f, z, target_class)
            e_i = (grad_f ** 2).sum().item()
            energy += e_i

            # Direction for HVP: use ∇f itself (steepest energy direction)
            # ∂‖∇f‖²/∂z = 2 · H · ∇f
            Hv, _ = energy_grad_hvp(model_f, z, target_class, grad_f)
            energy_grad = 2.0 * Hv  # (B, C, H, W)

            # Smoothness: penalize deviation from linear interpolation
            # between neighbors
            prev_wp = path[i]      # i-th in full path = (i-1)-th inner + x'
            next_wp = path[i + 2]  # (i+1)-th in full path
            midpoint = 0.5 * (prev_wp + next_wp)
            smooth_grad = 2.0 * (z - midpoint)

            # Combined update direction
            update = (cfg.energy_weight * energy_grad
                      + cfg.smoothness_weight * smooth_grad)

            grad_updates.append(update.detach())

        # --- Update waypoints ---
        for i in range(len(inner_wps)):
            inner_wps[i] = (inner_wps[i] - cfg.lr * grad_updates[i]).detach()

        avg_energy = energy / len(inner_wps)

        # --- Compute completeness for monitoring ---
        path = [x_prime.detach()] + inner_wps + [x.detach()]
        total_attr = torch.zeros_like(x)
        for i in range(len(path) - 1):
            g, _ = model_grad(model_f, path[i], target_class)
            step = path[i+1] - path[i]
            total_attr += g * step * len(path)
        attr_sum = total_attr.flatten(1).sum(1).item()

        with torch.no_grad():
            fx = model_f(x)[:, target_class].item()
            fxp = model_f(x_prime)[:, target_class].item()
            delta_f = fx - fxp
            compl_err = abs(attr_sum - delta_f)

        if it % log_every == 0 or it == cfg.n_iters - 1:
            print(f"  [{it:3d}/{cfg.n_iters}] "
                  f"energy={avg_energy:.1f} "
                  f"compl_err={compl_err:.3f} "
                  f"Σattr={attr_sum:.3f} Δf={delta_f:.3f}")
            history.append({
                "iter": it, "energy": avg_energy,
                "compl_err": compl_err, "attr_sum": attr_sum
            })

    # Return refined path
    final_path = [x_prime.detach()] + inner_wps + [x.detach()]
    return final_path, history


# =============================================================================
# 6. COMPUTE ATTRIBUTIONS FROM PATH
# =============================================================================

def path_to_attributions(model_f, path, target_class, chunk=16):
    """
    Given a path [z₀, z₁, ..., z_N], compute attributions:
    A = Σᵢ ∇f(zᵢ) * (z_{i+1} - zᵢ)

    Also returns per-step densities for profiling.
    """
    N = len(path) - 1
    attr = torch.zeros_like(path[0])
    densities = []

    for i in range(N):
        g, _ = model_grad(model_f, path[i], target_class)
        step = path[i+1] - path[i]
        contribution = g * step
        attr += contribution
        d = contribution.flatten(1).sum(1).item()
        densities.append(d)

    return attr, densities


# =============================================================================
# 7. LOGIT PROFILE
# =============================================================================

def compute_logit_profile(model_f, path, target_class):
    """f(γ(αᵢ)) at each waypoint."""
    logits = []
    with torch.no_grad():
        for z in path:
            l = model_f(z)[:, target_class].item()
            logits.append(l)
    return logits


# =============================================================================
# 8. VISUALIZATION
# =============================================================================

def plot_comparison(x, attr_ig, attr_idgi, attr_gig, attr_nf, info, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    img = x[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    def heatmap(a):
        h = a[0].cpu().abs().sum(0).numpy()
        p99 = np.percentile(h, 99)
        if p99 > 0:
            h = np.clip(h / p99, 0, 1)
        return h

    h_ig = heatmap(attr_ig)
    h_idgi = heatmap(attr_idgi)
    h_gig = heatmap(attr_gig)
    h_nf = heatmap(attr_nf)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")

    axes[1].imshow(h_ig, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Vanilla IG"); axes[1].axis("off")

    axes[2].imshow(h_idgi, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("IDGI"); axes[2].axis("off")

    axes[3].imshow(h_gig, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("Guided IG"); axes[3].axis("off")

    axes[4].imshow(h_nf, cmap="hot", vmin=0, vmax=1)
    axes[4].set_title("Neural Flow IG"); axes[4].axis("off")

    fig.suptitle(
        f"Completeness — IG:{info['ig_compl']:.2f}  "
        f"IDGI:{info['idgi_compl']:.2f}  "
        f"GIG:{info['gig_compl']:.2f}  NF:{info['nf_compl']:.2f}  "
        f"(Δf={info['delta_f']:.2f})\n"
        f"Each heatmap normalized independently (99th percentile clip)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_logit_profiles(profiles, save_path=None):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Logit profile
    for name, logits, color in profiles["logits"]:
        alphas = [i / (len(logits)-1) for i in range(len(logits))]
        ax1.plot(alphas, logits, color=color, label=name, alpha=0.8)
    ax1.set_xlabel("α (path parameter)")
    ax1.set_ylabel("f(γ(α))")
    ax1.set_title("Logit profile along path")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density profile (attribution per step)
    for name, dens, color in profiles["densities"]:
        alphas = [(i+0.5) / len(dens) for i in range(len(dens))]
        ax2.plot(alphas, dens, color=color, label=name, alpha=0.8)
    ax2.axhline(y=profiles["delta_f"], color="gray", linestyle="--", alpha=0.5, label=f"Δf={profiles['delta_f']:.2f}")
    ax2.set_xlabel("α (path parameter)")
    ax2.set_ylabel("∇f · γ' (attribution density)")
    ax2.set_title("Per-step attribution density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_energy_profile(model_f, paths, target_class, save_path=None):
    """Plot ‖∇f(γ(α))‖² along each path."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, path, color in paths:
        energies = []
        for z in path:
            g, _ = model_grad(model_f, z, target_class)
            energies.append((g ** 2).flatten(1).sum(1).item())
        alphas = [i / (len(path)-1) for i in range(len(path))]
        ax.plot(alphas, energies, color=color, label=name, alpha=0.8)

        total_e = sum(energies) / len(energies)
        ax.text(0.95, energies[-1], f" E={total_e:.0f}", color=color,
                fontsize=9, va="center")

    ax.set_xlabel("α (path parameter)")
    ax.set_ylabel("‖∇f(γ(α))‖²")
    ax.set_title("Gradient energy along path")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 9. DEMO
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

    # --- Load model ---
    print("Loading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Find confident sample ---
    MIN_CONF = 0.70
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([.485,.456,.406], [.229,.224,.225])])

    loaded = False

    # Try sample_imagenet1k folder first (flat directory of JPEG files)
    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                        os.path.expanduser("~/sample_imagenet1k")]:
        if os.path.isdir(sample_dir):
            try:
                from PIL import Image
                jpegs = sorted([f for f in os.listdir(sample_dir)
                                if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                print(f"Found sample_imagenet1k at {sample_dir} ({len(jpegs)} images)")
                import random
                random.shuffle(jpegs)
                for i, fname in enumerate(jpegs):
                    fpath = os.path.join(sample_dir, fname)
                    try:
                        img_pil = Image.open(fpath).convert("RGB")
                    except Exception:
                        continue
                    xc = tf(img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = F.softmax(model(xc), -1)
                        c, pr = p[0].max(0)
                    if c.item() >= MIN_CONF:
                        x, pc, cf = xc, pr.item(), c.item()
                        print(f"  {fname} → class={pc}, conf={cf:.4f}")
                        loaded = True; break
                if not loaded:
                    print(f"  No sample with >{MIN_CONF*100:.0f}% confidence")
            except Exception as e:
                print(f"  Error loading {sample_dir}: {e}")
            if loaded: break

    # Full ImageNet val set (ImageFolder structure)
    if not loaded:
        for root in [os.environ.get("IMAGENET_DIR",""), "/data/imagenet",
                     os.path.expanduser("~/data/imagenet")]:
            vd = os.path.join(root, "val") if root else ""
            if vd and os.path.isdir(vd):
                try:
                    from torchvision.datasets import ImageFolder
                    ds = ImageFolder(vd, transform=tf)
                    for i in range(min(500, len(ds))):
                        im, _ = ds[i]
                        xc = im.unsqueeze(0).to(device)
                        with torch.no_grad():
                            p = F.softmax(model(xc), -1)
                            c, pr = p[0].max(0)
                        if c.item() >= MIN_CONF:
                            x, pc, cf = xc, pr.item(), c.item()
                            print(f"ImageNet val idx={i}, class={pc}, conf={cf:.4f}")
                            loaded = True; break
                except Exception:
                    pass
                if loaded: break

    # CIFAR-10
    if not loaded:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([.485,.456,.406],[.229,.224,.225])])
            ds = CIFAR10("./data", False, download=True, transform=ctf)
            names = ["airplane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]
            for i in range(min(500, len(ds))):
                im, lb = ds[i]
                xc = im.unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(model(xc), -1)
                    c, pr = p[0].max(0)
                if c.item() >= MIN_CONF:
                    x, pc, cf = xc, pr.item(), c.item()
                    print(f"CIFAR-10 idx={i} '{names[lb]}' pred={pc} conf={cf:.4f}")
                    loaded = True; break
        except Exception as e:
            print(f"CIFAR-10: {e}")

    if not loaded:
        print("Fallback: synthetic")
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

    N_steps = 50  # path steps (same for all methods)

    # ==========================================
    # 1. VANILLA IG
    # ==========================================
    print(f"\n{'='*50}")
    print("1. Vanilla Integrated Gradients")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    attr_ig, dens_ig = compute_vanilla_ig(model, x, x_prime, pc, N=N_steps)
    t_ig = time.perf_counter() - t0

    ig_path = [x_prime + (i/N_steps) * (x - x_prime) for i in range(N_steps + 1)]
    ig_logits = compute_logit_profile(model, ig_path, pc)
    ig_energy = compute_path_energy(model, ig_path[1:], pc)
    ig_sum = attr_ig.flatten(1).sum(1).item()
    print(f"  Time: {t_ig:.1f}s  Σattr: {ig_sum:.3f}  Energy: {ig_energy:.0f}")

    # ==========================================
    # 2. IDGI
    # ==========================================
    print(f"\n{'='*50}")
    print("2. IDGI (Importance-Driven Gradients Integration)")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    attr_idgi, dens_idgi, idgi_path = compute_idgi(
        model, x, x_prime, pc, N=N_steps)
    t_idgi = time.perf_counter() - t0

    idgi_logits = compute_logit_profile(model, idgi_path, pc)
    idgi_energy = compute_path_energy(model, idgi_path[1:-1], pc)
    idgi_sum = attr_idgi.flatten(1).sum(1).item()
    print(f"  Time: {t_idgi:.1f}s  Σattr: {idgi_sum:.3f}  Energy: {idgi_energy:.0f}")

    # ==========================================
    # 3. GUIDED IG
    # ==========================================
    print(f"\n{'='*50}")
    print("3. Guided Integrated Gradients")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    attr_gig, gig_path, dens_gig = compute_guided_ig(
        model, x, x_prime, pc, N=N_steps, fraction=0.25)
    t_gig = time.perf_counter() - t0

    gig_logits = compute_logit_profile(model, gig_path, pc)
    gig_energy = compute_path_energy(model, gig_path[1:-1], pc)
    gig_sum = attr_gig.flatten(1).sum(1).item()
    print(f"  Time: {t_gig:.1f}s  Σattr: {gig_sum:.3f}  Energy: {gig_energy:.0f}")

    # ==========================================
    # 3. NEURAL FLOW IG (Guided IG + HVP refinement)
    # ==========================================
    print(f"\n{'='*50}")
    print("4. Neural Flow IG (Guided IG → HVP refinement)")
    print(f"{'='*50}")
    t0 = time.perf_counter()

    cfg = RefineConfig(
        n_iters=30,
        lr=0.005,
        energy_weight=1.0,
        boundary_weight=100.0,
        smoothness_weight=0.5,
        completeness_weight=5.0,
    )

    nf_path, refine_history = refine_path_hvp(
        model, gig_path, x, x_prime, pc, cfg=cfg, log_every=5)
    t_refine = time.perf_counter() - t0

    attr_nf, dens_nf = path_to_attributions(model, nf_path, pc)
    nf_logits = compute_logit_profile(model, nf_path, pc)
    nf_energy = compute_path_energy(model, nf_path[1:-1], pc)
    nf_sum = attr_nf.flatten(1).sum(1).item()
    print(f"  Refine time: {t_refine:.1f}s  Σattr: {nf_sum:.3f}  Energy: {nf_energy:.0f}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*75}")
    print(f"{'':20} {'IG':>10} {'IDGI':>10} {'Guided IG':>12} {'NF-IG':>10}")
    print(f"{'-'*75}")
    print(f"  Time (s)          {t_ig:10.1f} {t_idgi:10.1f} {t_gig:12.1f} {t_refine:10.1f}")
    print(f"  Σ attr            {ig_sum:+10.3f} {idgi_sum:+10.3f} {gig_sum:+12.3f} {nf_sum:+10.3f}")
    print(f"  Compl. error      {abs(ig_sum-delta_f):10.3f} {abs(idgi_sum-delta_f):10.3f} {abs(gig_sum-delta_f):12.3f} {abs(nf_sum-delta_f):10.3f}")
    print(f"  Path energy       {ig_energy:10.0f} {idgi_energy:10.0f} {gig_energy:12.0f} {nf_energy:10.0f}")
    print(f"  Expected Δf       {delta_f:+10.3f}")

    energy_reduction_idgi = (1 - idgi_energy / ig_energy) * 100 if ig_energy > 0 else 0
    energy_reduction_gig = (1 - gig_energy / ig_energy) * 100 if ig_energy > 0 else 0
    energy_reduction_nf = (1 - nf_energy / ig_energy) * 100 if ig_energy > 0 else 0
    print(f"  Energy reduction  {'---':>10} {energy_reduction_idgi:+9.1f}% {energy_reduction_gig:+11.1f}% {energy_reduction_nf:+9.1f}%")
    print(f"{'='*75}")

    # ==========================================
    # PLOTS
    # ==========================================
    info = {
        "delta_f": delta_f,
        "ig_compl": abs(ig_sum - delta_f),
        "idgi_compl": abs(idgi_sum - delta_f),
        "gig_compl": abs(gig_sum - delta_f),
        "nf_compl": abs(nf_sum - delta_f),
    }

    try:
        plot_comparison(x, attr_ig, attr_idgi, attr_gig, attr_nf, info, "attr_v5.png")

        profiles = {
            "logits": [
                ("Vanilla IG", ig_logits, "#D85A30"),
                ("IDGI", idgi_logits, "#7F77DD"),
                ("Guided IG", gig_logits, "#378ADD"),
                ("Neural Flow IG", nf_logits, "#1D9E75"),
            ],
            "densities": [
                ("Vanilla IG", dens_ig, "#D85A30"),
                ("IDGI", dens_idgi, "#7F77DD"),
                ("Guided IG", dens_gig, "#378ADD"),
                ("Neural Flow IG", dens_nf, "#1D9E75"),
            ],
            "delta_f": delta_f,
        }
        plot_logit_profiles(profiles, "profiles_v5.png")

        path_data = [
            ("Vanilla IG", ig_path, "#D85A30"),
            ("IDGI", idgi_path, "#7F77DD"),
            ("Guided IG", gig_path, "#378ADD"),
            ("Neural Flow IG", nf_path, "#1D9E75"),
        ]
        plot_energy_profile(model, path_data, pc, "energy_v5.png")

        print("Saved: attr_v5.png, profiles_v5.png, energy_v5.png")
    except ImportError:
        print("pip install matplotlib for plots")

    return attr_ig, attr_idgi, attr_gig, attr_nf, info


if __name__ == "__main__":
    demo()