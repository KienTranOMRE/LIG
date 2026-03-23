"""
Entropy-Aligned Path IG — v8
==============================
Unified framework connecting IG attribution to information theory.

Two decompositions along any path γ:
  - Gradient (IG):  gᵢ = ∇f(γᵢ)·Δγᵢ       sums to ≈ Δf
  - Entropy (MI):   Δhᵢ = H(γᵢ) - H(γᵢ₊₁)  sums to ΔH

where H(z) = entropy of softmax(f(z)) = prediction uncertainty.

Objective: min_γ D_KL(q ‖ p)
  where q = normalized entropy profile (ground truth importance)
        p = normalized gradient profile (IG's approximation)

When D_KL = 0, IG attributions perfectly reflect information gain.

Special cases:
  - IG: straight line, p and q may diverge (gradient spike ≠ entropy gain)
  - IDGI: reweights by |Δf| ≈ proxy for Δh, partially aligns p with q
  - Guided IG: reorders features to concentrate both p and q at path end
  - EA-IG: optimizes path to align p with q directly

Requirements: pip install torch torchvision numpy matplotlib
"""

import time
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


# =============================================================================
# 1. MODEL GRADIENT UTILITIES
# =============================================================================

def model_grad(model_f, z, target_class):
    z_leaf = z.detach().requires_grad_(True)
    logits = model_f(z_leaf)
    f_val = logits[:, target_class]
    g = torch.autograd.grad(f_val.sum(), z_leaf, create_graph=False)[0]
    return g.detach(), f_val.detach()


def model_grads_batched(model_f, points, target_class, chunk=8):
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


def softmax_entropy(model_f, z):
    """H(softmax(f(z))) — prediction entropy at point z."""
    with torch.no_grad():
        logits = model_f(z)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        H = -(probs * log_probs).sum(dim=-1)  # (B,)
    return H.item()


# =============================================================================
# 2. VANILLA IG
# =============================================================================

def compute_vanilla_ig(model_f, x, x_prime, target_class, N=50, chunk=16):
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
    device = x.device
    B = x.shape[0]
    direction = x - x_prime
    alphas = torch.linspace(0, 1.0, N + 1, device=device)

    logit_vals = []
    with torch.no_grad():
        for a in alphas:
            z = x_prime + a * direction
            logit_vals.append(model_f(z)[:, target_class])

    delta_f = logit_vals[-1] - logit_vals[0]

    weights = []
    for i in range(1, N + 1):
        weights.append((logit_vals[i] - logit_vals[i-1]).abs())
    w_stack = torch.stack(weights, 0)
    w_sum = w_stack.sum(0, keepdim=True).clamp(min=1e-8)
    w_norm = w_stack / w_sum

    step_alphas = torch.linspace(1/N, 1.0, N, device=device)
    waypoints = torch.stack([x_prime + a * direction for a in step_alphas], 0)
    grads = model_grads_batched(model_f, waypoints, target_class, chunk=chunk)

    g_flat = grads.reshape(N, B, -1)
    g_norm = g_flat.norm(dim=2, keepdim=True).clamp(min=1e-10)
    g_unit = g_flat / g_norm

    w_exp = w_norm.unsqueeze(-1)
    weighted = (w_exp * g_unit).sum(dim=0)

    d_sum = weighted.sum(dim=1, keepdim=True).clamp(min=1e-10)
    attr = (delta_f.unsqueeze(-1) * weighted / d_sum).view_as(x)

    densities = []
    for i in range(N):
        d_flat = delta_f.unsqueeze(-1) * w_exp[i] * g_unit[i] / d_sum
        densities.append(d_flat.sum(1).item())

    path = [x_prime + a * direction for a in alphas]
    return attr, densities, path


# =============================================================================
# 4. GUIDED IG (rank-based inverse weighting)
# =============================================================================

def compute_guided_ig(model_f, x, x_prime, target_class, N=50):
    device = x.device
    B = x.shape[0]

    z = x_prime.clone()
    attr = torch.zeros_like(x)
    waypoints = [z.clone()]
    densities = []

    for i in range(N):
        grad_f, _ = model_grad(model_f, z, target_class)
        remaining = x - z

        rem_flat = remaining.flatten(1)
        grad_flat = grad_f.abs().flatten(1)

        score = grad_flat * rem_flat.abs()
        ranks = score.argsort(dim=1).argsort(dim=1).float() + 1
        weights = 1.0 / ranks
        weights = weights / weights.mean(dim=1, keepdim=True)

        step_flat = (1.0 / N) * weights * rem_flat
        step = step_flat.view_as(z)

        attr += grad_f * step
        d = (grad_f * step).flatten(1).sum(1)
        densities.append(d.item())

        z = z + step
        waypoints.append(z.clone())

    return attr, waypoints, densities


# =============================================================================
# 5. ENTROPY-ALIGNED PATH IG (the unified method)
# =============================================================================

@dataclass
class EAConfig:
    n_iters: int = 10
    clamp_lo: float = 0.1
    clamp_hi: float = 10.0
    momentum: float = 0.5
    log_every: int = 1


def entropy_aligned_ig(model_f, x, x_prime, target_class, N=50,
                       cfg=EAConfig(), chunk=16):
    """
    Entropy-Aligned Path IG.

    Iteratively adjust the path velocity so that the IG gradient profile
    matches the entropy-based information gain profile.

    At each iteration:
      1. Forward passes: compute H(softmax(f(γᵢ))) at all waypoints
      2. Gradient passes: compute ∇f(γᵢ)·γ'ᵢ at all waypoints
      3. Normalize both to distributions: qᵢ (entropy), pᵢ (gradient)
      4. Scale velocity: γ'ᵢ *= qᵢ/pᵢ (align gradient profile with entropy)
      5. Momentum + endpoint correction + reintegrate

    Cost: n_iters × (N forward + N backward) ≈ 20× vanilla IG.
    """
    device = x.device
    B = x.shape[0]

    with torch.no_grad():
        f_x = model_f(x)[:, target_class].item()
        f_xp = model_f(x_prime)[:, target_class].item()
    delta_f = f_x - f_xp

    # Initialize: straight-line path
    direction = x - x_prime
    velocities = [direction.clone() for _ in range(N)]
    waypoints = [x_prime + (i / N) * direction for i in range(N + 1)]

    history = []

    for outer in range(cfg.n_iters):
        # ===== Step 1: Compute entropy profile (forward passes only) =====
        entropies = []
        for i in range(N + 1):
            H_i = softmax_entropy(model_f, waypoints[i])
            entropies.append(H_i)

        # Entropy decrease at each step: Δhᵢ = H(γᵢ) - H(γᵢ₊₁)
        # Positive Δh means entropy decreased (model became more certain)
        delta_h = [entropies[i] - entropies[i+1] for i in range(N)]
        total_delta_h = sum(delta_h)

        # ===== Step 2: Compute gradient profile =====
        grads = []
        grad_densities = []
        for i in range(N):
            g, _ = model_grad(model_f, waypoints[i], target_class)
            grads.append(g)
            d_i = (g * velocities[i]).flatten(1).sum(1).item()
            grad_densities.append(d_i)

        # ===== Step 3: Normalize to distributions =====
        # Entropy profile q: importance according to information gain
        abs_dh = [abs(dh) for dh in delta_h]
        sum_abs_dh = sum(abs_dh) + 1e-10
        q = [dh / sum_abs_dh for dh in abs_dh]

        # Gradient profile p: importance according to IG
        abs_gd = [abs(gd) for gd in grad_densities]
        sum_abs_gd = sum(abs_gd) + 1e-10
        p = [gd / sum_abs_gd for gd in abs_gd]

        # ===== Step 4: Compute D_KL(q || p) for monitoring =====
        dkl = 0.0
        for i in range(N):
            if q[i] > 1e-12 and p[i] > 1e-12:
                dkl += q[i] * math.log(q[i] / p[i])

        # ===== Step 5: Scale velocities to align p with q =====
        for i in range(N):
            if p[i] < 1e-12:
                scale = 1.0
            else:
                scale = q[i] / p[i]

            scale = max(cfg.clamp_lo, min(cfg.clamp_hi, scale))

            v_new = scale * velocities[i]
            velocities[i] = cfg.momentum * v_new + (1 - cfg.momentum) * velocities[i]

        # ===== Step 6: Renormalize total displacement =====
        # The scaling changes the total displacement. Renormalize so that
        # (1/N) Σ velocities[i] = (x - x') exactly.
        # This prevents velocity explosion while preserving the relative
        # scaling between steps (which is what carries the q/p alignment).
        total_disp = sum(v for v in velocities) / N  # actual displacement
        target_disp = direction  # x - x'
        # Per-dimension correction factor
        # Where total_disp is near zero, keep velocity as is
        ratio = target_disp.flatten(1) / (total_disp.flatten(1) + 1e-12)
        # Clamp ratio to prevent explosion on near-zero dimensions
        ratio = ratio.clamp(-10, 10)
        for i in range(N):
            v_flat = velocities[i].flatten(1)
            velocities[i] = (v_flat * ratio).view_as(x)

        # ===== Step 7: Reintegrate path =====
        waypoints = [x_prime.clone()]
        for i in range(N):
            waypoints.append(waypoints[-1] + (1.0 / N) * velocities[i])

        # ===== Logging =====
        endpoint_l2 = (waypoints[-1] - x).flatten(1).norm(1).mean().item()

        # Recompute actual densities
        actual_densities = []
        for i in range(N):
            g, _ = model_grad(model_f, waypoints[i], target_class)
            step = waypoints[i+1] - waypoints[i]
            d = (g * step).flatten(1).sum(1).item() * N
            actual_densities.append(d)

        attr_sum = sum(actual_densities) / N
        compl_err = abs(attr_sum - delta_f)

        # Recompute profiles for logging
        abs_gd_new = [abs(d) for d in actual_densities]
        sum_gd_new = sum(abs_gd_new) + 1e-10
        p_new = [d / sum_gd_new for d in abs_gd_new]

        cv = np.std(actual_densities) / (abs(np.mean(actual_densities)) + 1e-10)

        if outer % cfg.log_every == 0 or outer == cfg.n_iters - 1:
            print(f"  [{outer:2d}/{cfg.n_iters}] "
                  f"D_KL={dkl:.3f} "
                  f"compl={compl_err:.3f} "
                  f"endpt={endpoint_l2:.4f} "
                  f"CV={cv:.2f} "
                  f"ΔH={total_delta_h:.3f} "
                  f"Σattr={attr_sum:.3f}")
            history.append({
                "iter": outer, "dkl": dkl, "compl_err": compl_err,
                "endpoint_l2": endpoint_l2, "cv": cv,
                "total_delta_h": total_delta_h,
            })

    # ===== Final attributions =====
    attr = torch.zeros_like(x)
    final_densities = []
    final_entropies = []
    for i in range(N + 1):
        final_entropies.append(softmax_entropy(model_f, waypoints[i]))
    final_delta_h = [final_entropies[i] - final_entropies[i+1] for i in range(N)]

    for i in range(N):
        g, _ = model_grad(model_f, waypoints[i], target_class)
        step = waypoints[i+1] - waypoints[i]
        attr += g * step
        final_densities.append((g * step).flatten(1).sum(1).item() * N)

    return attr, waypoints, final_densities, final_delta_h, history


# =============================================================================
# 6. METRICS
# =============================================================================

def concentration_ratio(densities, delta_f, tau=0.9):
    N = len(densities)
    abs_d = sorted([abs(d) for d in densities], reverse=True)
    target = tau * abs(delta_f)
    cumsum = 0.0
    for k, d in enumerate(abs_d, 1):
        cumsum += d
        if cumsum >= target:
            return k / N
    return 1.0


def density_cv(densities):
    d = np.array(densities)
    return np.std(d) / (abs(np.mean(d)) + 1e-10)


def compute_kl(q, p):
    """D_KL(q || p) for two normalized distributions."""
    dkl = 0.0
    for qi, pi in zip(q, p):
        if qi > 1e-12 and pi > 1e-12:
            dkl += qi * math.log(qi / pi)
    return dkl


def compute_logit_profile(model_f, path, target_class):
    logits = []
    with torch.no_grad():
        for z in path:
            logits.append(model_f(z)[:, target_class].item())
    return logits


def compute_entropy_profile(model_f, path):
    entropies = []
    for z in path:
        entropies.append(softmax_entropy(model_f, z))
    return entropies


# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def plot_comparison(x, attrs, info, save_path=None):
    import matplotlib.pyplot as plt

    img = x[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    def heatmap(a):
        h = a[0].cpu().abs().sum(0).numpy()
        p99 = np.percentile(h, 99)
        return np.clip(h / max(p99, 1e-10), 0, 1)

    names = list(attrs.keys())
    n = len(names) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
    for i, name in enumerate(names):
        axes[i+1].imshow(heatmap(attrs[name]), cmap="hot", vmin=0, vmax=1)
        axes[i+1].set_title(name); axes[i+1].axis("off")

    parts = [f"{n}:{info['compl'][n]:.2f}" for n in names]
    fig.suptitle(f"Completeness — {', '.join(parts)}  (Δf={info['delta_f']:.2f})\n"
                 f"Per-image normalized (99th pctl)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_profiles(profiles, delta_f, entropy_data=None, save_path=None):
    import matplotlib.pyplot as plt

    ncols = 3 if entropy_data else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

    # Logit profile
    ax1 = axes[0]
    for name, logits, color in profiles["logits"]:
        a = [i / (len(logits)-1) for i in range(len(logits))]
        ax1.plot(a, logits, color=color, label=name, alpha=0.8)
    ax1.set_xlabel("α"); ax1.set_ylabel("f(γ(α))")
    ax1.set_title("Logit profile"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Attribution density
    ax2 = axes[1]
    for name, dens, color in profiles["densities"]:
        a = [(i+0.5)/len(dens) for i in range(len(dens))]
        ax2.plot(a, dens, color=color, label=name, alpha=0.8)
    N = len(profiles["densities"][0][1])
    ax2.axhline(delta_f/N, color="gray", ls=":", alpha=0.3, label=f"Δf/N={delta_f/N:.2f}")
    ax2.set_xlabel("α"); ax2.set_ylabel("∇f·γ' (density)")
    ax2.set_title("Gradient density (IG)"); ax2.legend(); ax2.grid(True, alpha=0.3)

    # Entropy profile (if available)
    if entropy_data and ncols == 3:
        ax3 = axes[2]
        for name, delta_h, color in entropy_data:
            a = [(i+0.5)/len(delta_h) for i in range(len(delta_h))]
            ax3.plot(a, delta_h, color=color, label=name, alpha=0.8)
        ax3.set_xlabel("α"); ax3.set_ylabel("Δhᵢ = H(γᵢ) - H(γᵢ₊₁)")
        ax3.set_title("Entropy decrease (information gain)")
        ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_alignment(grad_densities, delta_h, delta_f, save_path=None):
    """Compare normalized gradient profile p vs entropy profile q."""
    import matplotlib.pyplot as plt

    N = len(grad_densities)
    abs_gd = [abs(d) for d in grad_densities]
    sum_gd = sum(abs_gd) + 1e-10
    p = [d / sum_gd for d in abs_gd]

    abs_dh = [abs(d) for d in delta_h]
    sum_dh = sum(abs_dh) + 1e-10
    q = [d / sum_dh for d in abs_dh]

    alphas = [(i + 0.5) / N for i in range(N)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(alphas, q, color="#1D9E75", label="q (entropy)", linewidth=2, alpha=0.8)
    ax1.plot(alphas, p, color="#D85A30", label="p (gradient)", linewidth=2, alpha=0.8)
    ax1.fill_between(alphas, q, p, alpha=0.15, color="#7F77DD")
    ax1.set_xlabel("α"); ax1.set_ylabel("Normalized importance")
    dkl = compute_kl(q, p)
    ax1.set_title(f"Profile alignment: D_KL(q‖p) = {dkl:.3f}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Scatter: q vs p
    ax2.scatter(p, q, c=alphas, cmap="viridis", alpha=0.7, s=40)
    max_val = max(max(p), max(q)) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="Perfect alignment")
    ax2.set_xlabel("p (gradient importance)")
    ax2.set_ylabel("q (entropy importance)")
    ax2.set_title("Per-step alignment (color = α)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("α (path position)")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_convergence(history, save_path=None):
    import matplotlib.pyplot as plt

    iters = [h["iter"] for h in history]
    dkls = [h["dkl"] for h in history]
    compls = [h["compl_err"] for h in history]
    cvs = [h["cv"] for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(iters, dkls, 'o-', color="#1D9E75", linewidth=2)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("D_KL(q ‖ p)")
    ax1.set_title("KL divergence (lower = better alignment)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, compls, 'o-', color="#D85A30", linewidth=2)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Completeness error")
    ax2.set_title("Completeness"); ax2.grid(True, alpha=0.3)

    ax3.plot(iters, cvs, 'o-', color="#7F77DD", linewidth=2)
    ax3.set_xlabel("Iteration"); ax3.set_ylabel("CV of density")
    ax3.set_title("Density smoothness"); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 8. DEMO
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
                print(f"Found {sample_dir} ({len(jpegs)} images)")
                for fname in jpegs:
                    try: img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
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
        "Guided IG": "#378ADD", "EA-IG": "#1D9E75"
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
    ig_entropy = compute_entropy_profile(model, ig_path)
    ig_delta_h = [ig_entropy[i] - ig_entropy[i+1] for i in range(N)]
    ig_sum = attr_ig.flatten(1).sum(1).item()
    cr_ig = concentration_ratio(dens_ig, delta_f)

    # Compute initial D_KL for IG
    abs_gd = [abs(d) for d in dens_ig]; sum_gd = sum(abs_gd) + 1e-10
    p_ig = [d / sum_gd for d in abs_gd]
    abs_dh = [abs(d) for d in ig_delta_h]; sum_dh = sum(abs_dh) + 1e-10
    q_ig = [d / sum_dh for d in abs_dh]
    dkl_ig = compute_kl(q_ig, p_ig)

    print(f"  Time: {t_ig:.1f}s  Σattr: {ig_sum:.3f}  D_KL: {dkl_ig:.3f}  CR₉₀: {cr_ig:.1%}")

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
    # IDGI uses same path as IG, same entropy profile
    abs_gd = [abs(d) for d in dens_idgi]; sum_gd = sum(abs_gd) + 1e-10
    p_idgi = [d / sum_gd for d in abs_gd]
    dkl_idgi = compute_kl(q_ig, p_idgi)  # same q since same path
    print(f"  Time: {t_idgi:.1f}s  Σattr: {idgi_sum:.3f}  D_KL: {dkl_idgi:.3f}  CR₉₀: {cr_idgi:.1%}")

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
    gig_entropy = compute_entropy_profile(model, gig_path)
    gig_delta_h = [gig_entropy[i] - gig_entropy[i+1] for i in range(N)]
    gig_sum = attr_gig.flatten(1).sum(1).item()
    cr_gig = concentration_ratio(dens_gig, delta_f)
    abs_gd = [abs(d) for d in dens_gig]; sum_gd = sum(abs_gd) + 1e-10
    p_gig = [d / sum_gd for d in abs_gd]
    abs_dh = [abs(d) for d in gig_delta_h]; sum_dh = sum(abs_dh) + 1e-10
    q_gig = [d / sum_dh for d in abs_dh]
    dkl_gig = compute_kl(q_gig, p_gig)
    print(f"  Time: {t_gig:.1f}s  Σattr: {gig_sum:.3f}  D_KL: {dkl_gig:.3f}  CR₉₀: {cr_gig:.1%}")

    # ==========================================
    # 4. ENTROPY-ALIGNED IG
    # ==========================================
    print(f"\n{'='*55}")
    print("4. Entropy-Aligned IG")
    print(f"{'='*55}")
    t0 = time.perf_counter()
    attr_ea, ea_path, dens_ea, ea_delta_h, ea_history = entropy_aligned_ig(
        model, x, x_prime, pc, N=N,
        cfg=EAConfig(n_iters=10, clamp_lo=0.1, clamp_hi=10.0, momentum=0.5),
        chunk=16,
    )
    t_ea = time.perf_counter() - t0
    ea_logits = compute_logit_profile(model, ea_path, pc)
    ea_sum = attr_ea.flatten(1).sum(1).item()
    cr_ea = concentration_ratio(dens_ea, delta_f)
    abs_gd = [abs(d) for d in dens_ea]; sum_gd = sum(abs_gd) + 1e-10
    p_ea = [d / sum_gd for d in abs_gd]
    abs_dh = [abs(d) for d in ea_delta_h]; sum_dh = sum(abs_dh) + 1e-10
    q_ea = [d / sum_dh for d in abs_dh]
    dkl_ea = compute_kl(q_ea, p_ea)
    print(f"  Time: {t_ea:.1f}s  Σattr: {ea_sum:.3f}  D_KL: {dkl_ea:.3f}  CR₉₀: {cr_ea:.1%}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*85}")
    print(f"{'':22} {'IG':>8} {'IDGI':>8} {'GuidedIG':>10} {'EA-IG':>8}")
    print(f"{'-'*85}")
    print(f"  Time (s)            {t_ig:8.1f} {t_idgi:8.1f} {t_gig:10.1f} {t_ea:8.1f}")
    print(f"  Σ attr              {ig_sum:+8.3f} {idgi_sum:+8.3f} {gig_sum:+10.3f} {ea_sum:+8.3f}")
    print(f"  Compl. error        {abs(ig_sum-delta_f):8.3f} {abs(idgi_sum-delta_f):8.3f} {abs(gig_sum-delta_f):10.3f} {abs(ea_sum-delta_f):8.3f}")
    print(f"  D_KL(q‖p)          {dkl_ig:8.3f} {dkl_idgi:8.3f} {dkl_gig:10.3f} {dkl_ea:8.3f}")
    print(f"  Concentration CR₉₀  {cr_ig:7.1%} {cr_idgi:7.1%} {cr_gig:9.1%} {cr_ea:7.1%}")
    print(f"  Expected Δf         {delta_f:+8.3f}")
    print(f"{'='*85}")

    # ==========================================
    # PLOTS
    # ==========================================
    attrs = {"IG": attr_ig, "IDGI": attr_idgi, "Guided IG": attr_gig, "EA-IG": attr_ea}
    info = {
        "delta_f": delta_f,
        "compl": {
            "IG": abs(ig_sum - delta_f), "IDGI": abs(idgi_sum - delta_f),
            "Guided IG": abs(gig_sum - delta_f), "EA-IG": abs(ea_sum - delta_f),
        },
    }

    try:
        plot_comparison(x, attrs, info, "attr_v8.png")

        profiles = {
            "logits": [
                ("IG", ig_logits, colors["IG"]),
                ("IDGI", idgi_logits, colors["IDGI"]),
                ("Guided IG", gig_logits, colors["Guided IG"]),
                ("EA-IG", ea_logits, colors["EA-IG"]),
            ],
            "densities": [
                ("IG", dens_ig, colors["IG"]),
                ("IDGI", dens_idgi, colors["IDGI"]),
                ("Guided IG", dens_gig, colors["Guided IG"]),
                ("EA-IG", dens_ea, colors["EA-IG"]),
            ],
        }
        entropy_data = [
            ("IG", ig_delta_h, colors["IG"]),
            ("Guided IG", gig_delta_h, colors["Guided IG"]),
            ("EA-IG", ea_delta_h, colors["EA-IG"]),
        ]
        plot_profiles(profiles, delta_f, entropy_data, "profiles_v8.png")

        # Alignment plot for EA-IG
        plot_alignment(dens_ea, ea_delta_h, delta_f, "alignment_v8.png")

        # Convergence
        if ea_history:
            plot_convergence(ea_history, "convergence_v8.png")

        print("Saved: attr_v8.png, profiles_v8.png, alignment_v8.png, convergence_v8.png")
    except ImportError:
        print("pip install matplotlib for plots")

    return attrs, info


if __name__ == "__main__":
    demo()