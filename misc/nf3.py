"""
Neural ODE Attribution for CNN — v4
=====================================
Complete rewrite. Two-phase training with direct velocity field.

Phase 1: Learn γ(0)=x', γ(1)=x  (boundary-only, fast)
Phase 2: Minimize path energy while keeping endpoints  (steering)

All loss terms O(1) regardless of image size.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# TIME EMBEDDING
# =============================================================================
class TimeEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t):
        if t.dim() == 0: t = t.unsqueeze(0)
        a = t.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([a.sin(), a.cos()], -1))


# =============================================================================
# VELOCITY FIELD — direct (no residual reparameterization)
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, ch, tdim):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.film = nn.Linear(tdim, 2 * ch)

    def forward(self, x, t):
        h = self.pw(self.dw(x))
        h = self.norm(h)
        s, b = self.film(t).chunk(2, dim=-1)
        h = h * (1 + s[..., None, None]) + b[..., None, None]
        return F.silu(h) + x


class VelocityField(nn.Module):
    """
    Direct v_θ(z, α) → ℝ^{C×H×W}.
    No residual trick. The network must learn the full velocity.
    Initialized so that v ≈ (x - x') via a learnable bias.
    """
    def __init__(self, in_ch=3, hid=32, tdim=64, nblocks=3):
        super().__init__()
        self.te = TimeEmbed(tdim)
        self.lift = nn.Conv2d(in_ch, hid, 3, padding=1)
        self.blocks = nn.ModuleList([ConvBlock(hid, tdim) for _ in range(nblocks)])
        self.out = nn.Conv2d(hid, in_ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # Learnable constant bias — initialized to (x-x') in init_bias()
        self.bias = nn.Parameter(torch.zeros(1))  # placeholder, replaced by init_bias

    def init_bias(self, x, x_prime):
        """Set the initial bias so v ≈ (x - x') at start."""
        self.bias = nn.Parameter((x - x_prime).detach().clone())

    def forward(self, alpha, z):
        B = z.shape[0]
        if alpha.dim() == 0: alpha = alpha.expand(B)
        t = self.te(alpha)
        h = self.lift(z)
        for blk in self.blocks:
            h = blk(h, t)
        return self.bias + self.out(h)


# =============================================================================
# ODE SOLVER
# =============================================================================
def solve_ode(v_theta, z0, N=20):
    """Euler integration, returns lists (graph-connected)."""
    dt = 1.0 / N
    traj, vels = [z0], []
    z = z0
    for i in range(N):
        a = torch.tensor(i * dt, device=z0.device, dtype=z0.dtype)
        v = v_theta(a, z)
        vels.append(v)
        z = z + dt * v
        traj.append(z)
    return traj, vels


# =============================================================================
# BATCHED MODEL GRADIENTS (detached)
# =============================================================================
def model_grads_batched(model_f, waypoints_list, target_class, chunk=8):
    """∇_z f(z) at each waypoint. Returns detached (M, B, C, H, W)."""
    wp = torch.stack(waypoints_list, 0).detach()
    M, B, C, H, W = wp.shape
    out = torch.empty_like(wp)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        flat = wp[s:e].reshape((e - s) * B, C, H, W).requires_grad_(True)
        logits = model_f(flat)
        t = logits[:, target_class]
        g = torch.autograd.grad(t.sum(), flat, create_graph=False)[0]
        out[s:e] = g.reshape(e - s, B, C, H, W)
    return out.detach()


# =============================================================================
# LOSS
# =============================================================================
@dataclass
class LossW:
    boundary: float = 200.0
    completeness: float = 5.0
    steering: float = 10.0
    alignment: float = 1.0
    reg: float = 0.5


@dataclass
class Metrics:
    total: float = 0.
    boundary: float = 0.
    completeness: float = 0.
    steering: float = 0.
    alignment: float = 0.
    reg: float = 0.
    path_int: float = 0.
    delta_f: float = 0.
    endpt_l2: float = 0.


def compute_loss(model_f, v_theta, x, x_prime, target_class, N=20,
                 w=LossW(), chunk=8, phase=2):
    """
    Phase 1: only boundary + reg (learn to connect endpoints)
    Phase 2: all terms (steer path for lower energy)

    All terms normalized to O(1).
    """
    B = x.shape[0]
    device = x.device

    traj, vels = solve_ode(v_theta, x_prime, N)

    # Δf
    with torch.no_grad():
        fx = model_f(x)[:, target_class].float()
        fxp = model_f(x_prime)[:, target_class].float()
    delta_f = (fx - fxp).detach()

    # ----- BOUNDARY: MSE per element -----
    L_bnd = ((traj[-1] - x) ** 2).mean()

    # ----- VELOCITY REGULARIZATION: MSE of velocity -----
    vel_stack = torch.stack(vels, 0)  # (N, B, C, H, W) connected
    L_reg = (vel_stack ** 2).mean()

    if phase == 1:
        loss = w.boundary * L_bnd + w.reg * L_reg
        endpt = ((traj[-1] - x) ** 2).flatten(1).sum(1).sqrt().mean().item()
        return loss, Metrics(total=loss.item(), boundary=w.boundary * L_bnd.item(),
                             reg=w.reg * L_reg.item(), delta_f=delta_f.mean().item(),
                             endpt_l2=endpt)

    # ----- Phase 2: add steering, completeness, alignment -----
    M = N
    indices = list(range(M))
    sel_traj = [traj[i + 1] for i in indices]
    sel_vels = [vels[i] for i in indices]

    mg = model_grads_batched(model_f, sel_traj, target_class, chunk=chunk)
    vs = torch.stack(sel_vels, 0)

    mg_f = mg.reshape(M, B, -1)
    vs_f = vs.reshape(M, B, -1)

    # STEERING: penalize high ‖∇f‖ at waypoints (proxy via velocity direction)
    # Idea: at waypoints where ‖∇f‖ is high, penalize velocity component
    # along ∇f direction. This makes the path avoid those regions.
    mg_n = mg_f.norm(dim=2, keepdim=True).clamp(min=1e-8)
    mg_d = mg_f / mg_n  # unit grad direction
    # velocity projection onto gradient direction
    v_proj = (vs_f * mg_d).sum(dim=2)  # (M, B)
    # weight by normalized gradient magnitude
    gw = (mg_n.squeeze(-1) / (mg_n.squeeze(-1).mean() + 1e-8)).detach()  # (M, B)
    # Penalize large projections at high-gradient waypoints
    # (we want velocity to be orthogonal to ∇f at noisy spots)
    L_steer = (gw * v_proj ** 2).mean()

    # COMPLETENESS: ∫ ∇f · v dα ≈ Δf
    dot = (mg_f * vs_f).sum(dim=2)  # (M, B)
    path_int = dot.mean(dim=0)  # (B,)
    L_compl = ((path_int - delta_f) ** 2 / (delta_f ** 2 + 1.0)).mean()

    # ALIGNMENT: cos(∇f, v)
    vs_n = vs_f.norm(dim=2, keepdim=True).clamp(min=1e-8)
    cos = (mg_f * vs_f).sum(dim=2, keepdim=True) / (mg_n * vs_n)
    L_align = (1 - cos).mean()

    loss = (w.boundary * L_bnd
            + w.completeness * L_compl
            + w.steering * L_steer
            + w.alignment * L_align
            + w.reg * L_reg)

    endpt = ((traj[-1] - x) ** 2).flatten(1).sum(1).sqrt().mean().item()

    return loss, Metrics(
        total=loss.item(),
        boundary=w.boundary * L_bnd.item(),
        completeness=w.completeness * L_compl.item(),
        steering=w.steering * L_steer.item(),
        alignment=w.alignment * L_align.item(),
        reg=w.reg * L_reg.item(),
        path_int=path_int.mean().item(),
        delta_f=delta_f.mean().item(),
        endpt_l2=endpt,
    )


# =============================================================================
# ATTRIBUTION EXTRACTION
# =============================================================================
def extract_attributions(model_f, v_theta, x, x_prime, target_class, N=50, chunk=16):
    v_theta.eval()
    device = x.device

    # GL quadrature
    try:
        import numpy as np
        nd, wt = np.polynomial.legendre.leggauss(N)
        nodes = torch.tensor((nd + 1) / 2, device=device, dtype=torch.float32)
        weights = torch.tensor(wt / 2, device=device, dtype=torch.float32)
    except ImportError:
        nodes = torch.linspace(0.5 / N, 1 - 0.5 / N, N, device=device)
        weights = torch.ones(N, device=device) / N

    wps, vs = [], []
    with torch.no_grad():
        z = x_prime.clone()
        pa = torch.tensor(0.0, device=device)
        for i in range(N):
            ta = nodes[i]
            nsub = max(1, int((ta - pa).item() * 80))
            dt = (ta - pa) / nsub
            for _ in range(nsub):
                z = z + dt * v_theta(pa, z)
                pa = pa + dt
            wps.append(z.clone())
            vs.append(v_theta(ta, z))

    grads = model_grads_batched(model_f, wps, target_class, chunk=chunk)
    vel_s = torch.stack(vs, 0)
    w = weights.view(N, 1, 1, 1, 1)
    attr = (w * grads * vel_s).sum(0)

    with torch.no_grad():
        fx = model_f(x)[:, target_class].float()
        fxp = model_f(x_prime)[:, target_class].float()
    expected = fx - fxp
    actual = attr.flatten(1).sum(1)
    ep_err = ((wps[-1] - x) ** 2).flatten(1).sum(1).sqrt()

    return attr.detach(), {
        "expected_sum": expected.cpu().numpy(),
        "actual_sum": actual.detach().cpu().numpy(),
        "completeness_error": (expected - actual.detach()).abs().cpu().numpy(),
        "endpoint_l2": ep_err.cpu().numpy(),
        "f_x": fx.cpu().numpy(),
        "f_xprime": fxp.cpu().numpy(),
    }


# =============================================================================
# EXPLAINER
# =============================================================================
class Explainer:
    def __init__(self, model_f, in_ch=3, hid=32, device="cuda"):
        self.f = model_f.to(device).eval()
        for p in self.f.parameters():
            p.requires_grad_(False)
        self.in_ch = in_ch
        self.hid = hid
        self.device = device

    def explain(self, x, x_prime, target_class,
                phase1_steps=150, phase2_steps=350,
                N=20, N_eval=50, lr=3e-3,
                w=LossW(), chunk=8, log_every=25):
        device = self.device
        x, x_prime = x.to(device), x_prime.to(device)

        v = VelocityField(self.in_ch, self.hid).to(device)
        v.init_bias(x, x_prime)

        opt = torch.optim.AdamW(v.parameters(), lr=lr, weight_decay=1e-4)
        total_steps = phase1_steps + phase2_steps
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.01)

        history = []

        # ===== PHASE 1: boundary only =====
        print("Phase 1: learning endpoints...")
        for step in range(phase1_steps):
            v.train()
            opt.zero_grad(set_to_none=True)
            loss, m = compute_loss(self.f, v, x, x_prime, target_class,
                                   N=N, w=w, chunk=chunk, phase=1)
            loss.backward()
            nn.utils.clip_grad_norm_(v.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % log_every == 0 or step == phase1_steps - 1:
                history.append({"step": step, "phase": 1, **vars(m)})
                print(f"  [{step:3d}/{phase1_steps}] "
                      f"L={m.total:.4f} bnd={m.boundary:.4f} "
                      f"reg={m.reg:.4f} endpt={m.endpt_l2:.4f}")

        # ===== PHASE 2: full optimization =====
        print("Phase 2: steering for energy reduction...")
        best = float('inf')
        patience = 0
        for step in range(phase2_steps):
            v.train()
            opt.zero_grad(set_to_none=True)
            loss, m = compute_loss(self.f, v, x, x_prime, target_class,
                                   N=N, w=w, chunk=chunk, phase=2)
            loss.backward()
            nn.utils.clip_grad_norm_(v.parameters(), 1.0)
            opt.step()
            sched.step()

            gstep = phase1_steps + step
            if m.total < best - abs(best) * 5e-4:
                best = m.total
                patience = 0
            else:
                patience += 1

            if step % log_every == 0 or step == phase2_steps - 1:
                history.append({"step": gstep, "phase": 2, **vars(m)})
                print(f"  [{step:3d}/{phase2_steps}] "
                      f"L={m.total:.4f} steer={m.steering:.4f} "
                      f"compl={m.completeness:.4f} align={m.alignment:.4f} "
                      f"bnd={m.boundary:.4f} reg={m.reg:.4f} "
                      f"| ∫={m.path_int:.3f} Δf={m.delta_f:.3f} "
                      f"endpt={m.endpt_l2:.4f}")

            if patience >= 80 and step > 50:
                print(f"  Early stop at phase2 step {step}")
                break

        attr, info = extract_attributions(self.f, v, x, x_prime, target_class,
                                          N=N_eval, chunk=chunk)
        info["history"] = history
        info["v_theta"] = v
        info["steps"] = gstep + 1
        return attr, info


# =============================================================================
# VANILLA IG
# =============================================================================
def vanilla_ig(model_f, x, x_prime, target_class, N=50, chunk=16):
    with torch.no_grad():
        d = x - x_prime
        alphas = torch.linspace(1/N, 1, N, device=x.device)
        wps = [x_prime + a * d for a in alphas]
    g = model_grads_batched(model_f, wps, target_class, chunk=chunk)
    with torch.no_grad():
        return (g.mean(0) * d)


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize(x, attr_ode, attr_ig, info, save_path=None):
    import matplotlib.pyplot as plt
    img = x[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    oh = attr_ode[0].cpu().abs().sum(0).numpy()
    ih = attr_ig[0].cpu().abs().sum(0).numpy()
    vmax = max(oh.max(), ih.max())
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(ih, cmap="hot", vmax=vmax); ax[1].set_title("Vanilla IG"); ax[1].axis("off")
    ax[2].imshow(oh, cmap="hot", vmax=vmax); ax[2].set_title("Neural ODE (ours)"); ax[2].axis("off")
    fig.suptitle(f"Completeness error: {info['completeness_error'][0]:.6f}")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_logit_profile(model_f, v_theta, x, x_prime, target_class, N=100, save_path=None):
    """Compare logit trajectories along straight line vs optimized path."""
    import matplotlib.pyplot as plt
    device = x.device
    alphas = torch.linspace(0, 1, N + 1, device=device)

    ig_logits, ode_logits = [], []
    with torch.no_grad():
        # Straight line
        for a in alphas:
            z = x_prime + a * (x - x_prime)
            ig_logits.append(model_f(z)[:, target_class].item())

        # ODE path
        v_theta.eval()
        z = x_prime.clone()
        dt = 1.0 / N
        for i in range(N + 1):
            ode_logits.append(model_f(z)[:, target_class].item())
            if i < N:
                a = torch.tensor(i * dt, device=device)
                z = z + dt * v_theta(a, z)

    a_np = alphas.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(a_np, ig_logits, 'r-', label='Straight line (IG)', alpha=0.8)
    ax.plot(a_np, ode_logits, 'b-', label='Optimized path (ours)', alpha=0.8)
    ax.set_xlabel("α (path parameter)")
    ax.set_ylabel(f"f(γ(α)) — logit for class {target_class}")
    ax.set_title("Logit Profile Along Path")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Smoothness metric
    ig_d = [abs(ig_logits[i+1] - ig_logits[i]) for i in range(N)]
    ode_d = [abs(ode_logits[i+1] - ode_logits[i]) for i in range(N)]
    ig_sm = sum(ig_d) / N
    ode_sm = sum(ode_d) / N
    reduction = (1 - ode_sm / ig_sm) * 100 if ig_sm > 0 else 0
    ax.text(0.02, 0.95, f"Smoothness — IG: {ig_sm:.6f}, Ours: {ode_sm:.6f}\n"
            f"Reduction: {reduction:.1f}%",
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# DEMO
# =============================================================================
def demo():
    import time, os
    try:
        import torchvision.models as models
        import torchvision.transforms as T
    except ImportError:
        print("pip install torchvision"); return

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    print("Loading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    MIN_CONF = 0.70
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([.485,.456,.406], [.229,.224,.225])])

    def find_conf(ds, mdl, dev, mc, mx):
        for i in range(min(len(ds), mx)):
            im, _ = ds[i]
            xc = im.unsqueeze(0).to(dev)
            with torch.no_grad():
                p = F.softmax(mdl(xc), -1)
                c, pr = p[0].max(0)
            if c.item() >= mc:
                return xc, pr.item(), c.item(), i
        return None

    loaded = False
    # ImageNet
    for root in [os.environ.get("IMAGENET_DIR",""), "/data/imagenet",
                 os.path.expanduser("~/data/imagenet")]:
        vd = os.path.join(root, "val") if root else ""
        if vd and os.path.isdir(vd):
            try:
                from torchvision.datasets import ImageFolder
                ds = ImageFolder(vd, transform=tf)
                r = find_conf(ds, model, device, MIN_CONF, 500)
                if r:
                    x, pc, cf, idx = r
                    print(f"ImageNet idx={idx}, class={pc}, conf={cf:.4f}")
                    loaded = True; break
            except: pass

    # CIFAR-10
    if not loaded:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([.485,.456,.406],[.229,.224,.225])])
            ds = CIFAR10("./data", False, download=True, transform=ctf)
            names = ["airplane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]
            r = find_conf(ds, model, device, MIN_CONF, 500)
            if r:
                x, pc, cf, idx = r
                _, lb = ds[idx]
                print(f"CIFAR-10 idx={idx} '{names[lb]}' pred={pc} conf={cf:.4f}")
                loaded = True
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
    print(f"Class: {pc}, confidence: {cf:.4f}")

    # --- Neural ODE ---
    print("\n--- Neural ODE Attribution (v4 — two-phase) ---")
    exp = Explainer(model, in_ch=3, hid=32, device=device)

    t0 = time.perf_counter()
    attr_ode, info = exp.explain(
        x, x_prime, pc,
        phase1_steps=150, phase2_steps=350,
        N=20, N_eval=50, lr=3e-3,
        w=LossW(boundary=200, completeness=5, steering=10, alignment=1, reg=0.5),
        chunk=8, log_every=25,
    )
    t_ode = time.perf_counter() - t0

    print(f"\nODE time:  {t_ode:.1f}s  |  steps: {info['steps']}")
    print(f"Compl err: {info['completeness_error'][0]:.6f}")
    print(f"Endpt L2:  {info['endpoint_l2'][0]:.6f}")

    # --- Vanilla IG ---
    print("\n--- Vanilla IG ---")
    t0 = time.perf_counter()
    attr_ig = vanilla_ig(model, x, x_prime, pc, N=50)
    t_ig = time.perf_counter() - t0

    ig_sum = attr_ig.flatten(1).sum(1).item()
    with torch.no_grad():
        exp_val = (model(x)[:,pc] - model(x_prime)[:,pc]).item()

    print(f"IG time:   {t_ig:.1f}s")
    print(f"Σ IG:      {ig_sum:.4f} (expected {exp_val:.4f})")

    print("\n" + "="*50)
    print(f"{'':15}  {'ODE':>10}  {'IG':>10}")
    print("-"*50)
    print(f"  Time (s)       {t_ode:10.1f}  {t_ig:10.1f}")
    print(f"  |attr| mean    {attr_ode.abs().mean():10.6f}  {attr_ig.abs().mean():10.6f}")
    print(f"  Σ attr         {info['actual_sum'][0]:+10.4f}  {ig_sum:+10.4f}")
    print(f"  Expected Δf    {exp_val:+10.4f}")
    print("="*50)

    try:
        visualize(x, attr_ode, attr_ig, info, "attr_v4.png")
        plot_logit_profile(model, info['v_theta'], x, x_prime, pc, save_path="logit_v4.png")
        print("Saved: attr_v4.png, logit_v4.png")
    except ImportError:
        print("pip install matplotlib for plots")

    return attr_ode, attr_ig, info


if __name__ == "__main__":
    demo()