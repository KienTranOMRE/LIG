"""
utility.py - Common utilities for attribution methods
======================================================

This module contains all shared functions and data structures used by
different attribution methods (IG, IDGI, Guided IG, LIG, etc.).

Includes:
- Data classes (AttributionResult, StepInfo, InsDelScores)
- Model wrapper (ClassLogitModel)
- Gradient computation utilities
- Metric computation (Var_nu, CV2, Q)
- Insertion/Deletion evaluation
- Optimization functions (optimize_mu, optimize_mu_signal_harvesting)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StepInfo:
    t: float
    f: float
    d_k: float
    delta_f_k: float
    r_k: float
    phi_k: float
    grad_norm: float
    mu_k: float


@dataclass
class InsDelScores:
    insertion_auc: float = 0.0
    deletion_auc: float = 0.0
    insertion_curve: list[float] = field(default_factory=list)
    deletion_curve: list[float] = field(default_factory=list)
    n_steps: int = 0
    mode: str = "pixel"


@dataclass
class AttributionResult:
    name: str
    attributions: torch.Tensor
    Q: float
    CV2: float
    Var_nu: float
    steps: list[StepInfo]
    Q_history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    insdel: Optional[InsDelScores] = None
    region_insdel: Optional[InsDelScores] = None

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "Q": self.Q,
            "CV2": self.CV2,
            "Var_nu": self.Var_nu,
            "steps": [asdict(s) for s in self.steps],
            "Q_history": self.Q_history,
            "elapsed_s": self.elapsed_s,
        }
        if self.insdel is not None:
            d["insertion_auc"] = self.insdel.insertion_auc
            d["deletion_auc"] = self.insdel.deletion_auc
            d["insertion_curve"] = self.insdel.insertion_curve
            d["deletion_curve"] = self.insdel.deletion_curve
        if self.region_insdel is not None:
            d["region_insertion_auc"] = self.region_insdel.insertion_auc
            d["region_deletion_auc"] = self.region_insdel.deletion_auc
            d["region_insertion_curve"] = self.region_insdel.insertion_curve
            d["region_deletion_curve"] = self.region_insdel.deletion_curve
        return d


# ═════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPER
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
# GRADIENT UTILITIES
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
    """
    B = x_batch.shape[0]
    with torch.enable_grad():
        x_in = x_batch.detach().clone().requires_grad_(True)
        model.zero_grad()
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
# HELPER FUNCTIONS
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


def _straight_line_pass(model: nn.Module, x: torch.Tensor,
                        baseline: torch.Tensor, N: int,
                        fwd_batch_size: int = 0):
    """
    Evaluate f and ∇f at N uniformly-spaced points along the straight line.

    Returns: (delta_x, target, grads, d_list, df_list, f_vals, gnorms)
        grads   : list of N gradient tensors  (each (1, C, H, W))
        d_list  : list of N floats (d_k = ∇f·Δγ_k)
        df_list : list of N floats (Δf_k)
        f_vals  : list of N+2 floats
        gnorms  : list of N floats (‖∇f‖)
    """
    delta_x = x - baseline
    step = delta_x / N

    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    alphas = torch.arange(N, device=x.device, dtype=x.dtype).view(N, 1, 1, 1) / N
    gamma_batch = baseline + alphas * delta_x

    if fwd_batch_size <= 0 or fwd_batch_size >= N:
        f_batch, grad_batch = _forward_and_gradient_batch(model, gamma_batch)
    else:
        f_chunks, g_chunks = [], []
        for i0 in range(0, N, fwd_batch_size):
            i1 = min(i0 + fwd_batch_size, N)
            fb, gb = _forward_and_gradient_batch(model, gamma_batch[i0:i1])
            f_chunks.append(fb)
            g_chunks.append(gb)
        f_batch = torch.cat(f_chunks, dim=0)
        grad_batch = torch.cat(g_chunks, dim=0)

    f_vals_inner = f_batch.tolist()
    f_vals = [f_bl] + f_vals_inner + [f_x]

    d_tensor = (grad_batch * step).view(N, -1).sum(dim=1)
    d_list = d_tensor.tolist()

    df_list = [f_vals[k + 1] - f_vals[k] for k in range(N)]

    gnorms = grad_batch.view(N, -1).norm(dim=1).tolist()

    grads = [grad_batch[k:k+1].clone() for k in range(N)]

    return delta_x, target, grads, d_list, df_list, f_vals, gnorms


# ═════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_Var_nu(d: torch.Tensor, delta_f: torch.Tensor,
                   mu: torch.Tensor) -> float:
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))

    nu = mu * delta_f ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return 0.0
    nu = nu / nu_sum

    phi_bar = (nu * phi).sum()
    var_nu = (nu * (phi - phi_bar) ** 2).sum()

    return float(var_nu)


def compute_CV2(d: torch.Tensor, delta_f: torch.Tensor,
                mu: torch.Tensor) -> float:
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


def compute_Q(d: torch.Tensor, delta_f: torch.Tensor,
              mu: torch.Tensor) -> float:
    num = (mu * d * delta_f).sum() ** 2
    den1 = (mu * d ** 2).sum()
    den2 = (mu * delta_f ** 2).sum()
    if den1 < 1e-15 or den2 < 1e-15:
        return 0.0
    return float(num / (den1 * den2))


def compute_all_metrics(d: torch.Tensor, delta_f: torch.Tensor,
                        mu: torch.Tensor) -> tuple[float, float, float]:
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))

    nu = mu * delta_f ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return 0.0, 0.0, 1.0

    nu = nu / nu_sum

    phi_bar = (nu * phi).sum()
    var_nu = float((nu * (phi - phi_bar) ** 2).sum())

    phi_bar_val = float(phi_bar)
    if abs(phi_bar_val) < 1e-12:
        cv2 = float("inf") if var_nu > 1e-15 else 0.0
        Q = 0.0 if var_nu > 1e-15 else 1.0
    else:
        cv2 = var_nu / (phi_bar_val ** 2)
        Q = 1.0 / (1.0 + cv2)

    return var_nu, cv2, Q


# ═════════════════════════════════════════════════════════════════════════════
# INSERTION/DELETION EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_insertion_deletion(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    attributions: torch.Tensor,
    n_steps: int = 100,
    batch_size: int = 16,
) -> InsDelScores:
    device   = x.device
    _, C, H, W = x.shape
    n_pixels = H * W

    importance = attributions[0].abs().sum(dim=0).flatten()
    sorted_idx = torch.argsort(importance, descending=True)

    ranks = torch.empty(n_pixels, dtype=torch.long, device=device)
    ranks[sorted_idx] = torch.arange(n_pixels, device=device)

    counts  = torch.linspace(0, n_pixels, n_steps + 1,
                             device=device).long()
    S       = counts.shape[0]

    masks_flat = ranks.unsqueeze(0) < counts.unsqueeze(1)
    masks_4d   = masks_flat.view(S, 1, H, W).expand(S, C, H, W)

    x_exp  = x.expand(S, -1, -1, -1)
    bl_exp = baseline.expand(S, -1, -1, -1)

    x_del = torch.where(masks_4d, bl_exp, x_exp)
    x_ins = torch.where(masks_4d, x_exp,  bl_exp)

    del_logits = torch.empty(S, device=device)
    ins_logits = torch.empty(S, device=device)

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)
        del_logits[start:end] = model(x_del[start:end])
        ins_logits[start:end] = model(x_ins[start:end])

    f_x  = float(model(x))
    f_bl = float(model(baseline))

    dx            = 1.0 / n_steps
    insertion_auc = float((ins_logits[:-1] + ins_logits[1:]).sum() * dx / 2)
    deletion_auc  = float((del_logits[:-1] + del_logits[1:]).sum() * dx / 2)

    logit_range = abs(f_x - f_bl)
    if logit_range > 1e-12:
        insertion_auc_norm = (insertion_auc - f_bl) / logit_range
        deletion_auc_norm  = (deletion_auc  - f_bl) / logit_range
    else:
        insertion_auc_norm = deletion_auc_norm = 0.0

    return InsDelScores(
        insertion_auc=insertion_auc_norm,
        deletion_auc=deletion_auc_norm,
        insertion_curve=ins_logits.tolist(),
        deletion_curve=del_logits.tolist(),
        n_steps=n_steps,
    )


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION: STANDARD μ-OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu(d: torch.Tensor, delta_f: torch.Tensor,
                tau: float = 0.01, n_iter: int = 200,
                lr: float = 0.05) -> torch.Tensor:
    """
    Find μ minimising  CV²(φ) + τ·Σ μ_k log μ_k.
    """
    device = d.device
    N = d.shape[0]

    valid = delta_f.abs() > 1e-12
    safe = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe, torch.ones_like(d)).detach()
    df2 = (delta_f ** 2).detach()

    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mu = torch.softmax(logits, dim=0)

        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        w = nu / nu_sum

        mean_phi = (w * phi).sum()
        var_phi = (w * (phi - mean_phi) ** 2).sum()

        cv2 = var_phi / (mean_phi ** 2 + 1e-15)

        entropy = (mu * torch.log(mu + 1e-15)).sum()
        loss = cv2 + tau * entropy

        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION: SIGNAL-HARVESTING μ-OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def mu_star_closed_form(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mode: str = "d",
) -> torch.Tensor:
    """
    Closed-form KKT stationary measure: μ*_k ∝ |d_k| ≈ |Δf_k|
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
    """
    device = d.device
    N = d.shape[0]

    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d)).detach()
    df2 = (delta_f ** 2).detach()
    abs_d = d.abs().detach()

    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mu = torch.softmax(logits, dim=0)

        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        w = nu / nu_sum

        mean_phi = (w * phi).sum()
        var_phi = (w * (phi - mean_phi) ** 2).sum()

        signal_term = (mu * abs_d).sum()
        l2_term = (mu ** 2).sum()

        loss = var_phi - lam * signal_term + (tau / 2.0) * l2_term

        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


def compute_signal_harvesting_objective(
    d: torch.Tensor,
    delta_f: torch.Tensor,
    mu: torch.Tensor,
    lam: float = 1.0,
    tau: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Evaluate the full signal-harvesting objective:
        L(γ,μ) = Var_ν(φ)  −  λ Σ_k μ_k |d_k|  +  (τ/2) ‖μ‖²₂

    Returns:
        (total_objective, var_nu_term, signal_term, l2_term)
    """
    var_nu = compute_Var_nu(d, delta_f, mu)
    signal = float((mu * d.abs()).sum())
    l2 = float((mu ** 2).sum())
    total = var_nu - lam * signal + (tau / 2.0) * l2
    return total, var_nu, signal, l2


# ═════════════════════════════════════════════════════════════════════════════
# DEVICE UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def get_device(force: Optional[str] = None) -> torch.device:
    if force:
        dev = torch.device(force)
        print(f"[device] {dev} (forced)")
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print(f"[device] CPU")
    return dev


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (default: 42)
    """
    import random
    import numpy as np

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Ensure deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_image_batch(backbone, device, n, min_conf=0.70):
    """Load n valid images in one pass, no repeats."""
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = []

    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                       os.path.expanduser("~/sample_imagenet1k")]:
        if not os.path.isdir(sample_dir):
            continue

        # Sorted, NO shuffle → deterministic
        jpegs = sorted([f for f in os.listdir(sample_dir)
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        print(f"Found {sample_dir} ({len(jpegs)} images), need {n} valid")
        
        random.shuffle(jpegs)  # Shuffle to get variety across runs, but deterministic order within a run
        for fname in jpegs:
            if len(results) >= n:
                break
            try:
                img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
            except Exception:
                continue

            xc = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                p = F.softmax(backbone(xc), dim=-1)
                c, pr = p[0].max(0)

            if c.item() >= min_conf:
                results.append((xc, pr.item(), c.item(),
                                f"{sample_dir}/{fname}", None))
                print(f"  [{len(results)}] {fname} → class={pr.item()}, conf={c.item():.4f}")

        if len(results) >= n:
            break

    if len(results) < n:
        print(f"Warning: only found {len(results)}/{n} valid images")

    return results

def load_image(backbone: nn.Module, device: torch.device, min_conf: float = 0.70, skip: int = 0):
    """
    Load an image for attribution testing.

    Tries to load from local ImageNet samples, falls back to CIFAR-10, then synthetic.

    Args:
        backbone: Model to use for confidence checking
        device: Device to use
        min_conf: Minimum classification confidence threshold
        skip: Number of valid images to skip (for batch testing)

    Returns:
        tuple: (x, target_class, confidence, source, class_name)
            - x: Input tensor (1, 3, 224, 224)
            - target_class: Predicted class index
            - confidence: Prediction confidence
            - source: Source path/description
            - class_name: Human-readable class name (if available)
    """
    import torchvision.transforms as T

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
            jpegs = sorted([f for f in os.listdir(sample_dir)
                            if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            random.shuffle(jpegs)
            if skip == 0:  # Only print on first call
                print(f"Found {sample_dir} ({len(jpegs)} images)")
            cskip = 0
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
                    cskip += 1
                    if skip > 0 and cskip <= skip:
                        continue
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"{sample_dir}/{fname}"
                    print(f"  [{skip+1}] {fname} → class={pc}, conf={cf:.4f}")
                    break
        except Exception as e:
            if skip == 0:
                print(f"  Error: {e}")
        if x is not None:
            break

    # Fallback: CIFAR-10
    if x is None:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
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
        m = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        torch.manual_seed(42 + skip)
        raw = (torch.randn(1, 3, 224, 224, device=device) * 0.2 + 0.5).clamp(0, 1)
        x = (raw - m) / s
        with torch.no_grad():
            p = F.softmax(backbone(x), dim=-1)
            c, pr = p[0].max(0)
            pc, cf = pr.item(), c.item()
        source = "synthetic"

    # Extract human-readable class name from filename if possible
    # ImageNet format: nXXXXXXXX_class_name.JPEG
    class_name = None
    if "/" in source:
        fname = source.rsplit("/", 1)[-1]
        name_part = fname.rsplit(".", 1)[0]        # strip extension
        parts = name_part.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("n") and parts[0][1:].isdigit():
            class_name = parts[1].replace("_", " ")

    return x, pc, cf, source, class_name


# ═════════════════════════════════════════════════════════════════════════════
# PATH OPTIMIZATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_group_cache: dict = {}


def _build_spatial_groups(model, x, baseline, G=16, patch_size=14):
    """Assign pixels to G groups by gradient importance. Cached."""
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

    row_ids = torch.arange(H, device=device) // patch_size
    col_ids = torch.arange(W, device=device) // patch_size
    patch_map = (row_ids[:, None] * n_cols + col_ids[None, :]).unsqueeze(0).unsqueeze(0)

    flat_pids = patch_map.flatten()
    flat_imp = importance.flatten()
    patch_sum = torch.zeros(n_patches, device=device).scatter_add_(0, flat_pids, flat_imp)
    patch_cnt = torch.zeros(n_patches, device=device).scatter_add_(
        0, flat_pids, torch.ones_like(flat_imp)
    )
    patch_imp = patch_sum / patch_cnt.clamp(min=1)

    order = torch.argsort(patch_imp)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n_patches, device=device)
    per_grp = n_patches // G
    p2g = torch.clamp(rank // per_grp, max=G - 1)

    gmap = p2g[patch_map.flatten()].view(1, 1, H, W)
    _group_cache[key] = gmap
    return gmap


def _build_path_2d(baseline, delta_x, V, group_map, N):
    """Path from grouped velocity schedule V (G, N)."""
    v_sums = V.sum(dim=1, keepdim=True).clamp(min=1e-12)
    W_norm = V / v_sums

    gmap_flat = group_map.flatten()
    weights = W_norm.T
    pixel_weights = weights[:, gmap_flat]

    _, C, H, W = baseline.shape
    pixel_weights = pixel_weights.view(N, 1, H, W)

    steps = delta_x * pixel_weights
    cum = torch.cumsum(steps, dim=0)

    gamma_stack = torch.cat([baseline, baseline + cum], dim=0)
    return list(gamma_stack.split(1, dim=0))


def _eval_path_batched(model, gamma_pts, N, device):
    """
    Evaluate d_k, Δf_k for a path in batched calls.
    Returns: (d_vec, df_vec) both (N,) tensors
    """
    all_pts = torch.cat(gamma_pts, dim=0)

    with torch.no_grad():
        f_all = model(all_pts)

    pts_N = all_pts[:N]
    grads_N = _gradient_batch(model, pts_N)

    steps = all_pts[1:] - all_pts[:N]
    d_vec = (grads_N * steps).view(N, -1).sum(dim=1)

    f_ext = torch.cat([f_all[0:1], f_all])
    df_vec = f_ext[1:N+1] - f_ext[:N]

    return d_vec, df_vec


def _signal_harvesting_path_obj(
    d_v: torch.Tensor,
    df_v: torch.Tensor,
    mu: torch.Tensor,
    lam: float = 1.0,
) -> float:
    """
    Path sub-objective: MSE_ν(φ,1) − λ Σ_k μ_k |d_k|
    """
    valid = df_v.abs() > 1e-12
    safe_df = torch.where(valid, df_v, torch.ones_like(df_v))
    phi = torch.where(valid, d_v / safe_df, torch.ones_like(d_v))

    nu = mu * df_v ** 2
    nu_sum = nu.sum()
    if nu_sum < 1e-15:
        return 0.0
    nu = nu / nu_sum

    mse = float((nu * (phi - 1.0) ** 2).sum())
    signal = float((mu * d_v.abs()).sum())

    return mse - lam * signal


def optimize_path_signal_harvesting(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    mu: torch.Tensor,
    N: int = 50,
    G: int = 16,
    patch_size: int = 14,
    n_iter: int = 15,
    lr: float = 0.08,
    lam: float = 1.0,
):
    """
    Optimize path via grouped spatial velocity scheduling under the
    signal-harvesting objective.
    """
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    V = torch.ones(G, N, device=device)
    best_obj = float("inf")
    best_V = V.clone()

    def _obj_of(Vm):
        gp = _build_path_2d(baseline, delta_x, Vm, gmap, N)
        d_v, df_v = _eval_path_batched(model, gp, N, device)
        return _signal_harvesting_path_obj(d_v, df_v, mu, lam=lam)

    eps = 0.05
    for it in range(n_iter):
        obj = _obj_of(V)
        if obj < best_obj:
            best_obj = obj
            best_V = V.clone()

        grad_V = torch.zeros_like(V)
        for g in range(G):
            k = torch.randint(0, N, (1,)).item()
            V[g, k] += eps
            obj_plus = _obj_of(V)
            grad_V[g, k] = (obj_plus - obj) / eps
            V[g, k] -= eps

        V = V - lr * grad_V
        V = torch.clamp(V, min=0.01)

    return _build_path_2d(baseline, delta_x, best_V, gmap, N)
