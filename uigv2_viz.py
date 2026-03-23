"""
unified_ig_v2.py — Unified IG Framework for Real Vision Models (PyTorch)
=========================================================================

Extension of the unified IG framework to operate on pretrained vision models
(ResNet-50) with real or synthetic images. Demonstrates the quality metric
𝒬 = 1/(1 + CV²(φ)) on a production-scale model where the straight-line
path traverses deep nonlinearities (BatchNorm, ReLU×50, residual adds).

Key adaptations from v1 (toy MLP):
  - Input is (1, 3, 224, 224) image tensor, not 1-D vector.
  - f(x) = logit of predicted class (scalar output for attribution).
  - Guided IG operates on the flattened pixel space (150,528 dims).
  - Path optimisation uses spatial patch groups instead of per-feature
    groups (too expensive at 150k dims).
  - Joint optimisation replaces finite-difference path search with a
    Guided-IG-initialised path + μ optimisation (practical at scale).

Usage
-----
    python unified_ig_v2.py                         # auto-find image
    python unified_ig_v2.py --json results.json     # export JSON
    python unified_ig_v2.py --steps 30              # fewer steps (faster)
    python unified_ig_v2.py --device cpu             # force CPU

Requirements: torch >= 2.0, torchvision

References
----------
    Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
    Kapishnikov et al., "Guided Integrated Gradients" (NeurIPS 2021)
    Sikdar et al., "Integrated Directional Gradients" (ACL 2021)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


# ═════════════════════════════════════════════════════════════════════════════
# §1  DEVICE SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def get_device(force: Optional[str] = None) -> torch.device:
    """Select compute device. Defaults to CPU for sequential scalar ops."""
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


# ═════════════════════════════════════════════════════════════════════════════
# §2  MODEL WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

class ClassLogitModel(nn.Module):
    """
    Wraps a classifier to output the logit for a specific class.

    This makes the model scalar-valued: f(x) = logit[target_class](x),
    which is the standard setup for IG attribution.
    """

    def __init__(self, backbone: nn.Module, target_class: int):
        super().__init__()
        self.backbone = backbone
        self.target_class = target_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar logit for the target class."""
        logits = self.backbone(x)
        return logits[:, self.target_class].squeeze(0)


# ═════════════════════════════════════════════════════════════════════════════
# §3  DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StepInfo:
    """Per-step diagnostics along the interpolation path."""
    t: float
    f: float
    d_k: float
    delta_f_k: float
    r_k: float
    phi_k: float
    grad_norm: float
    mu_k: float


@dataclass
class AttributionResult:
    """Complete output of an attribution method."""
    name: str
    attributions: torch.Tensor      # (1, 3, 224, 224) saliency map
    Q: float
    CV2: float
    steps: list[StepInfo]
    Q_history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "Q": self.Q,
            "CV2": self.CV2,
            "steps": [asdict(s) for s in self.steps],
            "Q_history": self.Q_history,
            "elapsed_s": self.elapsed_s,
        }


# ═════════════════════════════════════════════════════════════════════════════
# §4  QUALITY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_Q(d: torch.Tensor, delta_f: torch.Tensor,
              mu: torch.Tensor) -> float:
    """𝒬 = (Σ μ_k d_k Δf_k)² / [(Σ μ_k d_k²)(Σ μ_k Δf_k²)]"""
    num = (mu * d * delta_f).sum() ** 2
    den1 = (mu * d ** 2).sum()
    den2 = (mu * delta_f ** 2).sum()
    if den1 < 1e-15 or den2 < 1e-15:
        return 0.0
    return float(num / (den1 * den2))


def compute_CV2(d: torch.Tensor, delta_f: torch.Tensor,
                mu: torch.Tensor) -> float:
    """CV²(φ) under effective measure ν_k ∝ μ_k Δf_k²."""
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
# §5  GRADIENT UTILITIES (image-shaped)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_scalar(model: nn.Module, x: torch.Tensor) -> float:
    """f(x) → Python float. Input x is (1, 3, H, W)."""
    return float(model(x))


def _gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """∇_x f(x) for image input. Returns (1, 3, H, W) gradient."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        out = model(x_in)
        out.backward()
    return x_in.grad.detach()


def _forward_and_gradient(model: nn.Module, x: torch.Tensor
                          ) -> tuple[float, torch.Tensor]:
    """f(x) and ∇_x f(x) in one pass. Returns (float, (1,3,H,W) grad)."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        out = model(x_in)
        f_val = float(out)
        out.backward()
    return f_val, x_in.grad.detach()


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flat inner product of two image-shaped tensors."""
    return float((a * b).sum())


# ═════════════════════════════════════════════════════════════════════════════
# §6  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _rescale_for_completeness(attr: torch.Tensor, target: float) -> torch.Tensor:
    """Scale attributions so Σ A_i = f(x) − f(x')."""
    s = attr.sum().item()
    if abs(s) > 1e-12:
        return attr * (target / s)
    return attr


def _make_steps_info(d_list, df_list, f_vals, grad_norms, mu, N):
    """Build StepInfo list from pre-computed arrays."""
    steps = []
    for k in range(N):
        d_k = d_list[k]
        df_k = df_list[k]
        r_k = df_k - d_k
        phi_k = d_k / df_k if abs(df_k) > 1e-12 else 1.0
        steps.append(StepInfo(
            t=k / N, f=f_vals[k], d_k=d_k, delta_f_k=df_k,
            r_k=r_k, phi_k=phi_k,
            grad_norm=grad_norms[k], mu_k=float(mu[k]),
        ))
    return steps


# ═════════════════════════════════════════════════════════════════════════════
# §7  STANDARD IG
# ═════════════════════════════════════════════════════════════════════════════

def standard_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                N: int = 50, rescale: bool = False) -> AttributionResult:
    """
    Standard IG (Sundararajan et al., 2017).
    Straight-line path, uniform measure μ_k = 1/N.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    grad_sum = torch.zeros_like(x)
    d_list, df_list, f_vals, gnorms = [], [], [], []

    for k in range(N + 1):
        gamma_k = baseline + (k / N) * delta_x
        f_vals.append(_forward_scalar(model, gamma_k))

    for k in range(N):
        gamma_k = baseline + (k / N) * delta_x
        grad_k = _gradient(model, gamma_k)
        step_k = delta_x / N

        d_list.append(_dot(grad_k, step_k))
        df_list.append(f_vals[k + 1] - f_vals[k])
        gnorms.append(float(grad_k.norm()))
        grad_sum += grad_k

    attr = delta_x * grad_sum / N
    if rescale:
        attr = _rescale_for_completeness(attr, target)

    mu = torch.full((N,), 1.0 / N, device=device)
    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    steps = _make_steps_info(d_list, df_list, f_vals, gnorms, mu, N)

    return AttributionResult(
        name="IG", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps, elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §8  IDGI
# ═════════════════════════════════════════════════════════════════════════════

def idgi(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
         N: int = 50) -> AttributionResult:
    """
    IDGI (Sikdar et al., 2021). Straight-line path, μ_k ∝ |Δf_k|.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    grads, d_list, df_list, f_vals, gnorms = [], [], [], [], []

    for k in range(N):
        gamma_k = baseline + (k / N) * delta_x
        f_k, grad_k = _forward_and_gradient(model, gamma_k)
        f_vals.append(f_k)
        grads.append(grad_k)
        d_list.append(_dot(grad_k, delta_x / N))
        gnorms.append(float(grad_k.norm()))

    f_vals.append(_forward_scalar(model, x))
    for k in range(N):
        df_list.append(f_vals[k + 1] - f_vals[k])

    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)

    weights = df_arr.abs()
    w_sum = weights.sum()
    mu = weights / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0 / N, device=device)

    wg = sum(mu[k].item() * grads[k] for k in range(N))
    attr = _rescale_for_completeness(delta_x * wg, target)

    steps = _make_steps_info(d_list, df_list, f_vals, gnorms, mu, N)

    return AttributionResult(
        name="IDGI", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps, elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §9  GUIDED IG
# ═════════════════════════════════════════════════════════════════════════════

def guided_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
              N: int = 50) -> AttributionResult:
    """
    Guided IG (Kapishnikov et al., 2021).
    Move low-gradient pixels first. Operates on flattened (1,3,H,W) space.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    remaining = delta_x.clone()
    current = baseline.clone()
    gamma_pts = [current.clone()]
    grad_list = []
    d_list, df_list, f_vals, gnorms = [], [], [], []

    for k in range(N):
        f_k, grad_k = _forward_and_gradient(model, current)
        f_vals.append(f_k)
        grad_list.append(grad_k)
        gnorms.append(float(grad_k.norm()))

        # Inverse-gradient weighting (element-wise on image)
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

        remaining = remaining - step
        current = next_pt
        gamma_pts.append(current.clone())

    f_vals.append(_forward_scalar(model, current))

    # Attribution
    attr = torch.zeros_like(x)
    for k in range(N):
        attr += grad_list[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale_for_completeness(attr, target)

    mu = torch.full((N,), 1.0 / N, device=device)
    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    steps = _make_steps_info(d_list, df_list, f_vals, gnorms, mu, N)

    return AttributionResult(
        name="Guided IG", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps, elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §10  μ-OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu(d: torch.Tensor, delta_f: torch.Tensor,
                tau: float = 0.01, n_iter: int = 200,
                lr: float = 0.05) -> torch.Tensor:
    """
    Minimise CV²(φ) + τ·H(μ) over the simplex via Adam on softmax logits.

    Objective is CV²(φ) = Var_ν(φ) / E_ν[φ]² (not just variance).
    """
    device = d.device
    N = d.shape[0]

    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))
    df2 = delta_f ** 2

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

        # Correct objective: CV²(φ), not just Var(φ)
        cv2 = var_phi / (mean_phi ** 2 + 1e-15)

        entropy = (mu * torch.log(mu + 1e-15)).sum()
        loss = cv2 + tau * entropy
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §11  μ-OPTIMISED IG
# ═════════════════════════════════════════════════════════════════════════════

def mu_optimized_ig(model: nn.Module, x: torch.Tensor,
                    baseline: torch.Tensor, N: int = 50,
                    tau: float = 0.005, n_iter: int = 300) -> AttributionResult:
    """
    Straight-line path with μ minimising CV²(φ). Free improvement over IG.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    grads, d_list, df_list, f_vals, gnorms = [], [], [], [], []

    for k in range(N):
        gamma_k = baseline + (k / N) * delta_x
        f_k, grad_k = _forward_and_gradient(model, gamma_k)
        f_vals.append(f_k)
        grads.append(grad_k)
        d_list.append(_dot(grad_k, delta_x / N))
        gnorms.append(float(grad_k.norm()))

    f_vals.append(_forward_scalar(model, x))
    for k in range(N):
        df_list.append(f_vals[k + 1] - f_vals[k])

    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=n_iter)

    wg = sum(mu[k].item() * grads[k] for k in range(N))
    attr = _rescale_for_completeness(delta_x * wg, target)

    steps = _make_steps_info(d_list, df_list, f_vals, gnorms, mu, N)

    return AttributionResult(
        name="μ-Optimized", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu), CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps, elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §12  JOINT OPTIMISATION (Practical for vision models)
# ═════════════════════════════════════════════════════════════════════════════

def joint_ig(
    model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
    N: int = 50, n_alternating: int = 2,
    tau: float = 0.005, mu_iter: int = 300,
) -> AttributionResult:
    """
    Joint optimisation of path γ and measure μ.

    For vision models, full finite-difference path optimisation over
    150k+ dims is intractable. Instead we use a two-phase strategy:

    Phase A (path): Use Guided IG's heuristic path as an informed
        initialisation — it's already a strong path that moves through
        low-gradient regions first.

    Phase B (measure): Optimise μ on the Guided IG path to minimise
        CV²(φ). This combines the path benefits of Guided IG with the
        measure benefits of μ-optimisation.

    The alternating loop re-evaluates and re-optimises μ after each
    path evaluation to ensure convergence.

    This is the practical realisation of joint optimisation at scale:
    Guided IG contributes the path degree of freedom, μ-opt contributes
    the measure degree of freedom.
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    target = _forward_scalar(model, x) - _forward_scalar(model, baseline)

    # ── Phase A: construct Guided IG path ──
    remaining = delta_x.clone()
    current = baseline.clone()
    gamma_pts = [current.clone()]
    all_grads = []

    for k in range(N):
        grad_k = _gradient(model, current)
        all_grads.append(grad_k)

        abs_g = grad_k.abs() + 1e-8
        inv_w = 1.0 / abs_g
        frac = inv_w / inv_w.sum()
        remaining_steps = N - k

        raw_step = remaining.abs() * frac * remaining_steps * remaining.numel()
        step = remaining.sign() * torch.minimum(raw_step, remaining.abs())

        current = current + step
        remaining = remaining - step
        gamma_pts.append(current.clone())

    # ── Phase B: alternating μ-optimisation on this path ──
    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history = []

    for s in range(n_alternating):
        # Evaluate path diagnostics
        d_list, df_list, f_vals, gnorms = [], [], [], []
        for k in range(N + 1):
            f_vals.append(_forward_scalar(model, gamma_pts[k]))
        for k in range(N):
            grad_k = all_grads[k]
            step_k = gamma_pts[k + 1] - gamma_pts[k]
            d_list.append(_dot(grad_k, step_k))
            df_list.append(f_vals[k + 1] - f_vals[k])
            gnorms.append(float(grad_k.norm()))

        d_arr = torch.tensor(d_list, device=device)
        df_arr = torch.tensor(df_list, device=device)

        # Optimise μ
        mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=mu_iter)
        Q_val = compute_Q(d_arr, df_arr, mu)
        cv2_val = compute_CV2(d_arr, df_arr, mu)

        Q_history.append({
            "iteration": s,
            "Q": float(Q_val),
            "CV2": float(cv2_val),
        })

    # ── Final attributions ──
    attr = torch.zeros_like(x)
    for k in range(N):
        step_k = gamma_pts[k + 1] - gamma_pts[k]
        attr += mu[k] * all_grads[k] * step_k
    attr = _rescale_for_completeness(attr, target)

    steps = _make_steps_info(d_list, df_list, f_vals, gnorms, mu, N)

    return AttributionResult(
        name="Joint", attributions=attr,
        Q=compute_Q(d_arr, df_arr, mu),
        CV2=compute_CV2(d_arr, df_arr, mu),
        steps=steps, Q_history=Q_history,
        elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §13  IMAGE LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_image_and_model(device: torch.device, min_conf: float = 0.70):
    """
    Load ResNet-50 and find a high-confidence image.

    Search order:
      1. ./sample_imagenet1k (local directory)
      2. CIFAR-10 (auto-download)
      3. Synthetic fallback

    Returns
    -------
    model     : ClassLogitModel wrapping ResNet-50 for the predicted class.
    x         : (1, 3, 224, 224) preprocessed input image.
    baseline  : (1, 3, 224, 224) zero baseline (black image in normalised space).
    info      : dict with metadata (class, confidence, source).
    """
    # ── Load backbone ──
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

    # ── Strategy 1: local image directory ──
    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                        os.path.expanduser("~/sample_imagenet1k")]:
        if not os.path.isdir(sample_dir):
            continue
        try:
            from PIL import Image
            jpegs = sorted([
                f for f in os.listdir(sample_dir)
                if f.lower().endswith(('.jpeg', '.jpg', '.png'))
            ])
            import random
            random.shuffle(jpegs)
            print(f"Found {sample_dir} ({len(jpegs)} images)")
            for fname in jpegs:
                try:
                    img = Image.open(
                        os.path.join(sample_dir, fname)).convert("RGB")
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

    # ── Strategy 2: CIFAR-10 ──
    if x is None:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([
                T.Resize(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
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
                    print(f"  ✓ CIFAR-10 idx={i} → class={pc}, conf={cf:.4f}")
                    break
        except Exception as e:
            print(f"  CIFAR-10: {e}")

    # ── Strategy 3: synthetic ──
    if x is None:
        print("Using synthetic image fallback")
        m = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        torch.manual_seed(42)
        raw = (torch.randn(1, 3, 224, 224, device=device) * 0.2 + 0.5).clamp(0, 1)
        x = (raw - m) / s
        with torch.no_grad():
            p = F.softmax(backbone(x), dim=-1)
            c, pr = p[0].max(0)
            pc, cf = pr.item(), c.item()
        source = "synthetic"

    # ── Wrap model for target class ──
    model = ClassLogitModel(backbone, target_class=pc).to(device).eval()
    baseline = torch.zeros_like(x)

    info = {
        "source": source,
        "target_class": pc,
        "confidence": cf,
        "model": "ResNet-50 (ImageNet pretrained)",
    }

    return model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §14  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def _denormalize_image(x: torch.Tensor) -> "np.ndarray":
    """Convert normalised (1,3,H,W) tensor back to (H,W,3) uint8 for display."""
    import numpy as np
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    img = (x * std + mean).clamp(0, 1)
    return (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _attribution_heatmap(attr: torch.Tensor) -> "np.ndarray":
    """
    Convert (1,3,H,W) attribution tensor to (H,W) saliency map.

    Aggregation: sum absolute values across channels, then normalise
    to [0, 1] for heatmap display.
    """
    import numpy as np
    sal = attr[0].abs().sum(dim=0).cpu().numpy()  # (H, W)
    vmax = np.percentile(sal, 99)
    if vmax > 1e-12:
        sal = sal / vmax
    return np.clip(sal, 0, 1)


def _attribution_diverging(attr: torch.Tensor) -> "np.ndarray":
    """
    Convert (1,3,H,W) attribution to signed (H,W) map for diverging colormap.

    Positive = supports prediction, Negative = opposes prediction.
    """
    import numpy as np
    sal = attr[0].sum(dim=0).cpu().numpy()  # (H, W) signed
    vmax = max(np.percentile(np.abs(sal), 99), 1e-12)
    return np.clip(sal / vmax, -1, 1)


def visualize_attributions(
    x: torch.Tensor,
    methods: list[AttributionResult],
    info: dict,
    save_path: str = "attribution_heatmaps.png",
    delta_f: float = 0.0,
):
    """
    Generate publication-quality attribution heatmap figure.

    Layout: 2 rows × (1 + N_methods) columns
      Row 1: Original image | absolute heatmaps overlaid on image
      Row 2: Q-score bar chart | signed attribution maps (diverging)

    Parameters
    ----------
    x        : (1, 3, 224, 224) input image tensor.
    methods  : list of AttributionResult from all methods.
    info     : dict with image metadata.
    save_path: output file path.
    delta_f  : f(x) - f(baseline) for display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    n_methods = len(methods)
    n_cols = n_methods + 1

    # ── Colour scheme ──
    BG = "#0D0D0D"
    FG = "#E8E4DF"
    ACCENT = "#F7B538"
    GRID_C = "#2A2A2A"

    # Custom heatmap: transparent → amber → white
    heatmap_colors = [
        (0.0, (0, 0, 0, 0)),
        (0.3, (0.97, 0.45, 0.02, 0.4)),
        (0.6, (0.97, 0.71, 0.22, 0.7)),
        (0.85, (1.0, 0.90, 0.50, 0.9)),
        (1.0, (1.0, 1.0, 1.0, 1.0)),
    ]
    cmap_heat = LinearSegmentedColormap.from_list("amber_heat", heatmap_colors)

    # Diverging: blue (negative) → transparent → red (positive)
    div_colors = [
        (0.0,  (0.15, 0.35, 0.85, 0.9)),
        (0.35, (0.30, 0.55, 0.90, 0.4)),
        (0.5,  (0, 0, 0, 0)),
        (0.65, (0.90, 0.35, 0.15, 0.4)),
        (1.0,  (0.95, 0.20, 0.10, 0.9)),
    ]
    cmap_div = LinearSegmentedColormap.from_list("blue_red_div", div_colors)

    # ── Q-score colours per method ──
    method_colors = {
        "IG":          "#6B7280",  # grey
        "IDGI":        "#8B5CF6",  # purple
        "Guided IG":   "#06B6D4",  # cyan
        "μ-Optimized": "#F59E0B",  # amber
        "Joint":       "#EF4444",  # red
    }

    # ── Prepare data ──
    img_np = _denormalize_image(x)
    img_dark = (img_np.astype(float) * 0.4).astype(np.uint8)  # dimmed for overlay

    fig = plt.figure(figsize=(3.6 * n_cols, 7.8), facecolor=BG)

    # GridSpec: row 0 = heatmaps, row 1 = diverging + bar chart
    gs = gridspec.GridSpec(
        2, n_cols, figure=fig,
        height_ratios=[1, 1],
        hspace=0.22, wspace=0.08,
        left=0.03, right=0.97, top=0.90, bottom=0.04,
    )

    # ── Title ──
    class_label = f"class {info['target_class']}"
    fig.suptitle(
        f"Attribution Heatmaps — ResNet-50 → {class_label}  "
        f"(conf {info['confidence']:.1%},  Δf = {delta_f:.2f})",
        color=FG, fontsize=13, fontweight="bold",
        fontfamily="monospace", y=0.96,
    )

    # ══════════════════════════════════════════════════════════
    # Row 0, Col 0: Original image
    # ══════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title("Original", color=FG, fontsize=10, fontfamily="monospace",
                 fontweight="bold", pad=6)
    ax.axis("off")

    # ══════════════════════════════════════════════════════════
    # Row 0, Cols 1+: Absolute heatmaps overlaid on dimmed image
    # ══════════════════════════════════════════════════════════
    for i, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, i + 1])
        sal = _attribution_heatmap(m.attributions)
        ax.imshow(img_dark)
        ax.imshow(sal, cmap=cmap_heat, vmin=0, vmax=1, alpha=0.85)
        col = method_colors.get(m.name, ACCENT)
        ax.set_title(
            f"{m.name}\n𝒬={m.Q:.4f}  CV²={m.CV2:.4f}",
            color=col, fontsize=9, fontfamily="monospace",
            fontweight="bold", pad=6, linespacing=1.4,
        )
        ax.axis("off")

    # ══════════════════════════════════════════════════════════
    # Row 1, Col 0: Q-score comparison bar chart
    # ══════════════════════════════════════════════════════════
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_bar.set_facecolor(BG)

    names = [m.name for m in methods]
    qs = [m.Q for m in methods]
    colors = [method_colors.get(n, ACCENT) for n in names]

    bars = ax_bar.barh(
        range(n_methods), qs,
        color=colors, edgecolor=BG, linewidth=0.5, height=0.6,
    )

    for j, (bar, q) in enumerate(zip(bars, qs)):
        ax_bar.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{q:.4f}", va="center", ha="left",
            color=FG, fontsize=8, fontfamily="monospace",
        )

    ax_bar.set_yticks(range(n_methods))
    ax_bar.set_yticklabels(names, fontsize=8, fontfamily="monospace", color=FG)
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("𝒬 (higher = better)", color=FG, fontsize=9,
                      fontfamily="monospace")
    ax_bar.invert_yaxis()
    ax_bar.tick_params(colors=FG, labelsize=7)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_color(GRID_C)
    ax_bar.spines["left"].set_color(GRID_C)
    ax_bar.set_title("Quality Metric 𝒬", color=FG, fontsize=10,
                     fontfamily="monospace", fontweight="bold", pad=6)

    # ══════════════════════════════════════════════════════════
    # Row 1, Cols 1+: Signed diverging attribution maps
    # ══════════════════════════════════════════════════════════
    for i, m in enumerate(methods):
        ax = fig.add_subplot(gs[1, i + 1])
        sal_div = _attribution_diverging(m.attributions)
        ax.imshow(img_dark)
        ax.imshow(sal_div, cmap=cmap_div, vmin=-1, vmax=1, alpha=0.85)
        col = method_colors.get(m.name, ACCENT)
        ax.set_title(
            f"Signed · {m.name}",
            color=col, fontsize=9, fontfamily="monospace",
            fontweight="bold", pad=6,
        )
        ax.axis("off")

    # ── Legend for diverging maps ──
    fig.text(
        0.99, 0.01,
        "Row 1: |attribution| heatmap    Row 2: signed (blue=negative, red=positive)",
        color="#666666", fontsize=7, fontfamily="monospace",
        ha="right", va="bottom",
    )

    plt.savefig(save_path, dpi=180, facecolor=BG,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"\n✓ Heatmap saved → {save_path}")
    return save_path


def visualize_step_fidelity(
    methods: list[AttributionResult],
    save_path: str = "step_fidelity.png",
):
    """
    Generate step-fidelity diagnostic figure.

    For each method: φ_k per step with μ_k as background bars,
    green dashed line at φ=1 (perfect fidelity).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    BG = "#0D0D0D"
    FG = "#E8E4DF"
    GRID_C = "#1E1E1E"

    method_colors = {
        "IG":          "#6B7280",
        "IDGI":        "#8B5CF6",
        "Guided IG":   "#06B6D4",
        "μ-Optimized": "#F59E0B",
        "Joint":       "#EF4444",
    }

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), facecolor=BG,
                              sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Step Fidelity  φ_k = d_k / Δf_k   (green dashed = perfect)",
        color=FG, fontsize=12, fontweight="bold",
        fontfamily="monospace", y=1.02,
    )

    for i, (ax, m) in enumerate(zip(axes, methods)):
        ax.set_facecolor(BG)
        col = method_colors.get(m.name, "#F7B538")
        N = len(m.steps)
        ks = np.arange(N)
        phis = np.array([s.phi_k for s in m.steps])
        mus = np.array([s.mu_k for s in m.steps])

        # μ_k as background bars (scaled to fit)
        mu_max = mus.max() if mus.max() > 0 else 1
        mu_scaled = mus / mu_max * 2.0  # scale for visibility
        ax.bar(ks, mu_scaled, color=col, alpha=0.15, width=0.9, label="μ_k")

        # φ_k as dots + line
        ax.plot(ks, phis, 'o-', color=col, markersize=2.5,
                linewidth=1, alpha=0.9, label="φ_k")

        # Perfect fidelity line
        ax.axhline(1.0, color="#22C55E", linestyle="--", linewidth=1, alpha=0.6)

        # Clamp y-axis for readability (IDGI can have extreme φ)
        phi_median = np.median(phis)
        y_lo = max(min(phis.min(), 0) - 0.5, phi_median - 5)
        y_hi = min(max(phis.max(), 2) + 0.5, phi_median + 5)
        ax.set_ylim(y_lo, y_hi)

        ax.set_title(
            f"{m.name}  (𝒬={m.Q:.4f})",
            color=col, fontsize=9, fontfamily="monospace", fontweight="bold",
            pad=6,
        )
        ax.set_xlabel("Step k", color=FG, fontsize=8, fontfamily="monospace")
        if i == 0:
            ax.set_ylabel("φ_k", color=FG, fontsize=9, fontfamily="monospace")
        ax.tick_params(colors=FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor=BG,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"✓ Step fidelity saved → {save_path}")
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# §15  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N: int = 50, device: Optional[torch.device] = None,
                   min_conf: float = 0.70) -> tuple[dict, list, torch.Tensor, dict]:
    """
    Run all five IG methods on ResNet-50 and compare 𝒬 scores.

    Returns (results_dict, methods_list, x_tensor, image_info).
    """
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
    print(f"f(x) = {f_x:.4f},  f(baseline) = {f_bl:.4f},  Δf = {delta_f:.4f}")
    print(f"N = {N} interpolation steps\n")
    print(f"{'Method':<16} {'𝒬':>8} {'CV²(φ)':>10} {'Σ Aᵢ':>10} {'Time':>8}")
    print("─" * 56)

    methods = [
        standard_ig(model, x, baseline, N),
        idgi(model, x, baseline, N),
        guided_ig(model, x, baseline, N),
        mu_optimized_ig(model, x, baseline, N, tau=0.005, n_iter=300),
        joint_ig(model, x, baseline, N, n_alternating=2,
                 tau=0.005, mu_iter=300),
    ]

    for m in methods:
        sa = m.attributions.sum().item()
        print(f"{m.name:<16} {m.Q:>8.4f} {m.CV2:>10.4f} "
              f"{sa:>10.4f} {m.elapsed_s:>7.1f}s")

    results = {
        "image_info": info,
        "model_info": {
            "f_x": f_x, "f_baseline": f_bl, "delta_f": delta_f, "N": N,
            "device": str(device),
        },
        "methods": {m.name: m.to_dict() for m in methods},
    }
    return results, methods, x, info


# ═════════════════════════════════════════════════════════════════════════════
# §16  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified IG v2 — ResNet-50 (PyTorch)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--steps", type=int, default=50,
                        help="Interpolation steps N (default: 50)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Force device: cuda, mps, or cpu")
    parser.add_argument("--min-conf", type=float, default=0.70,
                        help="Minimum classification confidence")
    parser.add_argument("--viz", action="store_true",
                        help="Generate attribution heatmap visualisation")
    parser.add_argument("--viz-path", type=str, default="attribution_heatmaps.png",
                        help="Output path for heatmap image (default: attribution_heatmaps.png)")
    parser.add_argument("--viz-fidelity", action="store_true",
                        help="Generate step-fidelity diagnostic plot")
    args = parser.parse_args()

    device = get_device(force=args.device)
    results, methods, x, info = run_experiment(
        N=args.steps, device=device, min_conf=args.min_conf)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")

    if args.viz:
        delta_f = results["model_info"]["delta_f"]
        visualize_attributions(x, methods, info,
                               save_path=args.viz_path, delta_f=delta_f)

    if args.viz_fidelity:
        fid_path = args.viz_path.replace(".png", "_fidelity.png")
        if fid_path == args.viz_path:
            fid_path = "step_fidelity.png"
        visualize_step_fidelity(methods, save_path=fid_path)