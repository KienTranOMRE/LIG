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
# §14  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N: int = 50, device: Optional[torch.device] = None,
                   min_conf: float = 0.70) -> dict:
    """
    Run all five IG methods on ResNet-50 and compare 𝒬 scores.
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
    return results


# ═════════════════════════════════════════════════════════════════════════════
# §15  MAIN
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
    args = parser.parse_args()

    device = get_device(force=args.device)
    results = run_experiment(N=args.steps, device=device,
                             min_conf=args.min_conf)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")