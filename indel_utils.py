
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

@torch.no_grad()
def compute_insertion_deletion(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    attributions: torch.Tensor,
    n_steps: int = 100,
    batch_size: int = 16,
) -> InsDelScores:
    """
    Insertion and Deletion metrics (Petsiuk et al., BMVC 2018).

    OPTIMISATION vs v2:
    - All S = n_steps+1 masks are built in one vectorised broadcast:
        ranks.unsqueeze(0) < counts.unsqueeze(1)  →  (S, H*W) bool tensor.
      No Python loop over steps for mask construction.
    - Two full (S, C, H, W) image batches (x_del, x_ins) are built with
      torch.where; forward passes are batched in groups of batch_size so
      the GPU stays busy rather than processing one image at a time.
    - model() now returns (B,) so each batch produces a vector of logits
      in one call.
    """
    device   = x.device
    _, C, H, W = x.shape
    n_pixels = H * W

    importance = attributions[0].abs().sum(dim=0).flatten()           # (H*W,)
    sorted_idx = torch.argsort(importance, descending=True)

    # Pixel rank map: ranks[p] = rank of pixel p (0 = most important)
    ranks = torch.empty(n_pixels, dtype=torch.long, device=device)
    ranks[sorted_idx] = torch.arange(n_pixels, device=device)

    counts  = torch.linspace(0, n_pixels, n_steps + 1,
                             device=device).long()                     # (S,)
    S       = counts.shape[0]

    # Build ALL boolean masks at once — no Python loop (OPT #3)
    masks_flat = ranks.unsqueeze(0) < counts.unsqueeze(1)             # (S, H*W)
    masks_4d   = masks_flat.view(S, 1, H, W).expand(S, C, H, W)      # (S,C,H,W)

    x_exp  = x.expand(S, -1, -1, -1)
    bl_exp = baseline.expand(S, -1, -1, -1)

    x_del = torch.where(masks_4d, bl_exp, x_exp)                      # (S,C,H,W)
    x_ins = torch.where(masks_4d, x_exp,  bl_exp)

    del_logits = torch.empty(S, device=device)
    ins_logits = torch.empty(S, device=device)

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)
        del_logits[start:end] = model(x_del[start:end])               # (B,)
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


def run_insertion_deletion(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    methods: list[AttributionResult],
    n_steps: int = 100,
) -> None:
    print(f"\n{'Method':<16} {'Ins AUC':>10} {'Del AUC':>10} {'Ins-Del':>10}")
    print("─" * 50)
    for m in methods:
        scores = compute_insertion_deletion(
            model, x, baseline, m.attributions, n_steps=n_steps)
        m.insdel = scores
        diff = scores.insertion_auc - scores.deletion_auc
        print(f"{m.name:<16} {scores.insertion_auc:>10.4f} "
              f"{scores.deletion_auc:>10.4f} {diff:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# §4c  REGION-BASED INSERTION / DELETION  (OPT #6: cumulative mask)
# ─────────────────────────────────────────────────────────────────────────────

def _build_grid_segments(H: int, W: int, patch_size: int = 14) -> "np.ndarray":
    import numpy as np
    segments = np.zeros((H, W), dtype=int)
    n_rows   = (H + patch_size - 1) // patch_size
    n_cols   = (W + patch_size - 1) // patch_size
    for r in range(n_rows):
        for c in range(n_cols):
            sid  = r * n_cols + c
            r0, r1 = r * patch_size, min((r + 1) * patch_size, H)
            c0, c1 = c * patch_size, min((c + 1) * patch_size, W)
            segments[r0:r1, c0:c1] = sid
    return segments


def _try_slic_segments(x: torch.Tensor,
                       n_segments: int = 200) -> "np.ndarray | None":
    try:
        from skimage.segmentation import slic
        import numpy as np
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
        img  = (x * std + mean).clamp(0, 1)
        img_np = img[0].permute(1,2,0).cpu().numpy()
        return slic(img_np, n_segments=n_segments, compactness=10, start_label=0)
    except ImportError:
        return None


@torch.no_grad()
def compute_region_insertion_deletion(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    attributions: torch.Tensor,
    patch_size: int = 14,
    use_slic: bool = True,
    n_slic_segments: int = 200,
) -> InsDelScores:
    """
    Region-based Insertion/Deletion (SIC-style).

    OPTIMISATION vs v2:
    - Cumulative mask: keep a running bool tensor and OR each new region
      in once instead of rebuilding from scratch at every step.
      Cost goes from O(S²·pixels/S) = O(S·pixels) per-step rebuilds to
      O(pixels/S) per step — a factor of S improvement in mask writes.
    - model() returns (B,) so float(model(img)) squeezes cleanly.
    """
    import numpy as np

    device   = x.device
    _, C, H, W = x.shape

    segments = _try_slic_segments(x, n_slic_segments) if use_slic else None
    if segments is None:
        segments = _build_grid_segments(H, W, patch_size)

    seg_ids          = np.unique(segments)
    n_segments       = len(seg_ids)
    importance_map   = attributions[0].abs().sum(dim=0).cpu().numpy()
    region_importance = np.array([importance_map[segments == sid].mean()
                                   for sid in seg_ids])
    sorted_region_idx = np.argsort(region_importance)[::-1]

    seg_tensor = torch.from_numpy(segments).to(device)

    insertion_curve: list[float] = []
    deletion_curve:  list[float] = []

    # Step 0: empty mask
    mask_2d = torch.zeros(H, W, dtype=torch.bool, device=device)

    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(x)
    insertion_curve.append(float(model(torch.where(mask_4d, x, baseline))))
    deletion_curve .append(float(model(torch.where(mask_4d, baseline, x))))

    # Incremental mask update — OPT #6
    for s in range(n_segments):
        region_id = int(seg_ids[sorted_region_idx[s]])
        mask_2d  |= (seg_tensor == region_id)                  # OR — no rebuild
        mask_4d   = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(x)
        insertion_curve.append(float(model(torch.where(mask_4d, x, baseline))))
        deletion_curve .append(float(model(torch.where(mask_4d, baseline, x))))

    ins_arr = np.array(insertion_curve)
    del_arr = np.array(deletion_curve)
    dx      = 1.0 / n_segments
    insertion_auc = float(np.sum(ins_arr[:-1] + ins_arr[1:]) * dx / 2)
    deletion_auc  = float(np.sum(del_arr[:-1] + del_arr[1:]) * dx / 2)

    f_x         = insertion_curve[-1]
    f_bl        = deletion_curve[-1]
    logit_range = abs(f_x - f_bl)
    if logit_range > 1e-12:
        insertion_auc_norm = (insertion_auc - f_bl) / logit_range
        deletion_auc_norm  = (deletion_auc  - f_bl) / logit_range
    else:
        insertion_auc_norm = deletion_auc_norm = 0.0

    return InsDelScores(
        insertion_auc=insertion_auc_norm,
        deletion_auc=deletion_auc_norm,
        insertion_curve=[float(v) for v in insertion_curve],
        deletion_curve=[float(v) for v in deletion_curve],
        n_steps=n_segments,
        mode="region",
    )


def run_region_insertion_deletion(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    methods: list[AttributionResult],
    patch_size: int = 14,
    use_slic: bool = True,
) -> None:
    seg_type = "SLIC" if use_slic else f"grid-{patch_size}"
    print(f"\nRegion-based Ins/Del ({seg_type})")
    print(f"{'Method':<16} {'R-Ins AUC':>10} {'R-Del AUC':>10} {'R-Diff':>10}")
    print("─" * 50)
    for m in methods:
        scores = compute_region_insertion_deletion(
            model, x, baseline, m.attributions,
            patch_size=patch_size, use_slic=use_slic)
        m.region_insdel = scores
        diff = scores.insertion_auc - scores.deletion_auc
        print(f"{m.name:<16} {scores.insertion_auc:>10.4f} "
              f"{scores.deletion_auc:>10.4f} {diff:>10.4f}")

