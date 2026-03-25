# ═════════════════════════════════════════════════════════════════════════════
# §1  DEVICE SELECTION
# ═════════════════════════════════════════════════════════════════════════════

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

from indel_utils import AttributionResult
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

def visualize_insertion_deletion(
    methods: list[AttributionResult],
    save_path: str = "insertion_deletion.png",
):
    """
    Generate insertion/deletion curve figure.

    Left panel: Insertion curves (higher = better).
    Right panel: Deletion curves (lower = better).
    Bottom: AUC summary bar chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Filter methods that have insdel scores
    scored = [m for m in methods if m.insdel is not None]
    if not scored:
        print("⚠ No insertion/deletion scores computed — skipping plot.")
        return None

    BG = "#0D0D0D"
    FG = "#E8E4DF"
    GRID_C = "#2A2A2A"

    method_colors = {
        "IG":          "#6B7280",
        "IDGI":        "#8B5CF6",
        "Guided IG":   "#06B6D4",
        "μ-Optimized": "#F59E0B",
        "Joint":       "#EF4444",
    }

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[2, 1],
        hspace=0.30, wspace=0.25,
        left=0.08, right=0.96, top=0.90, bottom=0.08,
    )

    fig.suptitle(
        "Insertion / Deletion Evaluation  (Petsiuk et al., 2018)",
        color=FG, fontsize=13, fontweight="bold",
        fontfamily="monospace", y=0.96,
    )

    # ── Top-left: Insertion curves ──
    ax_ins = fig.add_subplot(gs[0, 0])
    ax_ins.set_facecolor(BG)
    for m in scored:
        col = method_colors.get(m.name, "#F7B538")
        curve = m.insdel.insertion_curve
        xs = np.linspace(0, 1, len(curve))
        ax_ins.plot(xs, curve, color=col, linewidth=1.8,
                    label=f"{m.name} (AUC={m.insdel.insertion_auc:.3f})",
                    alpha=0.9)

    ax_ins.set_title("Insertion (higher AUC = better)",
                     color="#22C55E", fontsize=11, fontfamily="monospace",
                     fontweight="bold", pad=8)
    ax_ins.set_xlabel("Fraction of pixels inserted", color=FG, fontsize=9,
                      fontfamily="monospace")
    ax_ins.set_ylabel("Target class logit", color=FG, fontsize=9,
                      fontfamily="monospace")
    ax_ins.legend(fontsize=7, facecolor=BG, edgecolor=GRID_C,
                  labelcolor=FG, loc="lower right",
                  prop={"family": "monospace"})
    ax_ins.tick_params(colors=FG, labelsize=7)
    ax_ins.grid(True, color=GRID_C, linewidth=0.3, alpha=0.5)
    for spine in ax_ins.spines.values():
        spine.set_color(GRID_C)

    # ── Top-right: Deletion curves ──
    ax_del = fig.add_subplot(gs[0, 1])
    ax_del.set_facecolor(BG)
    for m in scored:
        col = method_colors.get(m.name, "#F7B538")
        curve = m.insdel.deletion_curve
        xs = np.linspace(0, 1, len(curve))
        ax_del.plot(xs, curve, color=col, linewidth=1.8,
                    label=f"{m.name} (AUC={m.insdel.deletion_auc:.3f})",
                    alpha=0.9)

    ax_del.set_title("Deletion (lower AUC = better)",
                     color="#EF4444", fontsize=11, fontfamily="monospace",
                     fontweight="bold", pad=8)
    ax_del.set_xlabel("Fraction of pixels deleted", color=FG, fontsize=9,
                      fontfamily="monospace")
    ax_del.set_ylabel("Target class logit", color=FG, fontsize=9,
                      fontfamily="monospace")
    ax_del.legend(fontsize=7, facecolor=BG, edgecolor=GRID_C,
                  labelcolor=FG, loc="upper right",
                  prop={"family": "monospace"})
    ax_del.tick_params(colors=FG, labelsize=7)
    ax_del.grid(True, color=GRID_C, linewidth=0.3, alpha=0.5)
    for spine in ax_del.spines.values():
        spine.set_color(GRID_C)

    # ── Bottom-left: Insertion AUC bar chart ──
    ax_bar_ins = fig.add_subplot(gs[1, 0])
    ax_bar_ins.set_facecolor(BG)
    names = [m.name for m in scored]
    ins_aucs = [m.insdel.insertion_auc for m in scored]
    colors = [method_colors.get(n, "#F7B538") for n in names]

    bars = ax_bar_ins.barh(range(len(scored)), ins_aucs,
                           color=colors, edgecolor=BG, height=0.55)
    for j, (bar, auc) in enumerate(zip(bars, ins_aucs)):
        ax_bar_ins.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{auc:.4f}", va="center", ha="left",
            color=FG, fontsize=8, fontfamily="monospace",
        )
    ax_bar_ins.set_yticks(range(len(scored)))
    ax_bar_ins.set_yticklabels(names, fontsize=8, fontfamily="monospace",
                               color=FG)
    ax_bar_ins.invert_yaxis()
    ax_bar_ins.set_title("Insertion AUC ↑", color="#22C55E", fontsize=10,
                         fontfamily="monospace", fontweight="bold", pad=6)
    ax_bar_ins.tick_params(colors=FG, labelsize=7)
    ax_bar_ins.spines["top"].set_visible(False)
    ax_bar_ins.spines["right"].set_visible(False)
    ax_bar_ins.spines["bottom"].set_color(GRID_C)
    ax_bar_ins.spines["left"].set_color(GRID_C)

    # ── Bottom-right: Deletion AUC bar chart ──
    ax_bar_del = fig.add_subplot(gs[1, 1])
    ax_bar_del.set_facecolor(BG)
    del_aucs = [m.insdel.deletion_auc for m in scored]

    bars = ax_bar_del.barh(range(len(scored)), del_aucs,
                           color=colors, edgecolor=BG, height=0.55)
    for j, (bar, auc) in enumerate(zip(bars, del_aucs)):
        ax_bar_del.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{auc:.4f}", va="center", ha="left",
            color=FG, fontsize=8, fontfamily="monospace",
        )
    ax_bar_del.set_yticks(range(len(scored)))
    ax_bar_del.set_yticklabels(names, fontsize=8, fontfamily="monospace",
                               color=FG)
    ax_bar_del.invert_yaxis()
    ax_bar_del.set_title("Deletion AUC ↓", color="#EF4444", fontsize=10,
                         fontfamily="monospace", fontweight="bold", pad=6)
    ax_bar_del.tick_params(colors=FG, labelsize=7)
    ax_bar_del.spines["top"].set_visible(False)
    ax_bar_del.spines["right"].set_visible(False)
    ax_bar_del.spines["bottom"].set_color(GRID_C)
    ax_bar_del.spines["left"].set_color(GRID_C)

    plt.savefig(save_path, dpi=180, facecolor=BG,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"✓ Insertion/Deletion saved → {save_path}")
    return save_path


def visualize_step_fidelity(
    methods: list[AttributionResult],
    save_path: str = "step_fidelity.png",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    BG, FG, GRID_C = "#0D0D0D", "#E8E4DF", "#1E1E1E"
    method_colors  = {
        "IG": "#6B7280", "IDGI": "#8B5CF6", "Guided IG": "#06B6D4",
        "μ-Optimized": "#F59E0B", "Joint": "#EF4444",
    }

    n    = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3.5),
                             facecolor=BG, sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Step Fidelity  φ_k = d_k / Δf_k   (green dashed = perfect)",
                 color=FG, fontsize=12, fontweight="bold",
                 fontfamily="monospace", y=1.02)

    for ax, m in zip(axes, methods):
        ax.set_facecolor(BG)
        col   = method_colors.get(m.name, "#F7B538")
        N     = len(m.steps)
        ks    = range(N)
        phis  = [s.phi_k for s in m.steps]
        mus   = [s.mu_k  for s in m.steps]
        phis_np = __import__("numpy").array(phis)
        mus_np  = __import__("numpy").array(mus)

        mu_max    = max(mus_np.max(), 1e-9)
        mu_scaled = mus_np / mu_max * 2.0
        ax.bar(ks, mu_scaled, color=col, alpha=0.15, width=0.9)
        ax.plot(ks, phis, 'o-', color=col, markersize=2.5, linewidth=1, alpha=0.9)
        ax.axhline(1.0, color="#22C55E", linestyle="--", linewidth=1, alpha=0.6)

        phi_med = __import__("numpy").median(phis_np)
        ax.set_ylim(max(phis_np.min() - 0.5, phi_med - 5),
                    min(phis_np.max() + 0.5, phi_med + 5))
        ax.set_title(f"{m.name}  (𝒬={m.Q:.4f})", color=col, fontsize=9,
                     fontfamily="monospace", fontweight="bold", pad=6)
        ax.set_xlabel("Step k", color=FG, fontsize=8, fontfamily="monospace")
        ax.tick_params(colors=FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor=BG,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"✓ Step fidelity saved → {save_path}")
    return save_path