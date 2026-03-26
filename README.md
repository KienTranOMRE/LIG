# LAM — Least Action Movement for Integrated Gradients

A unified variational framework that reveals all Integrated Gradients (IG) variants as approximate solutions to the same conservation law: **the step fidelity ratio must be constant along the interpolation path**.

This conservation law is the attribution analogue of **Snell's law in optics**, where the Hessian of the model plays the role of refractive index.

## The Idea

Standard IG integrates gradients along a straight line from a baseline to the input. In practice, this path traverses regions where the model's output is flat (wasting interpolation budget) and regions of sharp nonlinearity (where the linear approximation breaks down). Two existing methods address this differently:

- **IDGI** reweights *which steps to trust* (the measure μ)
- **Guided IG** changes *where to evaluate gradients* (the path γ)

LAM shows both are optimising the same objective — minimising the weighted variance of step fidelity:

```
min_{γ,μ}  Var_ν(φ)  =  Σ_k ν_k (φ_k − φ̄_ν)²
```

where φ_k = d_k / Δf_k is the ratio of gradient-predicted to actual output change at step k.

## Methods

| Method | Path γ | Measure μ | What it optimises |
|--------|--------|-----------|-------------------|
| Standard IG | straight line | uniform | nothing |
| IDGI | straight line | μ_k ∝ \|Δf_k\| | μ heuristic |
| Guided IG | low-grad-first | uniform | γ heuristic |
| μ-Optimised | straight line | optimal | min CV²(φ) over μ |
| **Joint (ours)** | **optimal** | **optimal** | **alternating min over (γ, μ)** |

## Results

ResNet-50 on ImageNet, N = 50 interpolation steps, zero baseline:

```
Method                Var_ν      CV²        𝒬     Time
────────────────────────────────────────────────────────
IG                 0.015749   0.0278   0.9730     0.1s
IDGI               0.005221   0.0100   0.9901     0.1s
Guided IG          0.012081   0.0403   0.9612     0.4s
μ-Optimized        0.000254   0.0005   0.9995     0.2s
Joint              0.000222   0.0004   0.9996    38.9s
```

𝒬 = 1/(1 + CV²) is the quality score (1 = perfect conservation). μ-Optimised achieves 𝒬 > 0.999 at zero additional cost over standard IG. Joint pushes further by also optimising the path.

## Files

```
LAM.py         Main framework — all five IG methods + visualisation
utilss.py      Metrics (Var_ν, CV², 𝒬), insertion/deletion evaluation,
               region-based evaluation, plotting utilities
```

## Quick Start

```bash
# Basic run (straight-line init, all 5 methods)
python LAM.py

# With attribution heatmaps and fidelity diagnostics
python LAM.py --viz --viz-fidelity

# Initialise Joint from Guided IG path (often better)
python LAM.py --guided-init

# Export results to JSON
python LAM.py --json results.json

# Fewer steps (faster)
python LAM.py --steps 30

# Force CPU
python LAM.py --device cpu
```

## Evaluation

```bash
# Pixel-level insertion/deletion (Petsiuk et al., 2018)
python LAM.py --insdel --viz-insdel

# Region-based insertion/deletion (SIC-style, uses SLIC superpixels)
python LAM.py --region-insdel --viz-region-insdel

# Region-based with grid patches instead of SLIC
python LAM.py --region-insdel --no-slic --patch-size 16

# Everything
python LAM.py --viz --viz-fidelity --insdel --viz-insdel \
              --region-insdel --viz-region-insdel --guided-init
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--steps N` | Interpolation steps | 50 |
| `--device DEVICE` | Force cuda/cpu | auto |
| `--min-conf F` | Minimum classification confidence | 0.70 |
| `--guided-init` | Init Joint from Guided IG path | off (straight line) |
| `--viz` | Generate attribution heatmaps | off |
| `--viz-path PATH` | Heatmap output path | attribution_heatmaps.png |
| `--viz-fidelity` | Step fidelity diagnostic plot | off |
| `--insdel` | Pixel insertion/deletion scores | off |
| `--insdel-steps N` | Pixel ins/del granularity | 100 |
| `--viz-insdel` | Pixel ins/del curve plot | off |
| `--region-insdel` | Region-based ins/del scores | off |
| `--viz-region-insdel` | Region ins/del curve plot | off |
| `--patch-size N` | Grid patch size for regions | 14 |
| `--no-slic` | Use grid patches instead of SLIC | off (SLIC preferred) |
| `--json PATH` | Export results to JSON | off |

## Quality Metrics

Three related metrics, all derived from the step fidelity φ_k = d_k / Δf_k:

- **Var_ν(φ)** — weighted variance of fidelity under effective measure ν_k ∝ μ_k Δf_k². The primary objective for μ-optimisation.
- **CV²(φ)** = Var_ν(φ) / φ̄² — scale-free coefficient of variation. Used as the optimisation target (prevents degenerate solutions where φ̄ → 0).
- **𝒬** = 1/(1 + CV²) — quality score in [0, 1]. Equals the squared cosine similarity between d and Δf under the μ-weighted inner product. 𝒬 = 1 ⟺ perfect conservation.

## Physical Analogy

The conservation law φ_k = const is structurally identical to Snell's law in optics:

| Concept | Optics (Fermat) | Attribution (LAM) |
|---------|-----------------|-------------------|
| Path | Light ray γ(t) | Interpolation path γ(t) |
| Endpoints | Source, detector | Baseline x', input x |
| Local cost | Refractive index n(x) | Hessian ‖H_f(x)‖ |
| Conservation | n sin θ = const | ρ(t) = const |
| Bending | At media interfaces | At flat/curved transitions of f |

The optimal attribution path bends away from regions of high curvature of f, just as light bends away from optically dense regions.

## Requirements

```
torch >= 2.0
torchvision
matplotlib  (for --viz flags)
scikit-image  (optional, for SLIC superpixels in --region-insdel)
```

## References

- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML 2017.
- Sikdar, Bhatt, Heese. "Integrated Directional Gradients." ACL 2021.
- Kapishnikov, Bolukbasi, Viégas, Terry. "Guided Integrated Gradients." CVPR 2021.
- Petsiuk, Das, Saenko. "RISE: Randomized Input Sampling for Explanation." BMVC 2018.