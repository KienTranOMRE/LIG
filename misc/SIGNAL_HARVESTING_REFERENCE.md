# Signal-Harvesting Integrated Gradients - Reference Guide

**Paper:** "The Least Action Principle for Integrated Gradients: A Unified Variational Framework"
**Implementation:** `signal_lam.py`
**Date:** March 2026

---

## Core Concept

**The Unified Variational Objective** unifies all IG variants under a single framework:

```
min_{γ,μ}  Var_ν(φ) - λ Σ_k μ_k |d_k| + (τ/2) ||μ||²₂
           ↑           ↑                  ↑
         Term 1      Term 2            Term 3
```

### Three Terms Explained

1. **Var_ν(φ)** - Linearization distortion (Fermat/Snell conservation law)
   - Measures how well gradient predictions match actual output changes
   - Original LAM objective (λ=0 case)

2. **-λ Σ_k μ_k |d_k|** - Signal harvesting (NEW CONTRIBUTION)
   - Forces μ to concentrate on steps where |d_k| is large
   - Prevents degenerate basin where μ focuses on flat regions
   - λ controls interpolation: λ=0 (pure conservation) ↔ λ→∞ (pure IDGI)

3. **(τ/2) ||μ||²** - L2 admissibility
   - Prevents μ from collapsing to Dirac spike
   - LINEAR stationary condition (unlike entropy) → recovers IDGI exactly
   - τ ∈ [0.005, 0.01] recommended

---

## Key Variables & Notation

| Symbol | Meaning | Computation |
|--------|---------|-------------|
| **d_k** | Gradient-predicted output change | ∇f(γ_k) · Δγ_k |
| **Δf_k** | Actual output change | f(γ_{k+1}) - f(γ_k) |
| **φ_k** | Step fidelity | d_k / Δf_k |
| **μ** | Attribution measure | Probability distribution over N steps |
| **ν_k** | Effective measure | μ_k Δf²_k / Σ_j μ_j Δf²_j |
| **γ** | Path | N+1 points from baseline to input |

---

## Mathematical Results

### Stationary Condition for μ (Eq. 14-15)

At Var_ν(φ) = 0, setting ∂/∂μ_k = 0 gives:

```
μ*_k ∝ |d_k|/τ ≈ |Δf_k|/τ
```

**This is EXACTLY IDGI's measure** - now derived from first principles!

### Why L2 instead of Entropy?

| Regularization | Stationary Condition | Recovers IDGI? |
|----------------|---------------------|----------------|
| **(τ/2)||μ||²** (L2) | μ*_k ∝ \|d_k\|/τ (LINEAR) | ✓ Yes |
| τΣμ_k log μ_k (Entropy) | μ*_k ∝ exp(\|d_k\|/τ) (EXPONENTIAL) | ✗ No |

### Euler-Lagrange for Path γ (Eq. 16)

```
d/dt[∂L/∂γ'] - ∂L/∂γ = 0  ⟹  forcing term ∝ λμ* · sgn(∇f·γ') · H_f γ'
```

The forcing term pushes γ toward **output-transition regions** (Guided IG behavior).

---

## All IG Methods as Special Cases (Table 1)

| Method | Path γ | λ | τ | Resulting μ* |
|--------|--------|---|---|--------------|
| **Standard IG** | fixed line | 0 | ∞ | uniform |
| **IDGI** | fixed line | >0 | →0 | μ_k ∝ \|Δf_k\| |
| **Guided IG** | heuristic | >0 | — | uniform |
| **μ-Optimised** | fixed line | 0 | >0 | min Var_ν only |
| **Joint (LAM)** | optimised | 0 | >0 | min Var_ν only |
| **Joint*** | optimised | >0 | >0 | min Var_ν - λΣμ\|d\| |

---

## Implementation Guide

### File: `signal_lam.py`

### §1: Core Objective Function

```python
compute_signal_harvesting_objective(d, delta_f, mu, lam=1.0, tau=0.01)
# Returns: (total_objective, var_nu_term, signal_term, l2_term)
```

### §2: Closed-Form μ* (KKT Stationary Point)

```python
mu_star_closed_form(d, delta_f, mode="d")
# mode="d"  : μ*_k ∝ |d_k|    (exact KKT)
# mode="df" : μ*_k ∝ |Δf_k|  (IDGI approximation)
```

### §3: μ-Optimization with Signal Harvesting (MOST PRACTICAL)

```python
optimize_mu_signal_harvesting(d, delta_f, lam=1.0, tau=0.01, n_iter=300, lr=0.05)
# Cost: O(N) arithmetic per iteration, ZERO extra model evaluations
# Limiting behavior:
#   λ → 0 : μ* → arg min Var_ν(φ)  (original LAM)
#   λ → ∞ : μ* → |d_k|/Σ|d_j|      (IDGI)
```

### §4: μ-Optimized IG (Straight Line + Optimal μ)

```python
mu_optimized_ig(model, x, baseline, N=50, lam=1.0, tau=0.01, n_iter=300)
# RECOMMENDED: Zero extra model evaluations beyond standard IG
# Returns: AttributionResult with Q, Var_ν, CV², steps, etc.
```

### §5-6: Path Optimization

```python
optimize_path_signal_harvesting(model, x, baseline, mu, N=50, G=16,
                                 patch_size=14, n_iter=15, lr=0.08, lam=1.0)
# Uses grouped velocity scheduling
# Cost: O(G) batched model evaluations per iteration
```

### §7: Joint* Optimization (COMPLETE SOLUTION)

```python
joint_star_ig(model, x, baseline, N=50, G=16, patch_size=14,
              n_alternating=2, lam=1.0, tau=0.01,
              mu_iter=300, path_iter=10, init_path=None)

# Alternating minimization:
#   Phase 1 (measure): fix γ, optimize μ via Eq. 24
#   Phase 2 (path):    fix μ, optimize γ via velocity scheduling
# Cost: 3-5× standard IG
# Regression guard: only accepts path if objective improves
```

### §8: Convenience Runner

```python
run_all_methods(model, x, baseline, N=50, lam=1.0, tau=0.01,
                G=16, patch_size=14, n_alternating=2,
                mu_iter=300, path_iter=10, guided_init=False)

# Returns: [IG, IDGI, Guided IG, μ-Optimized*, Joint(λ=0), Joint*]
```

---

## Hyperparameter Selection

### Signal-Harvesting Strength: λ

```python
λ = 0.0    # Pure conservation (original LAM)
λ = 0.5    # Mild signal harvesting
λ = 1.0    # Balanced (RECOMMENDED default)
λ = 2.0    # Strong signal harvesting
λ → ∞      # Pure IDGI limit
```

**Recommendation:** λ ∈ [0.5, 2.0], tune on validation set using insertion/deletion AUC

### L2 Admissibility: τ

```python
τ = 0.005  # Allows more concentration
τ = 0.01   # RECOMMENDED default
τ = 0.05   # Forces more uniformity
τ → 0      # Approaches IDGI (spike at max |d_k|)
τ → ∞      # Forces uniform μ
```

**Recommendation:** τ ∈ [0.005, 0.01]

### Path Optimization

```python
G = 16           # Number of spatial groups (more = finer control, slower)
patch_size = 14  # Grid patch size for grouping
n_alternating = 2  # Alternating iterations (2-3 sufficient)
mu_iter = 300    # Iterations for μ optimization
path_iter = 10   # Iterations for path optimization per alternation
```

---

## Quality Metrics

### Primary: Q (Quality)

```python
Q = 1 / (1 + CV²(φ))  ∈ [0, 1]

Q = 1  ⟺  φ_k = constant (perfect conservation law)
Q → 0  ⟺  high variance in step fidelity
```

### Components:

```python
Var_ν(φ) = Σ_k ν_k (φ_k - φ̄_ν)²    # Variance of step fidelity
CV²(φ) = Var_ν(φ) / φ̄²_ν           # Coefficient of variation squared
```

### Objective Value:

```python
Obj = Var_ν(φ) - λ·(Σ μ_k |d_k|) + (τ/2)·||μ||²
# Lower is better
# Joint* minimizes this directly
```

---

## Usage Patterns

### Pattern 1: Drop-in IDGI Replacement (Zero Extra Cost)

```python
from signal_lam import mu_optimized_ig

result = mu_optimized_ig(
    model, x, baseline,
    N=50,
    lam=1.0,    # Use signal harvesting
    tau=0.01
)

print(f"Q = {result.Q:.4f}")
print(f"Var_ν = {result.Var_nu:.6f}")
attributions = result.attributions
```

### Pattern 2: Full Joint Optimization

```python
from signal_lam import joint_star_ig, guided_ig

# Optional: warm-start from Guided IG
gig = guided_ig(model, x, baseline, N=50)

result = joint_star_ig(
    model, x, baseline,
    N=50,
    lam=1.0,
    tau=0.01,
    n_alternating=2,
    init_path=gig.gamma_pts  # Warm start
)

# Track optimization progress
for h in result.Q_history:
    print(f"Iter {h['iteration']}: Q_mu={h['Q_after_mu']:.4f}, "
          f"Q_path={h['Q_after_path']:.4f}, best_Q={h['best_Q']:.4f}")
```

### Pattern 3: Method Comparison

```python
from signal_lam import run_all_methods

methods = run_all_methods(
    model, x, baseline,
    N=50, lam=1.0, tau=0.01,
    guided_init=True  # Warm-start Joint methods
)

for m in methods:
    print(f"{m.name:<16} Q={m.Q:.4f} Var_ν={m.Var_nu:.6f} time={m.elapsed_s:.1f}s")
```

### Pattern 4: Command-Line Experiment

```bash
python signal_lam.py \
    --steps 50 \
    --lam 1.0 \
    --tau 0.01 \
    --guided-init \
    --viz \
    --viz-path results/attributions.png \
    --insdel \
    --viz-insdel \
    --json results/metrics.json
```

---

## Key Insights from Paper

### Why Original LAM Has Issues

1. **Degenerate basin:** Var_ν(φ) → 0 can be achieved by concentrating μ on flat regions where φ_k ≈ c ≠ 1
2. **No signal preference:** Var_ν(φ) doesn't distinguish between transition and flat regions
3. **Signal harvesting solves both:** -λΣμ_k|d_k| forces μ toward high-|d_k| steps

### Why IDGI Dominates Empirically

IDGI is the **exact stationary measure** of the signal-harvesting action at λ > 0. Original LAM methods (Joint, μ-Opt) operate at λ = 0 and miss signal harvesting entirely.

### Physical Intuition

The optimal attribution system is a **signal-harvesting optical instrument**:
- **Fermat/Snell:** Bend path to avoid high-curvature regions (minimize Var_ν)
- **Signal harvesting:** Concentrate detection window on bright transition region (maximize -λΣμ|d|)
- **Both are necessary:** Neither alone is sufficient

### Computational Trade-offs

| Method | Extra Model Evals | Quality Gain | Use Case |
|--------|-------------------|--------------|----------|
| **μ-Optimized*** | 0 | High | Production (cheap, effective) |
| **Joint*** | O(G·n_iter·n_alt) | Highest | Research (best quality) |

---

## Dependencies

```python
# From existing codebase
from utilss import (
    AttributionResult, StepInfo,
    compute_Var_nu, compute_CV2, compute_Q, compute_all_metrics,
)

from lam import (
    _forward_scalar, _forward_batch, _forward_and_gradient,
    _forward_and_gradient_batch, _gradient, _gradient_batch,
    _dot, _rescale, _build_steps, _straight_line_pass,
    _build_spatial_groups, _build_path_2d, _eval_path_batched,
)
```

---

## Common Q&A

**Q: When should I use μ-Optimized* vs Joint*?**
A: Use μ-Optimized* for production (zero extra cost). Use Joint* when you need maximum quality and can afford 3-5× cost.

**Q: How does this relate to IDGI?**
A: IDGI is the exact solution when λ>0, τ→0, straight-line path. μ-Optimized* generalizes IDGI by also minimizing Var_ν.

**Q: What if Q is still low?**
A: Try increasing λ (stronger signal harvesting), check if model has high curvature, or use Joint* to optimize path.

**Q: Can I use this with non-image inputs?**
A: Yes. Path optimization (Joint*) uses spatial grouping optimized for images, but μ-Optimized* works on any input.

**Q: How do I choose λ?**
A: Start with λ=1.0. If Var_ν is high but signal is harvested well, decrease λ. If attributions are noisy, increase λ.

---

## Experimental Results (Paper Section 8)

### Synthetic (MLP, N=20)
- **μ-Opt**: Mean Q=0.961, Std=0.022 (very consistent)
- **Joint**: Mean Q=0.972, Std=0.016 (best quality)
- Standard IG: Mean Q=0.812, Std=0.270 (high variance)

### Vision (ResNet-50, N=50)
- Joint* achieves lowest Var_ν while maintaining high signal harvesting
- Insertion/deletion AUC improvements over Standard IG
- 3-5× wall-clock cost vs Standard IG

---

## References

**Main Paper:**
"The Least Action Principle for Integrated Gradients: A Unified Variational Framework"
Draft Report, March 27, 2026

**Key Equations:**
- Eq. 11: Continuous signal-harvesting action
- Eq. 14-15: Stationary measure μ*_k ∝ |d_k|
- Eq. 16: Euler-Lagrange with forcing term
- Eq. 20: Discrete unified objective
- Eq. 24: μ-optimization sub-problem

**Prior Work:**
- Sundararajan et al. 2017: Original Integrated Gradients
- Sikdar et al. 2021: IDGI (now derived as special case)
- Kapishnikov et al. 2021: Guided IG (now derived as approximate stationary path)

---

## Quick Start Code

```python
#!/usr/bin/env python3
"""Minimal example of signal-harvesting IG."""

import torch
from signal_lam import mu_optimized_ig, joint_star_ig
from lam import load_image_and_model

# Load model and image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, x, baseline, info = load_image_and_model(device, min_conf=0.70)

# Method 1: μ-Optimized (RECOMMENDED - zero extra cost)
result_mu = mu_optimized_ig(
    model, x, baseline,
    N=50,
    lam=1.0,      # Signal harvesting strength
    tau=0.01,     # L2 admissibility
    n_iter=300    # Optimization iterations for μ
)

print(f"μ-Optimized: Q={result_mu.Q:.4f}, Var_ν={result_mu.Var_nu:.6f}")

# Method 2: Joint* (full optimization - 3-5× cost)
result_joint = joint_star_ig(
    model, x, baseline,
    N=50,
    lam=1.0,
    tau=0.01,
    n_alternating=2,    # Alternating iterations
    mu_iter=300,        # μ optimization iterations
    path_iter=10        # Path optimization iterations
)

print(f"Joint*: Q={result_joint.Q:.4f}, Var_ν={result_joint.Var_nu:.6f}")

# Access attributions
attributions = result_mu.attributions  # Shape: (1, C, H, W)
```

---

**END OF REFERENCE**
