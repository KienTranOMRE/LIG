# μ-Optimized Integrated Gradients

**Optimizing Weights for Discretized Integrated Gradients**

A direct optimization framework for finding optimal quadrature weights in discrete Integrated Gradients. The standard IG implementation uses uniform weights (1/N) that treat every step equally, regardless of whether the gradient approximation is faithful at that step. We formulate the problem of finding optimal weights μ ∈ P_N that minimize the variance of step fidelities under an L2 regularization penalty.

**Key Result:** μ-Optimized IG requires **zero additional model evaluations** beyond standard IG and consistently improves the conservation quality metric Q.

## The Problem

Standard Integrated Gradients with N steps computes:

```
A_i = (1/N) Σ_k ∇f(γ_k) · (x_i − x'_i)
```

The uniform weight 1/N treats every step equally. However, at finite N, the linear approximation `∇f(γ_k) · Δγ_k ≈ Δf_k` is not equally accurate everywhere:

- Some steps are **highly faithful**: the gradient predicts the output change well (φ_k ≈ 1)
- Other steps are **poor**: the output changes nonlinearly (φ_k varies wildly)

The uniform weighting averages over all steps indiscriminately, mixing faithful and unfaithful contributions.

## The Solution

Find weights μ ∈ P_N that minimize the variance of step fidelities:

```
min_{μ∈P_N}  Var_ν(φ) + (τ/2) ||μ||²₂
```

where:
- **Var_ν(φ)** — fidelity variance: measures how inconsistently gradients approximate output changes
- **||μ||²₂** — L2 penalty: prevents degenerate weight collapse to a single step
- **τ** — regularization parameter balancing consistency and smoothness (typically 0.005–0.01)

### Step Fidelity

The **step fidelity** at step k is the ratio:

```
φ_k = d_k / Δf_k
```

where:
- `d_k = ∇f(γ_k) · Δγ_k` — gradient-predicted output change
- `Δf_k = f(γ_{k+1}) − f(γ_k)` — actual output change

When φ_k = 1, the first-order Taylor approximation perfectly predicts the actual change.

**Conservation property:** If φ_k = c (constant) for all steps, the weighted attribution is perfectly consistent.

## Methods Compared

| Method | Weights μ | What it optimizes |
|--------|-----------|-------------------|
| Standard IG | uniform (1/N) | nothing |
| IDGI | μ_k ∝ \|Δf_k\| | signal coverage (heuristic) |
| **μ-Optimized** | **optimal** | **min Var_ν(φ) + (τ/2)‖μ‖²** |

### Computational Cost

**Critical observation:** All three methods require the same model evaluations.

| Method | Forward passes | Backward passes | Extra arithmetic |
|--------|----------------|-----------------|------------------|
| Standard IG | N + 1 | N | — |
| IDGI | N + 1 | N | O(N) |
| **μ-Optimized** | **N + 1** | **N** | **O(NT)** |

The μ-optimization loop adds **zero additional model evaluations**. The O(NT) arithmetic cost is negligible compared to the O(N) neural network forward/backward passes.

## Results

ResNet-50 on ImageNet, N = 50 interpolation steps, zero baseline:

```
Method            Var_ν      CV²        𝒬       Time
──────────────────────────────────────────────────────
IG              0.015749   0.0278   0.9730      0.1s
IDGI            0.005221   0.0100   0.9901      0.1s
μ-Optimized     0.000254   0.0005   0.9995      0.2s
```

**𝒬 = 1/(1 + CV²)** is the quality score (1 = perfect conservation).

μ-Optimized achieves **𝒬 > 0.999** at negligible computational cost over standard IG.

## Files

```
u_optimize.py    μ-Optimization framework — three IG methods
lam.py          Base IG implementations and utilities
utilss.py       Metrics (Var_ν, CV², 𝒬), evaluation, plotting
```

## Quick Start

```bash
# Basic run (all 3 methods: IG, IDGI, μ-Optimized)
python u_optimize.py

# With attribution heatmaps and fidelity diagnostics
python u_optimize.py --viz --viz-fidelity

# Export results to JSON
python u_optimize.py --json results.json

# Adjust regularization parameter τ
python u_optimize.py --tau 0.005

# Fewer steps (faster)
python u_optimize.py --steps 30

# Force CPU
python u_optimize.py --device cpu
```

## Evaluation

```bash
# Pixel-level insertion/deletion (Petsiuk et al., 2018)
python u_optimize.py --insdel --viz-insdel

# Region-based insertion/deletion (SIC-style, uses SLIC superpixels)
python u_optimize.py --region-insdel --viz-region-insdel

# Region-based with grid patches instead of SLIC
python u_optimize.py --region-insdel --no-slic --patch-size 16

# Everything
python u_optimize.py --viz --viz-fidelity --insdel --viz-insdel \
                     --region-insdel --viz-region-insdel
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--steps N` | Interpolation steps | 50 |
| `--tau F` | L2 regularization parameter τ | 0.01 |
| `--device DEVICE` | Force cuda/cpu | auto |
| `--min-conf F` | Minimum classification confidence | 0.70 |
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
| `--seed N` | Random seed for reproducibility | 42 |
| `--skip N` | Skip N images in dataset | 0 |

## Quality Metrics

### Effective Measure

Not all steps matter equally. Steps with tiny |Δf_k| carry negligible attribution signal. The **effective measure** captures this:

```
ν_k = (μ_k Δf²_k) / Σ_j μ_j Δf²_j
```

Steps where output is flat (Δf_k ≈ 0) have ν_k ≈ 0 regardless of μ_k.

### Fidelity Variance

The fidelity variance under the effective measure:

```
φ̄_ν = Σ_k ν_k φ_k
Var_ν(φ) = Σ_k ν_k (φ_k − φ̄_ν)²
```

### Quality Metric Q

The attribution quality metric:

```
Q = 1 / (1 + CV²_ν(φ))
```

where `CV²_ν(φ) = Var_ν(φ) / φ̄²_ν` is the squared coefficient of variation.

**Properties:**
- Q ∈ [0, 1]
- Q = 1 ⟺ φ_k = const for all steps with ν_k > 0 (perfect conservation)
- Equivalent form: `Q = (Σ μ_k d_k Δf_k)² / [(Σ μ_k d²_k)(Σ μ_k Δf²_k)]`

This is a squared weighted correlation between d_k and Δf_k under μ (Cauchy-Schwarz ratio).

## The Optimization Algorithm

### Objective

```
min_{μ∈P_N}  -Q(μ) + (τ/2) ||μ||²₂
```

Maximizing Q(μ) is equivalent to minimizing Var_ν(φ) when the mean φ̄_ν varies slowly.

### Gradient

Define:
```
P = Σ_k μ_k d_k Δf_k
D = Σ_k μ_k d²_k
F = Σ_k μ_k Δf²_k
```

Then `Q(μ) = P² / (DF)` and:

```
∂Q/∂μ_k = (P/DF) [2d_k Δf_k − (P/D)d²_k − (P/F)Δf²_k]
```

The L2 penalty contributes: `∂/∂μ_k [(τ/2)||μ||²] = τμ_k`

### Projected Gradient Descent

1. Precompute: `a_k = d_k Δf_k`, `b_k = d²_k`, `c_k = Δf²_k`
2. Initialize: `μ_k = 1/N` for all k
3. For t = 1, ..., T:
   - Compute P, D, F
   - Compute gradient: `g_k = -(P/DF)[2a_k − (P/D)b_k − (P/F)c_k] + τμ_k`
   - Update: `μ ← μ − η·g`
   - Project onto simplex: `μ ← Proj_{P_N}(μ)`

**Cost:** O(N) arithmetic per iteration, operates entirely on precomputed vectors d_k, Δf_k.

## Why L2 Regularization?

Without the L2 penalty (τ = 0), minimizing Var_ν(φ) alone admits **degenerate solutions**:

**Example:** Consider N = 100 steps where:
- Steps 40–60: transition region (|Δf_k| ≫ 0, φ_k varies)
- Other steps: flat region (|Δf_k| ≈ 0)

Minimizing Var_ν(φ) alone can yield μ concentrated on flat steps, where φ_k is trivially near-constant because both d_k and Δf_k are near zero. The resulting Q ≈ 1 is **vacuous**: attributions computed from steps where f doesn't change.

The L2 penalty prevents this by penalizing extreme concentration:
- **||μ||²₂** = Σ μ²_k (Herfindahl index of concentration)
- Minimized by uniform distribution: ||μ||²₂ = 1/N
- Maximized by Dirac spike: ||μ||²₂ = 1

With τ > 0, concentrating μ on a small number of flat steps incurs high ||μ||²₂ cost, forcing the optimizer to spread weight—including over informative transition regions.

### Role of τ

The parameter τ controls the trade-off:

- **τ → 0⁺**: regularization vanishes, μ* minimizes Var_ν(φ) alone (risk of degeneracy)
- **τ → ∞**: L2 penalty dominates, μ* → 1/N (recovers standard IG)
- **Intermediate τ**: balances consistency and spread. Empirically, **τ ∈ [0.005, 0.01]** works well (allows 5–15 steps to carry most weight when N = 50)

## Relationship to IDGI

**IDGI** (Integrated Directional Gradients) uses the closed-form weights `μ_k ∝ |Δf_k|`. This is **not** a solution to our objective, which contains no signal-coverage term. The two approaches are complementary:

- **IDGI**: assigns weight based on *how much the output changes* at each step (|Δf_k|). Closed-form heuristic, no optimization required. Targets signal coverage.

- **μ-Optimized**: assigns weight to minimize *how inconsistently the gradient approximation works* across steps (Var_ν(φ)). Requires iterative optimization but directly targets conservation quality metric Q.

In practice, μ-Optimized weights often resemble IDGI weights (both upweight transition-region steps), but for different reasons:
- IDGI upweights where |Δf_k| is large (heuristic)
- μ-Optimized upweights where φ_k varies most, to equalize fidelities (principled)

**Combination:** You can initialize the optimizer with IDGI weights instead of uniform weights, often accelerating convergence.

## Completeness Axiom

The completeness axiom requires `Σ_i A_i = f(x) − f(x')`.

For arbitrary weights μ ∈ P_N, completeness does not hold exactly (it holds only for μ_k = 1/N as N → ∞). We restore it by final rescaling:

```
A_i ← A_i · [f(x) − f(x')] / Σ_j A_j
```

This is the same rescaling used by IDGI and other weighted IG variants.

**Note:** The quality metric Q is **orthogonal to completeness**—it measures how faithfully the attributions decompose the output change, not whether they sum to the correct total.

## Requirements

```
torch >= 2.0
torchvision
matplotlib         (for --viz flags)
scikit-image       (optional, for SLIC superpixels in --region-insdel)
```

## References

- Sundararajan, M., Taly, A., and Yan, Q. "Axiomatic Attribution for Deep Networks." ICML 2017.
- Sikdar, S., Bhatt, P., and Heese, R. "Integrated Directional Gradients: Feature Interaction Attribution for Neural NLP Models." ACL 2021.
- Petsiuk, V., Das, A., and Saenko, K. "RISE: Randomized Input Sampling for Explanation of Black-box Models." BMVC 2018.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T. "Efficient Projections onto the ℓ1-ball for Learning in High Dimensions." ICML 2008.
