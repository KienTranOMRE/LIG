# Signal-Harvesting IG - Quick Cheat Sheet

## The Core Formula (Eq. 20)

```
min_{γ,μ}  Var_ν(φ) - λ Σ μ_k |d_k| + (τ/2) ||μ||²
           ────┬────   ─────┬─────   ────┬────
           Distortion  Signal      Admissibility
           (Snell)     Harvesting  (anti-spike)
```

## Key Result

**μ*_k ∝ |d_k| ≈ |Δf_k|**  ← This is EXACTLY IDGI!

## Quick API Reference

### Zero Extra Cost (RECOMMENDED)
```python
from signal_lam import mu_optimized_ig

result = mu_optimized_ig(model, x, baseline, N=50, lam=1.0, tau=0.01)
# Cost: Same as Standard IG + O(N) arithmetic
# Returns: AttributionResult with .attributions, .Q, .Var_nu, .CV2
```

### Full Optimization (3-5× cost)
```python
from signal_lam import joint_star_ig

result = joint_star_ig(model, x, baseline, N=50, lam=1.0, tau=0.01,
                       n_alternating=2, mu_iter=300, path_iter=10)
# Alternates: μ optimization ↔ path optimization
```

### Run All Methods
```python
from signal_lam import run_all_methods

methods = run_all_methods(model, x, baseline, N=50, lam=1.0, tau=0.01)
# Returns: [IG, IDGI, Guided IG, μ-Opt*, Joint(λ=0), Joint*]
```

## Hyperparameters

| Param | Range | Default | Effect |
|-------|-------|---------|--------|
| **λ** | 0.5-2.0 | 1.0 | Signal harvesting strength (0=pure LAM, ∞=pure IDGI) |
| **τ** | 0.005-0.01 | 0.01 | L2 penalty (smaller=more concentration) |
| **N** | 50-100 | 50 | Number of interpolation steps |

## All Methods as Special Cases

```
Standard IG:    λ=0,  τ→∞   →  uniform μ,        straight line
IDGI:           λ>0,  τ→0   →  μ ∝ |Δf|,        straight line
Guided IG:      λ>0,  τ=—   →  uniform μ,        heuristic path
μ-Optimized*:   λ>0,  τ>0   →  optimal μ,        straight line ← PRACTICAL
Joint*:         λ>0,  τ>0   →  optimal μ,        optimal path  ← COMPLETE
```

## Metrics

```python
Q = 1/(1 + CV²)  ∈ [0,1]     # Higher is better (1 = perfect)
Var_ν(φ) ≥ 0                 # Lower is better (0 = perfect)
Obj = Var_ν - λ·signal + τ·L2  # Lower is better
```

## CLI Usage

```bash
python signal_lam.py --steps 50 --lam 1.0 --tau 0.01 \
    --guided-init --viz --viz-path results.png \
    --insdel --json metrics.json
```

## The Innovation

**Problem:** Original LAM min Var_ν(φ) has degenerate basin (μ on flat regions)

**Solution:** Add -λ Σ μ_k |d_k| to force μ toward transition region

**Result:** IDGI and Guided IG are now special cases of one unified objective!

## Physical Analogy

| IG Component | Wave Optics | Physical Meaning |
|--------------|-------------|------------------|
| Var_ν(φ) | Wavefront aberration | Minimize distortion (Snell's law) |
| -λΣμ\|d\| | Signal intensity | Concentrate on bright region |
| Path γ | Light ray | Trajectory through input space |
| Measure μ | Detector window | Where to integrate |

## Code Organization (signal_lam.py)

```
§1  compute_signal_harvesting_objective()          ← Evaluate Eq. 20
§2  mu_star_closed_form()                         ← Closed-form μ* ∝ |d_k|
§3  optimize_mu_signal_harvesting()               ← Optimize μ (Phase 1)
§4  mu_optimized_ig()                             ← MAIN API (zero extra cost)
§5  _signal_harvesting_path_obj()                 ← Path objective
§6  optimize_path_signal_harvesting()             ← Optimize path (Phase 2)
§7  joint_star_ig()                               ← FULL API (alternating min)
§8  run_all_methods()                             ← Convenience runner
§9  run_experiment()                              ← CLI entry point
```

## Decision Tree

```
Do you need attributions?
├─ Yes, and I want best quality regardless of cost
│  └─ Use joint_star_ig() with λ=1.0, τ=0.01, n_alternating=2
│
├─ Yes, but I need it fast (production)
│  └─ Use mu_optimized_ig() with λ=1.0, τ=0.01  ← RECOMMENDED
│
├─ I want to compare methods
│  └─ Use run_all_methods() and check Q metric
│
└─ I just need standard IDGI
   └─ Use mu_optimized_ig() with λ→large, τ→0.01 (approximates IDGI)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low Q but high signal harvested | Decrease λ (favor conservation) |
| Noisy attributions | Increase λ (favor signal harvesting) |
| μ too concentrated (spike) | Increase τ |
| μ too uniform | Decrease τ |
| Slow Joint* | Reduce G, path_iter, or n_alternating |

## Key Files

- `signal_lam.py` - Main implementation
- `lam.py` - Path utilities, standard IG variants
- `utilss.py` - Metrics (Var_ν, CV², Q), AttributionResult
- `SIGNAL_HARVESTING_REFERENCE.md` - Full documentation

## One-Liner Examples

```python
# Simplest usage
result = mu_optimized_ig(model, x, baseline)

# Custom parameters
result = mu_optimized_ig(model, x, baseline, N=100, lam=2.0, tau=0.005)

# Full optimization
result = joint_star_ig(model, x, baseline, n_alternating=3)

# Warm-start Joint* from Guided IG
from lam import guided_ig
gig = guided_ig(model, x, baseline, N=50)
result = joint_star_ig(model, x, baseline, init_path=gig.gamma_pts)
```

## Remember

1. **μ-Optimized*** is the **practical contribution** - zero extra cost, high quality
2. **Joint*** is the **theoretical completion** - full optimization of Eq. 20
3. **IDGI is now derived**, not heuristic: μ*_k ∝ |d_k|
4. **L2 penalty is crucial**: linear condition → recovers IDGI exactly
5. **λ interpolates**: 0 (pure LAM) ↔ ∞ (pure IDGI)
