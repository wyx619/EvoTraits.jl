# Environment and time-shift models

[中文](environment_shifts_zh.md) · [Index](README.md) · [Workflows](workflows.md)

## Environmental rates

`fit_t_env` is a univariate model in which the diffusion rate varies with an environmental curve. `EnvExp` and `EnvLin` describe rate dependence, not a causal effect on the trait mean or OU optimum.

```julia
environment = EnvCurve([0.0, 10.0], [15.0, 20.0])
fit = fit_t_env(tree, y, environment; model=:EnvExp,
    quadrature_order=16, max_iterations=2000)
```

Use `smooth_env(history.age, history.value; df=10, scale=true)` for an explicit smoothed curve. The history must cover the tree time range. Inspect quadrature order and breakpoints for sharp pulses. Compare with BM1 and EB using `fit_compare`; environmental models have no ASR interface.

If `E(t)` is the supplied environmental value, EnvExp uses rate `sigma2 * exp(beta * E(t))`; EnvLin uses `sigma2 * (1 - E(t)) + beta * E(t)`, subject to positive rates. Thus EnvLin `beta` is the rate at environment 1, not the additive slope. Scaling the environment changes the interpretation of both parameters. In the EnvExp case `beta=0` reduces to BM; in EnvLin `beta=sigma2` does so.

Known tip standard errors can be passed to `fit_t_env`; `error=missing` estimates a common nuisance standard deviation.

## Time shifts

`fit_shift` and `fit_mvshift` test a specified boundary between early and late processes. Models include SR and combinations of BM, OU and EB.

```julia
shift = fit_mvshift(tree, X; model=:OUBMi, age=20.0)
shift.loglik, shift.aic, shift.converged
```

The boundary is a hypothesis supplied by the analyst; the function does not search for it. Time units must match tree branch lengths. Shift models currently do not provide ASR.

## Preparing an environmental history

Environmental time is **age before present**, not elapsed time from the root. For a 100 Ma tree the curve must cover the whole interval 0–100 Ma. No extrapolation is supplied to fill missing ancient history.

```julia
ages = collect(0.0:1.0:100.0)
temperature = 18 .+ 3 .* sin.(ages ./ 12)
linear = EnvCurve(ages, temperature)  # Piecewise linear
smooth = smooth_env(ages, temperature; df = 10, scale = false)
smooth(25.0)
```

`EnvCurve` requires at least two distinct finite nonnegative ages and finite values; it sorts the ages. `smooth_env` requires at least six unique ages, positive finite weights, and `2 < df < number_of_unique_ages`. The effective degrees of freedom control flexibility, not the number of input observations or interpolation points. Both interfaces default to `scale=false`. With `scale=true`, the environmental range is rescaled to [0,1]; constant histories cannot be scaled this way.

Pick smoothing independently of the outcome when possible and examine sensitivity. Trying many smoothing choices and selecting the smallest model AIC does not automatically account for the smoothing search.

## Rate coefficients and positivity

```text
EnvExp: v(t) = sigma2 * exp(beta * E(t))
EnvLin: v(t) = sigma2 * (1 - E(t)) + beta * E(t)
```

In EnvExp, a positive beta increases diffusion with E and a negative beta decreases it. In EnvLin, sigma2 and beta are the rates at E=0 and E=1. If E is raw temperature, those are 0°C and 1°C; scaling to [0,1] changes that interpretation to environmental endpoints. EnvLin must remain positive over the used environmental history.

These models explain changes in **diffusion rate**, not directional changes in the trait mean. They do not fit the derivative of temperature unless the derivative itself is deliberately supplied as E.

## Numerical integration and observation errors

`quadrature_order=16` is a default, not a guarantee for every curve. Compare 8, 16, 32 and 64 points for sharp pulses or strong exponential effects, and use `quadrature_breakpoints` to mark relevant ages. Assess both branch variances and fixed-parameter logLik before attributing fit differences to biology.

`error=nothing` disables the common observation-error term. `error=missing` or `NaN` estimates a common nuisance standard deviation. Known standard errors may be supplied as a vector, taxon dictionary or an appropriate DataFrame column; these are standard deviations, not variances. Supplying known errors also enables estimation of the nuisance term. Terminal variances include known SE² plus nuisance². Read `fit.nuisance` and `fit.nuisance_variance` on their respective scales.

Retain `environment`, `edge_variances`, `quadrature_order`, `quadrature_breakpoints`, parameters and convergence diagnostics with results. See [the executable environmental example](examples/05_environment_shifts.jl).

## Shift models in detail

| Family | Early process → late process |
| --- | --- |
| SR | Brownian rate change |
| OUBM / BMOU | OU → BM / BM → OU |
| EBOU / OUEB | EB → OU / OU → EB |
| EBBM / BMEB | EB → BM / BM → EB |

Except SR, each family also has an `i` version (for example `:OUBMi`) allowing separate diffusion covariance parameters for the two periods. Supply exactly one boundary representation: `age`, a compatible two-period `mapped_edge`, or precomputed input from `shift_precalc`. A supplied age must lie strictly between zero and tree height. Reusing precomputation after changing the tree is invalid.

```julia
height = maximum(tree.dist_from_root)
pre = shift_precalc(tree; age = height / 2, ntraits = size(X, 2))
shift = fit_mvshift(tree, X; model = :OUBMi, precalc = pre,
    A_decomp = :diagonal, Sigma_decomp = :diagonal,
    max_iterations = 1000)
```

`fit_shift` accepts a vector and returns the same matrix-based result family with one trait. `Sigma_decomp=:cholesky` gives full positive-definite diffusion; `:diagonal` gives diagonal diffusion; `:equaldiagonal` gives equal diagonal variances with zero off-diagonal entries; `:equal` gives equal diagonal variances while permitting correlation. Equal does not mean every entry is equal.

`scale_height=false` preserves original time units. With `scale_height=true`, the fit uses a normalized height; examine `source_height`, `height`, `time_scale` and `age` before interpreting rates. Do not treat normalized-time parameters as original-time rates without conversion.

EB `beta_bounds` default to (-2,2). Inputs outside the allowed mapping can map to zero effect; inspect `input_beta`, `beta` and diagnostics rather than interpreting a flat parameter direction as strong evidence for BM.

`diagnose_shift(tree, traits, fit)` computes numerical curvature diagnostics without refitting. It can be expensive because finite-difference Hessians require many likelihood calls. These diagnostics do not supply a general parameter confidence-interval or global-optimum guarantee. Shift models here assume ultrametric trees, do not accept observation-error modeling, and have no ASR interface.
