# Univariate continuous models

[中文](continuous_zh.md) · [Index](README.md) · [Workflows](workflows.md)

| Model | Fit | Hypothesis |
| --- | --- | --- |
| BM1 | `fit_bm1` | One Brownian diffusion rate |
| BMM | `fit_bmm` | Regime-specific diffusion rates |
| OU1 | `fit_ou1` | One OU optimum |
| OUM | `fit_oum` | Regime-specific optima |
| OUMV / OUMA / OUMVA | `fit_oumv` / `fit_ouma` / `fit_oumva` | Variation in diffusion / attraction / both |
| EB / EBM | `fit_eb` / `fit_ebm` | Time-varying rates / regime-specific EB |

Use a tree-aligned vector; regime models require mapped branch information. Fixed-parameter likelihoods use names such as `ou1_LL` and `oum_LL`.

```julia
fit = fit_ou1(tree, y; max_iterations = 1000)
fit.loglik, fit.aic, fit.converged
asr = estim_node(tree, y, fit)
```

For OU1/OUM, root mean `:regime` and covariance `:fixed` are defaults; `:stationary` mean and `:random` covariance are also supported. Public OUMV/OUMA/OUMVA use `:regime + :fixed`. Root assumptions affect likelihood and reconstruction; inspect `mean_diagnostics` for regime-optimum identifiability. See [Reconstruction](reconstruction.md) and [API conventions](api.md).

## What changes among OU models?

The univariate OU process is written as:

```math
dY = \alpha_r (\theta_r - Y)\,dt + \sqrt{\sigma_r^2}\,dW
```

The active regime `r` comes from the mapped branch segment.

| Model | Attraction α | Diffusion σ² | Optimum θ |
| --- | --- | --- | --- |
| OU1 | Shared | Shared | One |
| OUM | Shared | Shared | Per regime |
| OUMV | Shared | Per regime | Per regime |
| OUMA | Per regime | Shared | Per regime |
| OUMVA | Per regime | Per regime | Per regime |

`sigma2` is a diffusion variance rate, not its square root. `alpha` has inverse-time units; its half-life is `log(2)/alpha`. `theta` has the trait's units and may be negative for log-transformed data.

## Fit a mapped model

```julia
oum = fit_oum(tree, y, sim;
    max_iterations = 1000, rel_tol = 1e-6,
    root_mean_mode = :regime, root_cov_mode = :fixed)
oumva = fit_oumva(tree, y, sim;
    max_iterations = 2000, rel_tol = 1e-6)
oumva.theta_regimes
oumva.alpha_regimes
oumva.sigma2
oumva.regime_names
```

The inputs `tree`, `y` and `sim` must be aligned as described in [SIMMAP](simmap.md). The public univariate fit functions do not expose the multivariate keywords `n_starts` or `polish_iterations`. Internal coarse-search and refinement stages are managed by the implementation. A larger `max_iterations` increases the search budget.

## Root mean and covariance are different choices

| Model | Root mean modes | Root covariance modes |
| --- | --- | --- |
| OU1 / OUM | `:regime`, `:stationary` | `:fixed`, `:random` |
| OUMV / OUMA / OUMVA | `:regime` | `:fixed` |

`:regime` starts the mean at the root regime optimum. `:stationary` uses normalized regime design weights. `:fixed` uses no root process variance, whereas `:random` adds the stationary root variance. No independently estimated `theta0` mode is exposed; removed legacy names such as `:free_theta0` should not be used.

Use the same settings for fixed-parameter likelihood evaluation, fitting and ASR. Different root assumptions can change both logLik and reconstructed ancestral values.

## Reading the results

| Field | Interpretation |
| --- | --- |
| `loglik`, `aic`, `nparams` | Likelihood, AIC and fitted parameter count |
| `success`, `converged` | Result validity and optimizer convergence; inspect both |
| `iterations`, `f_calls` | Optimizer accounting, not interchangeable time measures |
| `theta_regimes` | Regime optima in multi-regime results |
| `alpha`, `alpha_regimes` | Shared or regime-specific attraction |
| `sigma2` | Shared/rate vector as appropriate to the result type |
| `mean_diagnostics` | Rank and conditioning of the mean design |
| `root_mean_mode`, `root_cov_mode` | Actual root assumptions used |

For single-regime results, the optimum is `theta`. `mean_diagnostics` reports `rank`, `ncoefficients`, `singular_values`, `identifiable` and `condition_number`.

Long branches with large `alpha * length` lose memory of ancestral states, so similar reconstructed values are expected. Check them against optima, branch lengths and identifiability diagnostics.

## BM and EB hypotheses

BM1 has one diffusion rate; BMM assigns rates to regimes. EB models diffusion changing exponentially with elapsed time from the root. A negative EB time coefficient means decreasing diffusion through evolutionary time. EBM adds regime structure to an early-burst process.

Compare these models on identical observations. Transforming height to ln(height) changes the response scale and likelihood; do not compare raw-height AIC against log-height AIC as if only the model changed. See [workflows](workflows.md), [fitting diagnostics](fitting.md), and the [mapped OU example](examples/03_continuous.jl).
