# Multivariate models

[中文](multivariate_zh.md) · [Index](README.md) · [SIMMAP](simmap.md)

Multivariate trait matrices have one row per tree tip and one column per trait. Available fits include `fit_mvbm1`, `fit_mvbmm`, `fit_mveb`, `fit_mvou1`, `fit_mvoum`, `fit_mvoumv`, `fit_mvouma`, `fit_mvoumva`, and `fit_mvoubm`.

```julia
# tree and sim must come from the same SIMMAP
X = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:lnH, :lnVD])
fit = fit_mvoumva(tree, X, sim;
    trait_names = ["lnH", "lnVD"], A_decomp = :diagonal)
fit.loglik, fit.aic, fit.converged
```

OUM models use mapped regimes. OUMV allows regime-specific diffusion; OUMA allows regime-specific attraction; OUMVA allows both. A diagonal attraction matrix does **not** imply independent traits when diffusion is correlated. OU1/OUM root choices are described in [Continuous models](continuous.md); multivariate OUMV/OUMA/OUMVA support regime mean and fixed root covariance.

`fit_mvoubm` models OU responses whose optima depend on BM predictors; `kY` specifies how many leading columns are responses. Parameter identifiability, convergence and input missingness require careful assessment. For candidate comparisons see [Workflows](workflows.md).

## Input dimensions and partial observations

Use an `ntips × p` matrix with traits in columns. The same ordered columns must be used in the trait names, parameter matrices and downstream ASR. Partial observations are represented by `NaN`; every retained tip and every trait column must contain at least one observation. Infinite values are invalid.

```julia
X = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:lnH, :lnVD])
@assert size(X, 1) == tree.ntips
@assert size(X, 2) == 2
```

The multivariate DataFrame fit wrappers select numeric columns after the first, taxon column. For precise column control, perform explicit alignment and pass a matrix. Do not assume every fit wrapper accepts `trait_cols`.

## A, Sigma and theta

The OU mean follows `dY = A * (theta_r - Y) dt + noise`. `A` describes attraction and cross-trait mean coupling; `Sigma` is the diffusion covariance rate.

| Model | A | Sigma | theta |
| --- | --- | --- | --- |
| mvOU1 | Shared | Shared | One vector |
| mvOUM | Shared | Shared | Regime-specific |
| mvOUMV | Shared | Regime-specific | Regime-specific |
| mvOUMA | Regime-specific | Shared | Regime-specific |
| mvOUMVA | Regime-specific | Regime-specific | Regime-specific |

In OU results, `A` and `Sigma` are three-dimensional block arrays. Shared parameters have one block; regime-specific parameters have a block per regime. `theta` has **regimes in rows and traits in columns**. BM results use lowercase `sigma`, so do not blindly reuse the OU field names.

```julia
fit.A[:, :, 1]
fit.Sigma[:, :, 1]
fit.theta
fit.trait_names
fit.regime_names
```

## Attraction-matrix parameterization

| `A_decomp` | Interpretation |
| --- | --- |
| `:cholesky` | Symmetric positive-definite attraction |
| `:diagonal` | Positive diagonal attraction rates |
| `:schur` | Supported two-trait Schur parameterization |
| `:decomposable_positive` | General decomposable attraction with positive real spectrum |

A diagonal `A` removes mean coupling but leaves correlated diffusion when `Sigma` has off-diagonal entries. It therefore does not by itself establish independent trait evolution. Changing units changes matrix entries; standardization may improve numerical conditioning but must be documented when interpreting parameters.

## Optimization options differ across functions

For `fit_mvoum`, commonly used controls include `optimization=:SBPLX_L_BFGS`, `max_iterations=300`, `polish_iterations=100`, `rel_tol=1e-5`, `n_starts=0` and `random_seed=nothing`. Here `n_starts` adds random candidates to structured initial candidates; zero does not mean no initialization strategy. Not all OU family functions expose all these controls. For example, do not add `polish_iterations` to a function whose signature omits it.

OU1/OUM allow the root combinations in the [univariate guide](continuous.md). OUMV/OUMA/OUMVA expose `:regime + :fixed`. Root choices, parameterization and trait scales must be held constant in a controlled comparison.

## OU responses with BM predictors: mvOUBM

`fit_mvoubm` separates the leading `kY` columns into OU responses Y; the remaining columns are BM predictors X. Response optima depend on predictors through `B`, with regime-specific intercepts. The predictor order is therefore part of the model definition.

```julia
# First column: response; second column: BM predictor.
coupled = fit_mvoubm(tree, X, sim;
    kY = 1, A_decomp = :diagonal, max_iterations = 300,
    trait_names = ["response", "predictor"])
coupled.B
coupled.vX0
coupled.theta
```

This function has its own root treatment and parameterization; it does not inherit the ordinary OU `root_mean_mode` / `root_cov_mode` controls. Its attraction options also include `:decomposable_real`, `:upper_triangular` and `:lower_triangular`; do not transfer these options to ordinary mvOUM without checking support. Likewise, ordinary mvOUM restart keywords should not be assumed to exist for mvOUBM.

See [the executable multivariate example](examples/04_multivariate.jl) and [fitting diagnostics](fitting.md).
