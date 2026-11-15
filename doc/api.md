# API conventions

[中文](api_zh.md) · [Index](README.md) · [Getting started](quickstart.md)

## Core inputs

`CompactTree` stores topology, branch lengths, traversal order and tip labels. Vectors have length `tree.ntips`; matrices have `tree.ntips` rows. Missing multivariate observations are represented by `NaN`, with at least one observed value per row and column. Discrete missing states use `?`, `NA`, `missing` or `NaN`; polymorphic states use `state1&state2`.

## Names and fixed likelihoods

Fit functions use `fit_...`; fixed-parameter likelihood functions use `..._LL`, for example `ou1_LL`, `mvoum_LL` and `corhmm_LL`. Fit results retain `loglik`, `aic`, `nparams`, `success` and `converged` where applicable. `aic` compares fits only when observations and assumptions match.

## Roots and regimes

Root settings are model parameters, not output labels. For OU1/OUM, the default is `root_mean_mode=:regime` and `root_cov_mode=:fixed`; `:stationary` and `:random` are supported in the corresponding APIs. Regime names follow SIMMAP labels when provided. A regime history must use the same tree and branch identity as the traits.

## Threads

Start Julia with `-t N`. EvoTraits sets BLAS to one thread on load. `set_engine_blas_threads!(n)` changes the process-wide BLAS setting. Julia-level threads and BLAS threads are separate resources.

## Compatibility

`read_tree` and `load_tree` have different return contracts. `transition_events_table` is the current table name; `transition_events_DataFrame` remains a compatibility alias. `keep_tip(tree, simmap, tips)` is the unified pruning method; `keep_tip_simmap` remains available.

## Public interface map

| Task | Entry points | Main return contract |
| --- | --- | --- |
| Load a computational tree | `load_tree`, `serialize_tree` | CompactTree |
| Prune a tree / mapped tree | `keep_tip`, `drop_tip` | Tree / tree-and-map pair |
| Fit discrete evolution | `fit_mk`, `fit_corhmm` | Rates, likelihood and diagnostics |
| Reconstruct discrete states | `asr_corhmm` | Requested state reconstruction |
| Sample histories | `simmap_corhmm` | Hidden and observed map collections |
| Fit a scalar trait | `fit_bm1`, `fit_ou1`, `fit_oum*`, `fit_eb*` | Scalar or multi-regime continuous result |
| Fit multiple traits | `fit_mvbm1`, `fit_mvoum*`, `fit_mvoubm` | Matrix-parameter fit results |
| Compare candidates | `fit_compare`, `fit_compare_estim` | Named fits, AIC rows, best fit, optional ASR |
| Reconstruct continuous states | `estim_node`, `estim_node_table` | ASR object / table |
| Environmental / temporal effects | `fit_t_env`, `fit_shift`, `fit_mvshift` | Family-specific fit results |

The asterisks in this table denote function families, not literal callable names. Model-specific guides list exact functions.

## Get signatures and help

```julia
using EvoTraits
methods(fit_oum)
methods(fit_mvoum)
@doc fit_corhmm
@doc fit_t_env
@doc fit_mvshift
```

In the Julia REPL, type `?fit_oum` for function help. Multiple dispatch selects methods from argument types.

## Fixed-parameter likelihood versus fitted result

Functions ending in `_LL` evaluate supplied process parameters, sometimes profiling nuisance mean coefficients according to the specific interface. Their positional parameter order is model-specific. For example, `oum_LL` and `mvoum_LL` do not take interchangeable scalar and matrix arguments. Check the signature before constructing a cross-model validation.

Reusing parameters after changing trait units, branch units, roots or regime order changes their statistical meaning.

## Result-type differences

corHMM stores optimizer information under `fit.fit`; continuous fits generally expose `converged` directly. OU diffusion is `Sigma` in multivariate OU results and `sigma` in multivariate BM results. Univariate multi-regime optima are `theta_regimes`; multivariate optima are `theta`. Shift results use matrix-valued parameters even for a single trait.

Use `propertynames(result)` to inspect a result and the relevant guide to interpret it; generic export code should not assume every result has identical fields.

## Output and persistence

Modeling does not require a DataFrame; aligned numeric arrays are sufficient. Table helpers return DataFrames for inspection and export. The package does not provide an automatic universal result archive or checkpoint format. If an application uses Julia's Serialization, declare and manage that dependency in the application environment and preserve the matching software version. See [fitting/export](fitting.md) for a plain-table workflow.

[Input contracts](inputs.md), [model comparison](workflows.md) and [troubleshooting](troubleshooting.md) provide the task-oriented details.
