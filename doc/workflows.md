# Model-comparison workflows

[中文](workflows_zh.md) · [Index](README.md) · [Continuous models](continuous.md)

`fit_compare` accepts a vector or matrix and returns fits, an AIC table, the selected name and fit. It does not reconstruct by default.

```julia
result = fit_compare(tree, y, [:BM1, :OU1, :EB])
result = fit_compare(tree, X, [:mvBM1, :mvOU1]; reconstruct = true)
```

Use `ContinuousModelCandidate` when candidates need different settings:

```julia
candidates = [
    ContinuousModelCandidate(name=:plain, model=:BM1),
    ContinuousModelCandidate(name=:env, model=:EnvLin,
        kwargs=(environment=environment, quadrature_order=32)),
]
comparison = fit_compare(tree, y, candidates)
```

`fit_compare_estim` explicitly requests ASR for univariate or multivariate inputs. `fit_compare_multivariate` remains available as a compatibility name and defaults to no ASR; pass `reconstruct=true` to request it. Environmental models cannot enter an ASR workflow.

AIC ranks finite successful fits, but a finite likelihood does not prove numerical convergence. Report the model, observations, parameter count, logLik, AIC, convergence status and diagnostics together.

## A complete regime-aware comparison

With aligned `tree`, `y` and `sim`:

```julia
comparison = fit_compare(tree, y, [:BM1, :BMM, :OUM];
    edge_segments = sim.edge_segments,
    max_iterations = 1000, rel_tol = 1e-6)
for row in comparison.aic_table
    println(row)
end
for (name, fit) in comparison.fits
    println(name, " success=", fit.success,
            " converged=", fit.converged)
end
selected = comparison.best_fit
```

Regime-dependent candidates need mapping context even when other candidates do not. The workflow can derive mapped durations from branch segments. Missing required context is an input error; it is not silently replaced by one regime.

For vector inputs use univariate symbols such as `:BM1, :BMM, :OU1, :OUM, :EB`. For matrix inputs use `:mvBM1, :mvBMM, :mvOU1, :mvOUM, :mvOUMVA, :mvOUBM` and other supported multivariate symbols. Matrix dispatch does not rename every symbol automatically.

## Candidate-specific settings

```julia
candidates = [
    ContinuousModelCandidate(name = :bm, model = :BM1),
    ContinuousModelCandidate(name = :oum, model = :OUM,
        kwargs = (max_iterations = 2000,
                  root_mean_mode = :regime,
                  root_cov_mode = :fixed)),
]
comparison = fit_compare(tree, y, candidates;
    edge_segments = sim.edge_segments)
```

Candidate keywords override common workflow settings. Names identify outputs and should be unique. Keep a copy of the candidate definitions alongside exported tables.

## Returned objects and selection

| Member | Contents |
| --- | --- |
| `fits` | Dictionary of named fit results |
| `aic_table` | Sorted vector of named tuples |
| `best_name`, `best_fit` | Selected candidate and result |
| `asr` | `nothing` unless reconstruction is requested |

AIC rows contain `name, model, aic, delta_aic, loglik, nparams, success`. They do not contain every convergence diagnostic. Selection uses successful finite-AIC fits, so inspect the individual results for convergence before making a scientific claim.

`aic_table(:first => fit1, :second => fit2)` also compares already fitted results. AIC is `2k - 2logLik`; smaller is preferred within a valid candidate set. AICc additionally needs an observation-count convention: supply `nobs` when it cannot be inferred. In multivariate results, observed trait cells may be counted rather than species. This is an implementation convention, not proof that trait cells are independent biological replicates.

## What makes an AIC comparison valid?

Use the same response, transformation, retained taxa and missingness pattern. Root assumptions and parameter constraints are part of the candidate definition and must be reported. Different discrete encodings have different observation spaces; a smaller discrete AIC for two states does not establish that two biological categories are superior to three.

Continuous models fitted to different SIMMAPs are conditional on different histories. Repeating maps estimates sensitivity to history sampling. Neither their mean AIC nor the minimum over maps is an integrated model likelihood. AIC support for a regime model also does not by itself establish a causal effect of the discrete state.

## Reconstruction is explicit

`fit_compare_estim` or `reconstruct=true` requests reconstruction. A comparison containing environmental candidates cannot request ASR, even if another candidate might win. To reconstruct a supported selected model, use its fitted result and the matching tree/map in a separate [ASR call](reconstruction.md).

For export, `DataFrame(comparison.aic_table)` creates a table; CSV writing can then be performed by `CSV.write`. See [reproducible fitting](fitting.md).
