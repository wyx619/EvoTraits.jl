# Fitting and diagnostics

[中文](fitting_zh.md) · [Index](README.md) · [Model comparison](workflows.md)

## A defensible fitting sequence

1. Validate tip names, trait order, missingness, time units and mapped branch lengths.
2. Fit a simple baseline, usually BM1, on the final analysis dataset.
3. Fit the biological candidates with explicit root and matrix constraints.
4. Inspect numerical validity, optimizer stopping state, boundaries and identifiability.
5. Compare eligible fits on the same observations, then reconstruct supported selected models.
6. Record settings and results before scaling to more maps or trees.

The [small runnable examples](examples/README.md) illustrate interfaces. Their small trees and budgets are not recommendations for a final empirical analysis.

## Reading status correctly

`success` is a model-specific validity flag. `converged` concerns optimizer termination; in corHMM it is stored under `fit.fit.converged`. A finite logLik is useful evidence of a valid evaluation but does not establish optimizer convergence. Conversely, convergence to a local optimum does not establish global optimality or identifiability.

Fixed-parameter likelihood evaluations are not optimization runs. Do not demand a convergence flag from them or report them as independently fitted models. Compare their logLik at identical parameters when diagnosing implementation differences.

## Budgets are not universal iteration counts

NLopt-based budgets can limit function evaluations; L-BFGS reports its own iteration and call counts. A value of 1000 therefore does not mean the same work in every model. Record `f_calls`, `iterations`, stopping information where available, and elapsed time. Do not compare these counters as if they were identical.

If a fit reaches a budget, increase the relevant budget, retry with alternative initialization where supported, and compare final likelihoods. Reducing the stopping tolerance alone is not a guarantee of a better solution.

## Threads and memory

Start Julia with `julia --project -t 8 analysis.jl`. Check `Threads.nthreads()`. EvoTraits initializes BLAS at one thread; `set_engine_blas_threads!(n)` changes that process-wide setting after loading.

corHMM additionally requires `Ntrials` and `Nthreads`; launching an eight-thread process alone does not request eight trials. Some continuous models parallelize candidate fits internally without a user thread keyword. Do not assume all likelihood recursions or all fitting functions are threaded.

Independent candidates require private workspaces. More threads can increase peak memory, and launching outer parallel jobs around internally parallel fits can oversubscribe resources. Begin with one map/tree, measure, then choose the batch policy.

## Minimal result export

With a univariate continuous `fit`:

```julia
using CSV, DataFrames
mkpath("results")
summary = DataFrame(
    model = [string(fit.model)],
    success = [fit.success],
    converged = [fit.converged],
    loglik = [fit.loglik],
    aic = [fit.aic],
    nparams = [fit.nparams],
)
CSV.write("results/fit_summary.csv", summary)
```

Also export parameter values, regime and trait labels, root assumptions, diagnostics and the exact analysis inputs. A CSV summary is not a complete reproducibility record. Use Julia's `versioninfo()` and preserve the analysis environment's Project/Manifest files. The package does not require serializing fit objects to perform an analysis.

## Interpretation checklist

Check whether parameter scales match the time and trait units, whether attraction half-lives are estimable, whether regimes have enough mapped exposure, and whether covariance estimates are close to singular. For OU mean designs, inspect `mean_diagnostics` separately from optimizer status.

Repeat key fits where alternative starts are supported. Report unresolved numerical or identification problems rather than selecting the smallest AIC without qualification. See [troubleshooting](troubleshooting.md).
