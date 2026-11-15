# Runnable examples

[中文](README_zh.md) · [Documentation index](../README.md)

These scripts create small synthetic inputs and run directly from the repository root after the project dependencies have been installed:

```text
julia --project -t 8 doc/examples/01_inputs.jl
julia --project -t 8 doc/examples/02_discrete_simmap.jl
julia --project -t 8 doc/examples/03_continuous.jl
julia --project -t 8 doc/examples/04_multivariate.jl
julia --project -t 8 doc/examples/05_environment_shifts.jl
```

| Script | Demonstrates | Checks |
| --- | --- | --- |
| [01_inputs.jl](01_inputs.jl) | Taxon alignment, Newick I/O, paired pruning | Ordering, retained tips, valid map |
| [02_discrete_simmap.jl](02_discrete_simmap.jl) | ARD, missing/polymorphic observations, marginal root posterior, SIMMAP | Finite fit, posterior sum, multi-history round trip |
| [03_continuous.jl](03_continuous.jl) | BM1/BMM/OUM comparison, OUM ASR, sliding windows | No implicit ASR, node IDs, baseline row |
| [04_multivariate.jl](04_multivariate.jl) | Diagonal-A mvOUM and ASR | Parameter dimensions, named trait output |
| [05_environment_shifts.jl](05_environment_shifts.jl) | Smoothed environment, EnvExp, precomputed SR | Finite likelihoods and valid fit outputs |

[common.jl](common.jl) supplies a reproducible 24-tip tree, two continuous traits and a synthetic two-regime history. That history is constructed for interface demonstration; only the discrete example samples posterior SIMMAPs.

Output is printed to the terminal. Tree and map round-trip files are written to temporary directories rather than the source-data folder. All scripts are independent: running the third does not require files produced by the second.

Small budgets keep examples practical. Inspect convergence and mean-design diagnostics even when the basic checks pass.
