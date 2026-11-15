# EvoTraits.jl

[简体中文](README_zh.md)

**A Julia framework for studying how traits evolve across lineages, traits, and time.**

EvoTraits brings discrete and continuous phylogenetic comparative methods into a connected, likelihood-based workflow. From fitting a discrete character and sampling its evolutionary history to testing regime-dependent trait evolution and reconstructing ancestors, it supports analyses that connect evolutionary hypotheses with interpretable model results.

## Highlights

- **From discrete histories to continuous evolution.** Fit Mk and corHMM models, sample SIMMAP histories, and use mapped regimes in continuous models while retaining tree and branch correspondence.
- **Model evolutionary heterogeneity.** Compare BM, EB and OU hypotheses, including regime-specific optima, attraction strengths and diffusion rates in univariate and multivariate analyses.
- **Explore multivariate adaptation.** Study joint trait evolution and use OU-BM models to assess how evolving predictors relate to adaptive optima.
- **Test environmental and temporal hypotheses.** Fit environmental rate models and compare evolutionary processes across a specified time boundary.
- **Keep inference traceable.** Examine likelihoods, AIC, convergence and identifiability diagnostics, then reconstruct ancestral nodes and branches for supported models.
- **Build on Julia's computational ecosystem.** Tree-based likelihood recursions and threaded fitting routines support reproducible analyses without requiring a dense covariance matrix across all observations.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/wyx619/EvoTraits.jl")
```

## Quick start

```julia
using EvoTraits

tree = mktempdir() do dir
    path = joinpath(dir, "tree.nwk")
    write(path, "((A:1,B:1):1,(C:1,D:1):1);")
    load_tree(path)
end
values = Dict("A" => 1.2, "B" => 1.5, "C" => 1.1, "D" => 1.8)
y = [values[label] for label in tree.tip_labels]

fit = fit_ou1(tree, y)
fit.loglik
fit.aic

asr = estim_node(tree, y, fit)
table = estim_node_table(tree, asr)
```

For a complete example with named taxa, see [Getting started](doc/quickstart.md). The model interfaces also accept vectors, matrices and dictionaries; `DataFrames.jl` provides tabular inputs and outputs, and `CSV.jl` can read and write CSV files.

## Documentation

| Topic | Guide |
| --- | --- |
| First analysis | [Getting started](doc/quickstart.md) |
| Documentation index | [Guides](doc/README.md) |
| Data preparation and missing values | [Inputs](doc/inputs.md) |
| Fitting, diagnostics and export | [Fitting](doc/fitting.md) |
| Trees and tip alignment | [Trees](doc/trees.md) |
| Discrete traits | [Discrete models](doc/discrete.md) |
| Stochastic character maps | [SIMMAP](doc/simmap.md) |
| Univariate continuous traits | [Continuous models](doc/continuous.md) |
| Multivariate traits and regimes | [Multivariate models](doc/multivariate.md) |
| Model comparison | [Workflows](doc/workflows.md) |
| Ancestral reconstruction | [Reconstruction](doc/reconstruction.md) |
| Simulating trees and traits | [Simulation](doc/simulation.md) |
| Environmental and time-shift models | [Environment and shifts](doc/environment_shifts.md) |
| Inputs, results and conventions | [API conventions](doc/api.md) |

Every guide has a Chinese counterpart. The [Chinese documentation index](doc/README_zh.md) provides the same reading paths.

Self-contained [example scripts](doc/examples/README.md) cover the principal workflows. See [Troubleshooting](doc/troubleshooting.md) for alignment, convergence and reconstruction questions.

## Computational notes

Start Julia with multiple threads for multi-start fitting, for example `julia -t 8`. EvoTraits sets BLAS to one thread when loaded to avoid oversubscription; use `set_engine_blas_threads!(n)` after loading if a different process-wide setting is required. Inspect `success`, `converged`, boundary estimates, and identifiability diagnostics before interpreting a fit. Compare models using the same tree, observations, and missing-data rules.

## License

EvoTraits.jl is distributed under the GNU General Public License, version 3 (GPL-3.0). See [LICENSE](LICENSE).
