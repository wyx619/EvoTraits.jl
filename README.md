# EvoTraits

**Likelihood-based phylogenetic analysis of discrete and continuous trait evolution in Julia.**

[中文说明](README_zh.md)

EvoTraits is a Julia package for likelihood-based phylogenetic comparative analysis of discrete and continuous traits. It integrates evolutionary model fitting, stochastic character mapping, ancestral-state reconstruction and model comparison within a shared computational framework.

The package supports the study of evolutionary heterogeneity among lineages, coordinated trait evolution, adaptive responses to evolving predictors, temporal changes in evolutionary processes and environmental dependence of evolutionary rates. It combines efficient computation with explicit statistical assumptions and structured, interpretable results.

## Principal Features

**Integrated discrete and continuous inference.** Discrete-state models and SIMMAP histories provide a basis for regime-dependent continuous models. Shared tree representations and consistent result interfaces facilitate analysis across these stages while preserving node and branch correspondence.

**Efficient phylogenetic likelihood computation.** Tree-based pruning recursions support univariate and multivariate likelihoods without constructing a dense covariance matrix across all observations. Supported multi-start fitting routines use Julia threads to optimize candidates in parallel.

**Flexible specification of evolutionary hypotheses.** Models accommodate differences in optima, attraction and diffusion among regimes, dependence of adaptive optima on evolving predictors, and contrasts between early and late evolutionary processes. Environmental histories provide a further basis for testing variation in evolutionary rates. Candidate-specific settings support comparison of alternative parameterizations within a common workflow.

**Structured results for reproducible analysis.** Fit objects retain parameter estimates, likelihoods, information criteria and convergence status. OU mean diagnostics distinguish parameter identifiability from optimizer convergence. Reconstruction tables preserve node identities and temporal coordinates for analyses of ancestral change and temporal patterns.

## Research Applications

| Biological question | EvoTraits capability |
|---|---|
| Does a trait evolve by drift, an early burst or attraction toward an optimum? | BM, EB and OU models |
| Do evolutionary regimes have different optima, attraction strengths or diffusion rates? | OUM, OUMV, OUMA and OUMVA, including multivariate counterparts |
| Does an evolving predictor influence the adaptive optimum of a response? | Multivariate OU-BM adaptation with `fit_mvoubm` |
| Did the evolutionary process change between early and late periods? | Univariate and multivariate time-shift models |
| Does the rate of trait evolution vary with environmental history? | EnvExp and EnvLin, compared with BM1 and EB |
| How do discrete traits change, including hidden rate variation? | Mk and corHMM-style models |
| What histories are consistent with observed tip states? | SIMMAP stochastic mapping |
| What were ancestral traits, and how did they vary through time? | Supported node/branch reconstruction and sliding-window summaries |

## Modeling Capabilities

### Environmental dependence of evolutionary rates

`fit_t_env` relates the diffusion rate of a continuous trait to environmental history through exponential (EnvExp) or linear (EnvLin) rate functions. Comparison with BM1 and EB evaluates support for environmental dependence, constant-rate evolution and an exponential temporal trend. These models describe variation in evolutionary rates; they do not estimate environmental effects on trait means or adaptive optima.

Environmental histories may be represented by piecewise-linear curves or cubic smoothing splines with a specified effective number of degrees of freedom. The `fit_compare` workflow compares all four models without requiring ancestral reconstruction.

Environmental models optionally account for known tip standard errors and estimate an additional common terminal variance. Use `error=missing` to estimate this component when standard errors are unavailable, or supply a vector in tree-tip order or a taxon-keyed dictionary. The default `error=nothing` retains the model without this component.

### Multivariate adaptive-response models

`fit_mvoubm` partitions traits into OU response variables and BM predictor variables. It jointly estimates predictor contributions to response optima, attraction strengths and diffusion parameters. Regime-specific intercepts allow adaptive relationships to vary among mapped evolutionary states, supporting the assessment of trait coadaptation hypotheses.

### Flexible multivariate OU parameterization

Diagonal attraction matrices allow trait-specific attraction rates, whereas supported asymmetric structures accommodate directional coupling among traits. The general positive-real-spectrum parameterization extends asymmetric attraction beyond the two-trait case. A diagonal attraction matrix does not imply independent trait evolution, because diffusion may remain correlated.

Supported multivariate OU fits combine structured starting values with optional random starts and reproducible seeds. Alternative parameterizations can be compared as named model candidates.

### Temporal heterogeneity in evolutionary processes

`fit_shift` and `fit_mvshift` provide 13 time-shift models combining BM, OU and EB components, including the rate-shift model SR. These models support tests of differences between early and late evolutionary processes within a common fitting and comparison framework.

The covariance combines period-specific stochastic contributions at a user-specified temporal boundary. Fitting estimates the evolutionary parameters for that hypothesis; it does not automatically locate an unknown boundary. Optional tree-height normalization and curvature diagnostics support model assessment.

### Consistent fitting and ancestral reconstruction

OU fitting and ancestral reconstruction use consistent root assumptions. Regime optima are estimated by generalized least squares, with diagnostics for their identifiability. Node and branch reconstructions retain their correspondence to the original tree, and time-window summaries support the study of reconstructed trait histories.

Unit and regression tests assess likelihoods, model relationships and reconstruction against independent calculations on small trees and reference cases. Missing observations and numerically challenging covariance structures receive targeted coverage.

## Installation and Examples

Install from the repository in Julia:

```julia
using Pkg
Pkg.add(url="https://github.com/wyx619/EvoTraits.jl")
```

For parallel fitting, start Julia with multiple threads, for example `julia -t 8`. EvoTraits sets BLAS to one thread on loading to avoid oversubscription. This setting is shared by all packages in the Julia process; use `set_engine_blas_threads!(n)` after loading to change it.

### Fit, compare and reconstruct

Assume `traits.csv` contains `taxon` and `lnH` columns matching the tree tips:

```julia
using EvoTraits, CSV, DataFrames

tree = load_tree("tree.tre")
traits = CSV.read("traits.csv", DataFrame)
y = align_traits_to_tree(tree, traits;
    taxon_col=:taxon, trait_cols=[:lnH])

analysis = fit_compare_estim(tree, y, [:BM1, :OU1, :EB])
analysis.aic_table
analysis.best_name
nodes = estim_node_table(tree, analysis.asr)
```

`load_tree` reads directly into the computation-ready tree representation. The explicit `read_tree` → `serialize_tree` route remains available.

### Compare environmental and temporal rate hypotheses

Assume `environment.csv` contains `age` (time before present, in the same units as tree branches) and `temperature` columns. The history must cover the present through the tree's root age and contain more than ten distinct ages for this example:

```julia
history = CSV.read("environment.csv", DataFrame)
environment = smooth_env(history.age, history.temperature; df=10, scale=true)

candidates = [
    ContinuousModelCandidate(name=:BM1, model=:BM1),
    ContinuousModelCandidate(name=:EB, model=:EB),
    ContinuousModelCandidate(name=:EnvExp, model=:EnvExp,
        kwargs=(environment=environment,)),
    ContinuousModelCandidate(name=:EnvLin, model=:EnvLin,
        kwargs=(environment=environment,)),
]
comparison = fit_compare(tree, y, candidates; max_iterations=2000)
comparison.aic_table
comparison.best_fit.converged
```

Use `EnvCurve(history.age, history.temperature; scale=true)` for piecewise-linear interpolation without smoothing; duplicate ages must first be resolved. `smooth_env` accepts repeated ages and requires an explicit df. Choose that value independently of the model comparison. `fit_compare` supports trait vectors and matrices and does not perform ASR by default; use `reconstruct=true` or `fit_compare_estim` for supported models. Environmental models do not support ASR. `fit_compare_multivariate` also defaults to no ASR; set `reconstruct=true` to restore its former behavior.

### Fit regime-dependent multivariate evolution

Here `mapped.simmap` contains mapped evolutionary histories, and `traits.csv` also contains `lnVD`. Use the tree returned with the mapping to preserve branch correspondence:

```julia
mapped = read_simmap("mapped.simmap")
tree, sim = mapped.tree, mapped.simmap
X = align_traits_to_tree(tree, traits;
    taxon_col=:taxon, trait_cols=[:lnH, :lnVD])

fit = fit_mvoumva(tree, X, sim;
    trait_names=["lnH", "lnVD"], A_decomp=:diagonal)
```

OU1, OUM and their regime-specific variants, including their multivariate counterparts, default to the root regime optimum as the root mean (`root_mean_mode=:regime`) and zero root process variance (`root_cov_mode=:fixed`). OU1/OUM and their multivariate counterparts also support a normalized mean design (`:stationary`) and stationary root covariance (`:random`), in any combination. Public OUMV/OUMA/OUMVA APIs and their multivariate counterparts accept only `:regime + :fixed`. A separate root mean is not estimated; inspect `fit.mean_diagnostics` before interpreting regime optima.

To model the first trait as an OU response to the second trait as a BM predictor:

```julia
adaptation = fit_mvoubm(tree, X, sim; kY=1)
```

### Fit a time-shift hypothesis

For a hypothesized boundary 20 time units before the present:

```julia
shift = fit_mvshift(tree, X; model=:OUBMi, age=20.0)
shift.loglik
shift.aic
shift.converged
```

The boundary must lie within the tree's time range. Use `fit_shift` for a single trait. Time-shift results are not currently connected to ancestral reconstruction.

## Model Coverage

| Family | Univariate | Multivariate |
|---|---|---|
| Brownian motion | BM1, BMM | mvBM1, mvBMM |
| Early burst | EB, EBM | mvEB |
| Environmental BM | EnvExp, EnvLin | Not currently implemented |
| Ornstein-Uhlenbeck | OU1, OUM, OUMV, OUMA, OUMVA | mvOU1, mvOUM, mvOUMV, mvOUMA, mvOUMVA |
| Adaptive OU-BM | — | mvOUBM |
| Time-shift | SR, OUBM/OUBMi, BMOU/BMOUi, EBOU/EBOUi, OUEB/OUEBi, EBBM/EBBMi, BMEB/BMEBi | Same 13 models |

Discrete analysis includes Mk, observed/hidden-rate corHMM-style models, ambiguous tip states, custom rate structures and ancestral reconstruction. SIMMAP connects discrete histories to regime-aware continuous models. Information criteria, simulation and cross-language tree references complete the toolkit.

## Scope and Statistical Considerations

Continuous models require bifurcating trees, and many require ultrametric trees. Missing-data support varies by model. Check taxon and SIMMAP alignment, compare AIC using the same observations and consistent assumptions, and assess convergence, boundary estimates and identifiability. Time-shift and environmental models require ultrametric trees and currently lack ASR; time-shift models do not support observation error, whereas environmental models support tip standard errors and a common terminal variance. Environmental models are univariate; smoothing and integration accuracy require particular attention for rapidly varying histories. Estimated trait associations alone do not establish causality.

## Documentation and Validation

Use Julia's help mode for individual APIs, for example `?fit_t_env`, `?smooth_env`, `?fit_mvoubm` or `?fit_mvshift`. The source is organized around discrete models, continuous models, reconstruction and simulation; `test/` contains unit and regression coverage.

