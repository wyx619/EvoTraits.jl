# EvoTraits

**Fit and analyze trait evolution models in Julia with high efficiency.**

[中文说明](README_zh.md)

EvoTraits is a Julia package for likelihood-based phylogenetic comparative analysis of discrete and continuous traits. It integrates evolutionary model fitting, stochastic character mapping, ancestral-state reconstruction and model comparison within a shared computational framework.

The package supports analyses of evolutionary heterogeneity among lineages, multivariate trait dynamics, adaptive responses to evolving predictors, temporal changes in evolutionary processes and environmental dependence of evolutionary rates. Its principal emphasis is computational efficiency combined with explicit statistical assumptions and structured, interpretable results.

## Principal Features

**Integrated discrete and continuous inference.** Discrete-state models and SIMMAP histories provide a basis for regime-dependent continuous models. Shared tree representations and consistent result interfaces facilitate analysis across these stages while preserving node and branch correspondence.

**Efficient phylogenetic likelihood computation.** Univariate and multivariate models share tree-based computational infrastructure. Pruning recursions evaluate likelihoods without constructing an observation-sized dense covariance matrix. Supported multi-start fitting paths use Julia threads for parallel candidate optimization.

**Flexible specification of evolutionary hypotheses.** Models accommodate differences in optima, attraction and diffusion among regimes, dependence of adaptive optima on evolving predictors, and contrasts between early and late evolutionary processes. Environmental histories provide a further basis for testing variation in evolutionary rates. Candidate-specific settings support comparison of alternative parameterizations within a common workflow.

**Structured results for reproducible analysis.** Fit objects retain parameter estimates, likelihoods, information criteria and convergence status. Reconstruction tables preserve node identities and temporal coordinates, supporting subsequent analyses of ancestral change and temporal patterns.

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

## Recent Developments

### Environmental dependence of evolutionary rates

`fit_t_env` relates the diffusion rate of a continuous trait to an environmental history through exponential (EnvExp) or linear (EnvLin) rate functions. Comparing these models with BM1 and EB distinguishes an environmental rate hypothesis from constant-rate evolution or a directional temporal trend. The formulation follows RPANDA's environmental BM models and estimates variation in evolutionary rates, rather than environmental effects on trait means or adaptive optima.

Environmental histories may be represented by piecewise-linear curves or explicitly smoothed with fixed-effective-df cubic splines. The `fit_compare` workflow compares all four models without requiring ancestral reconstruction. Shared integration intervals avoid redundant storage and computation; EnvLin uses analytical integration for these known curve types.

In a local validation using a complete InfTemp history, an 87-tip tree and simulated traits, four-model fitting was approximately 12.5 times faster after cache optimization, while the retained 16-point EnvExp cache decreased by 93.7%. These are single-case measurements excluding compilation and smoothing, not general performance guarantees. Independent integration and likelihood checks preserved numerical agreement and model ranking.

### Multivariate adaptive-response models

`fit_mvoubm` partitions traits into OU response variables and BM predictor variables. The model estimates predictor contributions to response optima together with attraction and diffusion parameters. Regime-specific intercepts allow the adaptive relationship to be evaluated in the context of mapped evolutionary histories. This formulation supports model-based assessment of adaptive-response hypotheses; estimated relationships do not, by themselves, establish causality.

### Flexible multivariate OU parameterization

Diagonal attraction matrices allow trait-specific attraction rates, whereas supported asymmetric structures accommodate directional coupling among traits. The general positive-real-spectrum parameterization extends asymmetric attraction beyond the two-trait case. A diagonal attraction matrix does not imply independent trait evolution, because diffusion may remain correlated.

Supported multivariate OU fits combine structured starting values with optional random multi-start optimization and reproducible seeds. Named candidates may specify distinct fitting settings, facilitating explicit comparison of alternative model structures.

### Temporal heterogeneity in evolutionary processes

`fit_shift` and `fit_mvshift` add 13 time-shift models combining BM, OU and EB components, including rate-shift SR. They share fixed-parameter likelihood APIs, reusable time mappings, optional tree-height normalization and explicit curvature diagnostics.

These models follow the reference's two-period covariance construction, in which period-specific stochastic contributions are added. They are distinct from a generic continuously switched stochastic differential equation. The temporal boundary is specified by the user; fitting estimates process parameters rather than detecting an unknown boundary automatically.

### Numerical stability and validation

Recent updates corrected fixed-theta evaluation and a parameter-dependent normalization term in profiled-root OU likelihoods. BMM/SR use covariance or square-root recursions for complete and partially missing observations to avoid catastrophic cancellation near singular diffusion matrices. Regression tests and independent small-tree covariance calculations assess numerical agreement under these conditions. Such validation does not establish global optimality of fitted solutions.

## Installation and Examples

Install from the repository in Julia:

```julia
using Pkg
Pkg.add(url="https://github.com/wyx619/EvoTraits.jl")
```

For parallel fitting, start Julia with multiple threads, for example `julia -t 8`. Limit BLAS threads when combining it with Julia-level parallelism:

```julia
using EvoTraits
set_engine_blas_threads!(1)
```

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

Use `EnvCurve(history.age, history.temperature; scale=true)` for piecewise-linear interpolation without smoothing; duplicate ages must first be resolved. `smooth_env` accepts repeated ages and requires an explicit df. Choose that value independently of the model comparison. `fit_compare` returns fits and information criteria without ASR; environmental models are not currently supported by `fit_compare_estim`.

### Fit regime-dependent multivariate evolution

Here `mapped.simmap` describes histories on the same tree, and `traits.csv` also contains `lnVD`:

```julia
sim = read_simmap("mapped.simmap")
X = align_traits_to_tree(tree, traits;
    taxon_col=:taxon, trait_cols=[:lnH, :lnVD])

fit = fit_mvoumva(tree, X, sim;
    trait_names=["lnH", "lnVD"], A_decomp=:diagonal)
```

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

Continuous models require bifurcating trees, and many require ultrametric trees. Missing-data support varies by model. Check taxon and SIMMAP alignment, compare AIC using the same observations and consistent assumptions, and inspect convergence and boundary estimates. Time-shift and environmental models currently require ultrametric trees and do not support observation error or ASR; environmental models are univariate, with explicit fixed-df smoothing. Check integration accuracy for steep environmental histories.

## Documentation and Validation

Use Julia's help mode for individual APIs, for example `?fit_t_env`, `?smooth_env`, `?fit_mvoubm` or `?fit_mvshift`. The source is organized around discrete models, continuous models, reconstruction and simulation; `test/` contains unit and regression coverage.

