# EvoTraits

> Fit and Analyze Trait Evolution Models in Julia with High Efficiency

EvoTraits is a Julia-native framework for phylogenetic comparative analysis. It is built around one internal tree representation, one continuous/discrete workflow surface, and one structured result style, so model fitting, ancestral reconstruction, SIMMAP-aware analysis, and model comparison can stay inside the same engine instead of being spread across loosely connected scripts.

## Why EvoTraits

Most comparative workflows are still fragmented:

- one package for discrete models
- another for continuous models
- another for SIMMAP
- another for ancestral reconstruction
- custom code for data alignment and reporting

EvoTraits is designed to replace that fragmentation with a coherent Julia framework:

- one core tree object: `CompactTree`
- tree-aware computation paths where possible
- structured fit and reconstruction result objects
- unified handling of univariate and multivariate trait models
- direct support for both continuous and discrete comparative workflows

## What EvoTraits Covers

### Continuous trait models

#### Univariate

- Brownian motion: `fit_bm1`, `fit_bmm`
- Early burst / ACDC: `fit_eb`, `fit_ebm`
- OU family: `fit_ou1`, `fit_oum`, `fit_oumv`, `fit_ouma`, `fit_oumva`

#### Multivariate

- Brownian motion: `fit_mvbm1`, `fit_mvbmm`
- Early burst: `fit_mveb`
- OU family: `fit_mvou1`, `fit_mvoum`, `fit_mvoumv`, `fit_mvouma`, `fit_mvoumva`

### Discrete trait models

- Mk: `fit_mk`, `asr_mk`
- corHMM-style hidden-rate models: `fit_corhmm`, `asr_corhmm`

### SIMMAP workflows

- sampling and simulation from fitted discrete models
- read / write / summarize SIMMAP trees
- SIMMAP-aware continuous model fitting and reconstruction

Main entry points include:

- `simmap_corhmm`
- `simmap_sample`
- `simmap_samples`
- `read_simmap`
- `write_simmap`
- `summary_simmap`

### Ancestral reconstruction

- node reconstruction: `estim_node`, `estim_node_table`
- branch reconstruction: `estim_branch_for_simmap`, `estim_branch_table`
- time-binned summaries: `summarize_node_estimates_by_time`

### Model-comparison workflows

- univariate workflow: `fit_compare_estim`
- multivariate workflow: `fit_compare_multivariate`
- shared model-comparison helpers: `aic`, `aicc`, `aic_table`, `best_model`

### Simulation and tree references

- tree simulation
- continuous trait simulation
- cross-language tree reference mapping via `PhyloRef`

## Design Direction

EvoTraits is not a wrapper around R packages. It is a rebuild in Julia, informed by the statistical logic and practical expectations established in the reference ecosystem:

- `ape`
- `phytools`
- `geiger`
- `castor`
- `OUwie`
- `mvMORPH`
- `Rphylopars`
- `corHMM`

The goal is not interface mimicry. The goal is a cleaner computational architecture with comparable methodological scope.

## Quick Start

### 1. Read a tree and fit a univariate OU model

```julia
using EvoTraits
using CSV, DataFrames

tree = serialize_tree(read_tree("tree.nwk"))
traits = CSV.read("trait.csv", DataFrame)

fit = fit_ou1(tree, traits)
asr = estim_node(tree, traits, fit)
tbl = estim_node_table(tree, asr)
```

### 2. Fit a multivariate OU model on a SIMMAP

```julia
using EvoTraits
using CSV, DataFrames

sim = read_simmap("mapped_tree.simmap")
traits = CSV.read("traits.csv", DataFrame)

fit = fit_mvoum(sim.tree, traits, sim.simmap)
asr = estim_node(sim.tree, traits, fit, sim.simmap)
tbl = estim_node_table(sim.tree, asr, fit)
```

### 3. Fit a hidden-rate discrete model and sample SIMMAP histories

```julia
fit = fit_corhmm(tree, states; model = :ARD, rate_cat = 2)
asr = asr_corhmm(fit)
maps = simmap_corhmm(fit; nsim = 100)
```

### 4. Run a model-comparison workflow

```julia
res = fit_compare_estim(tree, y, [:BM1, :OU1, :EB])
res.best_name
res.aic_table
```

## Data Conventions

- primary tree object: `CompactTree`
- tree tip order is the alignment backbone
- `DataFrame` trait input is expected to use the first column for taxon labels
- multivariate continuous input may contain partial missing values
- SIMMAP segments must sum to branch length

## Result Style

EvoTraits favors structured result objects over ad hoc tuples. Typical fields include:

- `model`
- `success`
- `loglik`
- `aic`
- `nparams`
- `converged`
- `iterations`
- `f_calls`

Model-specific parameters are stored directly in the fit result, for example:

- `sigma2`
- `alpha`
- `beta`
- `theta`
- `A`
- `Sigma`
- `trait_names`
- `regime_names`

## Repository Layout

```text
src/
  continuous/   continuous models, workflows, reconstruction
  discrete/     Mk, corHMM, SIMMAP
  phyloref/     cross-language tree mapping
  simulate/     tree and trait simulation
  io.jl         tree structure and Newick I/O
  criteria.jl   shared information criteria
```

```text
test/           unit and regression tests
validation/     empirical and alignment workflows
reference/      reference implementations and study material
docs/           notebooks and support documents
```

## Installation

From the repository root:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

From another Julia environment:

```julia
import Pkg
Pkg.add(url = "https://github.com/<your-org-or-user>/EvoTraits.jl")
```

## Testing

Run the full test suite:

```julia
import Pkg
Pkg.test()
```

Run focused subsets:

```powershell
julia --project=. test/run_subset.jl discrete/corHMM/corhmm
julia --project=. test/run_subset.jl continuous/univariate/ou_family
julia --project=. test/run_subset.jl continuous/multivariate/ou_family
julia --project=. test/run_subset.jl workflows
```

## Current Position

EvoTraits is already usable for a substantial comparative-analysis surface:

- discrete trait modeling
- continuous trait model fitting
- SIMMAP-aware workflows
- ancestral reconstruction
- multivariate OU-family analysis
- model-comparison workflows

It is still active research software, but it is no longer a loose collection of experiments. The current codebase is organized as a unified comparative framework with explicit testing and validation.

## Requirements

- Julia `1.10` to `1.12`
- R is not required for ordinary package use
- some validation workflows may rely on separate R environments

## Citation

If you use EvoTraits in research, cite both this repository and the underlying comparative-method literature relevant to your workflow.
