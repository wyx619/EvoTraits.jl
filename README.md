# EvoTraits

> **Fit and Analysis Trait Evolution Models in Julia with High Efficiency**

EvoTraits is a Julia framework for phylogenetic comparative analysis. It brings tree I/O, discrete and continuous trait models, SIMMAP histories, ancestral-state reconstruction, and model comparison into one typed workflow.

The project treats the phylogenetic tree as the computational backbone: data are aligned to tree tips, likelihoods are evaluated through tree-pruning kernels, and reconstruction results retain node, branch, and time coordinates for downstream analysis.

## Why EvoTraits

R has an exceptional comparative-methods ecosystem. EvoTraits is for users who need a complementary execution model:

- **Julia-native execution:** ordinary package use does not require R or an R bridge.
- **One internal tree representation:** CompactTree stores topology, branch lengths, traversals, labels, and node distances in array-oriented form.
- **Pruning-first likelihoods:** continuous and discrete calculations operate on tree structure; multivariate routines use trait-space matrix operators rather than an observation-sized dense VCV matrix.
- **Structured results:** fit objects retain likelihood, AIC, convergence diagnostics, parameter estimates, trait names, regime names, and model metadata.
- **A connected analysis path:** align data, fit models, compare AIC, sample histories, reconstruct nodes or branches, and export tables inside one engine.
- **Validation-friendly output:** stable tree references and explicit time fields support comparisons with ape, corHMM, OUwie, mvMORPH, geiger, and castor.

EvoTraits is not an R wrapper. It preserves the statistical ideas users rely on while using Julia's type system, native execution, and composable workflows to make large analyses easier to control.

## Capabilities

### Continuous models

Univariate:

- Brownian motion: fit_bm1, fit_bmm
- Early burst: fit_eb, fit_ebm
- OU family: fit_ou1, fit_oum, fit_oumv, fit_ouma, fit_oumva

Multivariate:

- Brownian motion: fit_mvbm1, fit_mvbmm
- Early burst: fit_mveb
- OU family: fit_mvou1, fit_mvoum, fit_mvoumv, fit_mvouma, fit_mvoumva

Regime-aware models accept SIMMAP edge segments. Model-specific optima, attraction matrices, and diffusion matrices are stored in dedicated result types.

### Discrete models

- Mk likelihood and reconstruction: fit_mk, asr_mk
- corHMM-style observed and hidden-rate models: fit_corhmm, asr_corhmm
- Rate-index and Q-matrix helpers: rateindex, rates_to_q, qfromindex
- Missing and ambiguous tip states through tip-prior matrices
- Root-prior modes including likelihood, flat, Yang, Maddison-FitzJohn, stationary, and explicit probabilities where supported

### SIMMAP

- simmap_sample and simmap_samples for generic Mk/SIMMAP sampling
- simmap_corhmm for corHMM fit results
- read_simmap and write_simmap for file exchange
- describe_simmap, summary_simmap, transition_times, and transition tables
- drop_tip_simmap and keep_tip_simmap for tree and map restriction

A SimmapSample contains ordered per-edge segments and mapped regime durations. Segment lengths are checked against branch lengths before regime-aware continuous fitting.

### Ancestral reconstruction

- Continuous nodes: estim_node and estim_node_table
- SIMMAP-aware continuous branches: estim_branch_for_simmap and estim_branch_table
- Discrete marginal or joint reconstruction: asr_mk and asr_corhmm
- Time-window summaries: summarize_node_estimates_by_time

Node and branch results expose both time_from_root and time_before_present. This supports parent-child change analysis, temporal summaries, and joins between continuous reconstruction and discrete regimes.

### Model comparison and simulation

- Information criteria: aic, aicc, aic_table, delta_aic, best_model
- Full comparison workflows: fit_compare_estim and fit_compare_multivariate
- Yule and birth-death tree simulation
- Univariate and multivariate BM, EB, and OU-family trait simulation
- Cross-language node and edge references through PhyloRef

## Quick start

### Load a tree and fit OU1

For a general Newick parse, use read_tree and explicitly serialize the result. For large-tree workflows, load_tree is the direct high-throughput path and returns CompactTree immediately.

~~~julia
using EvoTraits
using CSV, DataFrames

tree = load_tree("tree.nwk")
traits = CSV.read("trait.csv", DataFrame)
lnH = align_traits_to_tree(tree, traits; taxon_col = :taxon, trait_cols = [:lnH])

fit = fit_ou1(tree, lnH; trait_name = "lnH")
asr = estim_node(tree, lnH, fit)
node_table = estim_node_table(tree, asr)
~~~

The explicit conversion path remains available:

~~~julia
parsed = read_tree("tree.nwk")
tree = serialize_tree(parsed)
~~~

### Fit a multivariate OU model on a SIMMAP

~~~julia
using EvoTraits
using CSV, DataFrames

tree = load_tree("tree.nwk")
sim = read_simmap("mapped_tree.simmap")
traits = CSV.read("traits.csv", DataFrame)
X = align_traits_to_tree(tree, traits; taxon_col = :taxon, trait_cols = [:lnH, :lnVD])

fit = fit_mvoum(tree, X, sim; trait_names = ["lnH", "lnVD"])
asr = estim_node(tree, X, fit, sim)
node_table = estim_node_table(tree, asr, fit; simmap = sim)
~~~

For two traits and a full asymmetric attraction matrix:

~~~julia
fit = fit_mvoumva(
    tree,
    X,
    sim;
    trait_names = ["lnH", "lnVD"],
    A_decomp = :schur,
)
~~~

The current Schur path targets the p = 2 asymmetric-A implementation. Cholesky remains the default for the symmetric positive-definite attraction parameterization.

### Fit a discrete model and sample histories

~~~julia
states = ["A", "B", "?", "A"]  # ordered to match tree tips
fit = fit_corhmm(tree; tip_states = states, model = :ARD)
asr = asr_corhmm(fit; mode = :marginal)
maps = simmap_corhmm(fit; nsim = 100)
~~~

For strict cross-language comparisons, provide state_order explicitly so Q-matrix rows, ASR columns, and SIMMAP labels have stable meanings.

### Compare models

~~~julia
workflow = fit_compare_estim(
    tree,
    lnH,
    [:BM1, :OU1, :EB];
    max_iterations = 400,
)

workflow.aic_table
workflow.best_name
workflow.asr
~~~

The multivariate counterpart is fit_compare_multivariate. The workflow returns all fits, a ranked AIC table, the selected model name, the selected fit, and its reconstruction.

## Tree and data conventions

- CompactTree is the internal representation used by likelihood, SIMMAP, and reconstruction kernels.
- load_tree(path) parses one Newick tree directly into CompactTree and rejects multiple trees in one file.
- read_tree(path) returns NewickTree.Node; serialize_tree converts it to CompactTree.
- write_tree(path, tree) writes supported tree objects; to_newick and from_compact_tree provide conversion helpers.
- DataFrame alignment uses tree tip labels as the backbone. Trait columns may contain missing values where supported.
- Current continuous pruning models require bifurcating trees; OU paths validate ultrametricity where required.
- SIMMAP segments must be ordered and sum to the corresponding branch length.
- Pass state_order for reproducible state numbering.

## Root treatment and numerical choices

OU fit results record root metadata. Depending on the model, relevant keywords include root_mean_mode, root_cov_mode, and A_decomp.

- Root mean handling distinguishes regime-root optimum from stationary-design treatment.
- Root covariance handling distinguishes fixed-root covariance from stationary root covariance.
- Multivariate OU defaults to A_decomp = :cholesky; the current asymmetric Schur implementation supports two traits.

These are statistical model choices, not merely implementation details. Keep them fixed when comparing models or validating against an R implementation.

## Threading

Start Julia with the desired number of threads, for example `julia -t 8 --project=.`. Continuous OU-family multi-start paths use Julia threads when available. Mk and corHMM expose fit-level `Ntrials` and `Nthreads` controls for trial parallelism. When combining Julia threads with threaded BLAS, set the BLAS thread count explicitly with `set_engine_blas_threads!(1)` to avoid oversubscription.

## Results and interoperability

Fit objects commonly expose:

~~~text
model, success, loglik, aic, nparams,
converged, iterations, f_calls
~~~

Model-specific fields include theta, alpha, sigma2, A, Sigma, trait_names, regime_names, root_mean_mode, root_cov_mode, and A_decomp as applicable.

PhyloRef and estim_node_table provide stable mappings between EvoTraits node IDs and R/ape-style node IDs, together with tip anchors and descendant-tip signatures.

## Repository layout

~~~text
src/
  io.jl          CompactTree and Newick I/O
  criteria.jl    shared information criteria
  phyloref/      tree identity and cross-language mappings
  discrete/      Mk, corHMM, and SIMMAP
  continuous/   univariate, multivariate, reconstruction, and workflows
  simulate/      tree and trait simulation

test/            unit and regression tests
validation/      empirical validation projects
reference/       local reference implementations and study material
docs/            notebooks and project documentation
~~~

## Installation

From a local checkout:

~~~julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
~~~

## Testing

~~~julia
import Pkg
Pkg.test()
~~~

Focused test files can be run directly with the project environment while developing a subsystem. Validation projects remain separate from the ordinary package test suite.

## Status and citation

EvoTraits is research software under active development. Formal analyses should be checked with simulated data and, where relevant, an independent implementation such as corHMM, OUwie, or mvMORPH.

If you use EvoTraits in research, cite this repository together with the original methodological papers for the models used in your analysis.
