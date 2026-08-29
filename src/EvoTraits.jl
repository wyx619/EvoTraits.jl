module EvoTraits

using MKL
using LinearAlgebra
using Statistics
using Random
using NewickTree
using NLopt
using Optim
using DataFrames
using StatsModels
using Distributions

# Base types and shared utilities
include("io.jl")
include("criteria.jl")

# Discrete trait models and SIMMAP
include("discrete/common.jl")
include("discrete/models.jl")
include("discrete/simmap.jl")

# Tree identity and cross-language mapping
include("phyloref.jl")

# Simulation
include("simulate.jl")

# Continuous trait models and reconstruction
include("continuous/checks.jl")
include("continuous/types.jl")
include("continuous/univariate.jl")
include("continuous/multivariate.jl")
include("continuous/reconstruction.jl")
include("continuous/workflows.jl")

# Core tree type
export CompactTree

# Tree I/O and conversion
export read_tree
export write_tree
export serialize_tree
export to_newick
export from_compact_tree
export to_real_tree

# Simulation
export SimulatedTree
export simulate_yule_simtree
export simulate_yule_tree
export simulate_birth_death_simtree
export simulate_birth_death_tree
export simulate_ultrametric_newick
export simulate_trait
export simulate_mv_data
export simulate_mvbm1
export simulate_mvbmm
export simulate_mvou1
export simulate_mvoum
export simulate_mvoumva
export simulate_mveb
export simulate_mvbm1_dataset

# Discrete likelihood, SIMMAP, and transition types
export MkLikelihoodResult
export MkPruningCache
export MkEndpointSample
export SimmapSegment
export SimmapSample
export SimmapSummary
export SimmapSampler
export SimmapTransitionEvent
export SimmapTransitionRate

# Univariate continuous result types
export ContinuousFitResult
export ContinuousMultiRegimeResult
export ContinuousASRResult
export ContinuousBranchPosteriorResult
export ContinuousWorkflowResult

# Multivariate continuous result types
export MVContinuousBMResult
export MVContinuousMultiBMResult
export MVContinuousASRResult
export MVContinuousBranchPosteriorResult
export MVContinuousWorkflowResult
export MVOUPrecalc
export MVOUParameterBundle
export MVContinuousOUResult

# Multivariate continuous fitting and likelihood entry points
export mvbm1_loglikelihood
export fit_mvbm1
export mvou1_loglikelihood
export fit_mvou1
export mvoum_loglikelihood
export fit_mvoum
export mvoumv_loglikelihood
export fit_mvoumv
export mvouma_loglikelihood
export fit_mvouma
export mvoumva_loglikelihood
export fit_mvoumva
export mvbmm_loglikelihood
export fit_mvbmm
export mveb_loglikelihood
export fit_mveb

# Univariate continuous fitting and likelihood entry points
export bm1_loglikelihood
export fit_bm1
export bmm_loglikelihood
export fit_bmm
export ou1_loglikelihood
export fit_ou1
export eb_loglikelihood
export fit_eb
export ebm_loglikelihood
export fit_ebm
export oum_loglikelihood
export fit_oum
export oumv_loglikelihood
export fit_oumv
export ouma_loglikelihood
export fit_ouma
export oumva_loglikelihood
export fit_oumva

# Shared model-comparison utilities
export aic
export aicc
export loglik
export nparams
export model
export delta_aic
export aic_table
export best_model
export fit_compare_estim
export fit_compare_multivariate

# Continuous reconstruction
export estim_node
export estim_node_table
export estim_branch_for_simmap
export estim_branch_table
export align_traits_to_tree

# Discrete model fit result types
export MkASRResult
export CorHMMStateData
export CorHMMFitResult
export CorHMMASRResult
export CorHMMSimmapResult

# Discrete trait fitting and inference
export tip_priors_from_states
export mk_loglikelihood
export mk_pruning_cache
export sample_mk_endpoints
export fit_mk
export asr_mk
export fit_corhmm
export asr_corhmm

# Discrete rate-model helpers
export mk_nrates
export mk_rates_to_Q
export mk_Q_to_rates

# SIMMAP workflows and utilities
export simmap_corhmm
export simmap_endpoints
export prepare_simmap_sampler
export simmap_sample
export simmap_samples
export transition_times
export transition_events_DataFrame
export transition_rates_through_time
export read_simmap
export write_simmap
export drop_tip_simmap
export keep_tip_simmap
export keep_tip
export drop_tip
export print_simmap
export describe_simmap
export summary_simmap

# The exported names above are the stable user-facing API. The implementation
# aliases `endpoints`, `preparesampler`, and `transitionevents` remain internal;
# `transition_events_DataFrame` is retained as a public compatibility name.

# Tree identity and mapping utilities
export PhyloRef
export build_phyloref
export phyloref_node_table
export phyloref_edge_table

# Reconstruction and time summaries
export summarize_node_estimates_by_time

# Runtime utilities
export set_engine_blas_threads!

"""
    set_engine_blas_threads!(n::Integer)

Set BLAS thread count explicitly. This is useful to avoid thread oversubscription
when Julia-level threading is used for trials, bootstraps, or simmap replicates.
"""
function set_engine_blas_threads!(n::Integer)
    BLAS.set_num_threads(n)
    return BLAS.get_num_threads()
end

end # module

