module EvoTraits

using MKL
using LinearAlgebra
using Statistics
using Random
using NewickTree
import NLopt
import Optim
using DataFrames
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
include("continuous/shifts.jl")
include("continuous/reconstruction.jl")
include("continuous/workflows.jl")

# Core tree type
export CompactTree

# Tree I/O and conversion
export read_tree
export load_tree
export write_tree
export serialize_tree
export to_newick
export from_compact_tree
export to_real_tree

# Simulation
export SimulatedTree
export sim_yule_simtree
export sim_yule_tree
export sim_birth_death_simtree
export sim_birth_death_tree
export sim_ultrametric_newick
export sim_trait
export sim_mv_data
export sim_mvbm1
export sim_mvbmm
export sim_mvou1
export sim_mvoum
export sim_mvoumva
export sim_mveb
export sim_mvbm1_dataset

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
export mvbm1_LL
export fit_mvbm1
export mvou1_LL
export fit_mvou1
export mvoum_LL
export fit_mvoum
export mvoumv_LL
export fit_mvoumv
export mvouma_LL
export fit_mvouma
export mvoumva_LL
export fit_mvoumva
export mvoubm_LL
export fit_mvoubm
export mvbmm_LL
export fit_mvbmm
export mveb_LL
export fit_mveb

# Univariate continuous fitting and likelihood entry points
export bm1_LL
export fit_bm1
export bmm_LL
export fit_bmm
export ou1_LL
export fit_ou1
export eb_LL
export fit_eb
export ebm_LL
export fit_ebm
export oum_LL
export fit_oum
export oumv_LL
export fit_oumv
export ouma_LL
export fit_ouma
export oumva_LL
export fit_oumva

# Time-shift models
export ShiftFitResult
export ShiftPrecalc
export shift_precalc
export shift_LL
export mvshift_LL
export fit_shift
export fit_mvshift
export diagnose_shift

# Environment-dependent models
export EnvCurve
export SmoothEnvCurve
export EnvFitResult
export smooth_env
export t_env_LL
export fit_t_env

# Shared model-comparison utilities
export aic
export aicc
export loglik
export nparams
export model
export delta_aic
export aic_table
export best_model
export ContinuousModelCandidate
export fit_compare_estim
export fit_compare
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
export mk_LL
export mk_pruning_cache
export sample_mk_endpoints
export fit_mk
export asr_mk
export corhmm_LL
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
export transition_events_table
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

# Legacy compatibility name; use `transition_events_table` in new code.
export transition_events_DataFrame

# The SIMMAP entry points `endpoints`, `preparesampler`, and `transitionevents`
# remain internal.

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

Set the process-wide BLAS thread count to the positive integer `n` and return
the resulting count. EvoTraits sets BLAS to one thread when the package loads
to avoid oversubscription with Julia-level threading. Call this function after
loading the package to override that default; other packages share this setting.
"""
function set_engine_blas_threads!(n::Integer)
    n > 0 || throw(ArgumentError("BLAS thread count must be positive"))
    BLAS.set_num_threads(n)
    return BLAS.get_num_threads()
end

function __init__()
    set_engine_blas_threads!(1)
    return nothing
end

end # module

