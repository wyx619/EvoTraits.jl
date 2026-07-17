"""
    MkFitResult

Result type returned by Mk maximum-likelihood fitting. In addition to the best
log-likelihood and AIC, it stores fitted rates, the corresponding transition
matrix, starting values, and basic optimizer diagnostics.
"""
Base.@kwdef struct MkFitResult
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    nstates::Int = 0
    root_prior::Symbol = :likelihoods
    nrates::Int = 0
    rates::Vector{Float64} = Float64[]
    transition_matrix::Matrix{Float64} = zeros(0, 0)
    start_rates::Vector{Float64} = Float64[]
    trial_logliks::Vector{Float64} = Float64[]
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
    state_labels::Vector{Any} = Vector{Any}()
    state_encoding::Dict{Any, Int} = Dict{Any, Int}()
end

function Base.show(io::IO, res::MkFitResult)
    labels = isempty(res.state_labels) ? "" : ", state_labels=$(res.state_labels)"
    print(io, "MkFitResult(success=$(res.success), states=$(res.nstates), rates=$(res.nrates), loglik=$(res.loglik), aic=$(res.aic), converged=$(res.converged), iterations=$(res.iterations)$(labels))")
end

function Base.show(io::IO, ::MIME"text/plain", res::MkFitResult)
    show(io, res)
end

"""
    _with_mk_fit_state_metadata(res::MkFitResult, state_labels::Vector{Any})

Return a copy of the given MkFitResult with the provided state_labels and
state_encoding inferred from the labels. This keeps all other fields intact.
"""
function _with_mk_fit_state_metadata(res::MkFitResult, state_labels::Vector{Any})
    state_encoding = Dict{Any, Int}()
    for (i, state) in enumerate(state_labels)
        state_encoding[state] = i
    end
    return MkFitResult(;
        success = res.success,
        loglik = res.loglik,
        aic = res.aic,
        nparams = res.nparams,
        nstates = res.nstates,
        root_prior = res.root_prior,
        nrates = res.nrates,
        rates = res.rates,
        transition_matrix = res.transition_matrix,
        start_rates = res.start_rates,
        trial_logliks = res.trial_logliks,
        converged = res.converged,
        iterations = res.iterations,
        f_calls = res.f_calls,
        state_labels = state_labels,
        state_encoding = state_encoding,
    )
end

"""
    MkLikelihoodResult

Lightweight result for a single Mk log-likelihood evaluation.
"""
Base.@kwdef struct MkLikelihoodResult
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nstates::Int = 0
    nparams::Int = 0
    root_prior::Symbol = :likelihoods
    scaling_shift::Float64 = 0.0
end

"""
    MkPruningCache

Reusable Mk pruning cache containing tip priors, internal-node log-posteriors,
and the validated transition matrix used for likelihood evaluation and endpoint
sampling.
"""
Base.@kwdef struct MkPruningCache
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nstates::Int = 0
    nparams::Int = 0
    root_prior::Symbol = :likelihoods
    scaling_shift::Float64 = 0.0
    node_priors::Matrix{Float64} = zeros(0, 0)
    logpost::Matrix{Float64} = zeros(0, 0)
    transition_matrix::Matrix{Float64} = zeros(0, 0)
end

"""
    MkEndpointSample

Sampled Mk root state together with branch start and end states for every edge.
"""
Base.@kwdef struct MkEndpointSample
    success::Bool = false
    nstates::Int = 0
    root_state::Int32 = 0
    node_states::Vector{Int32} = Int32[]
    edge_start_states::Vector{Int32} = Int32[]
    edge_end_states::Vector{Int32} = Int32[]
    loglik::Float64 = NaN
end

"""
    SimmapSegment

Single branch segment in a stochastic character map, represented by a discrete
state label and the segment length spent in that state.
"""
Base.@kwdef struct SimmapSegment
    state::Int32 = 0
    length::Float64 = 0.0
end

"""
    SimmapSample

Full stochastic character map result, including sampled node states,
branch-endpoint states, ordered per-edge segments, and the dense
`mapped_edge[edge, state]` summary.
"""
Base.@kwdef struct SimmapSample
    success::Bool = false
    nstates::Int = 0
    root_state::Int32 = 0
    state_labels::Vector{String} = String[]
    node_states::Vector{Int32} = Int32[]
    edge_start_states::Vector{Int32} = Int32[]
    edge_end_states::Vector{Int32} = Int32[]
    edge_segments::Vector{Vector{SimmapSegment}} = Vector{Vector{SimmapSegment}}()
    mapped_edge::Matrix{Float64} = zeros(0, 0)
    loglik::Float64 = NaN
end

