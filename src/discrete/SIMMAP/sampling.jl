"""
    simmap_samples(tree; tip_states, rate_model=:ARD, nsim=1, kwargs...)

Fit an Mk model from tip states and draw one or more stochastic character maps.
This is a convenience wrapper around `fit_mk`, `tip_priors_from_states`, and
`simmap_sample`.
"""
function simmap_samples(
    tree::CompactTree;
    tip_states = nothing,
    state_order = nothing,
    rate_model::Symbol = :ARD,
    nsim::Integer = 1,
    root_prior = :flat,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    fit_kwargs...,
)
    nsim >= 1 || throw(ArgumentError("nsim must be at least 1"))
    _require_order_sensitive_state_order(rate_model, state_order)

    if tip_states isa AbstractDict
        encoded, state_labels = _encode_tip_states_from_mapping(tree, tip_states; state_order = state_order)
    elseif tip_states isa AbstractVector{<:Pair}
        encoded, state_labels = _encode_tip_states_from_pairs(tree, tip_states; state_order = state_order)
    elseif tip_states isa AbstractVector{<:Integer}
        isempty(tip_states) && throw(ArgumentError("tip_states cannot be empty"))
        encoded = Int.(tip_states)
        nstates = maximum(encoded)
        nstates >= 2 || throw(ArgumentError("Need at least two observed states for SIMMAP sampling"))
        state_labels = Any[i for i in 1:nstates]
    else
        throw(ArgumentError("tip_states must be integer states, a tip-label mapping, or vector of label=>state pairs"))
    end

    nstates = length(state_labels)
    fit = fit_mk(tree, nstates; tip_states = encoded, rate_model = rate_model, root_prior = root_prior, root_prior_probs = root_prior_probs, fit_kwargs...)
    fit.success || return [SimmapSample(success = false, nstates = nstates, state_labels = string.(state_labels), loglik = fit.loglik) for _ in 1:nsim]
    priors = tip_priors_from_states(tree, encoded, nstates)
    sampler = prepare_simmap_sampler(
        tree,
        priors,
        fit.transition_matrix;
        root_prior = fit.root_prior,
        root_prior_probs = root_prior_probs === nothing ? fit.root_prior_probs : root_prior_probs,
        nparams = fit.nparams,
        state_labels = state_labels,
    )
    return simmap_samples(sampler; nsim = nsim, rng = rng)
end

function simmap_samples(
    tree::CompactTree,
    fit::MkFitResult;
    tip_states = nothing,
    state_order = nothing,
    nsim::Integer = 1,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    nsim >= 1 || throw(ArgumentError("nsim must be at least 1"))
    fit.success || return [SimmapSample(success = false, nstates = fit.nstates, state_labels = string.(fit.state_labels), loglik = fit.loglik) for _ in 1:nsim]

    if tip_states isa AbstractDict
        encoded, state_labels = _encode_tip_states_from_mapping(tree, tip_states; state_order = state_order)
    elseif tip_states isa AbstractVector{<:Pair}
        encoded, state_labels = _encode_tip_states_from_pairs(tree, tip_states; state_order = state_order)
    elseif tip_states isa AbstractVector{<:Integer}
        encoded = Int.(tip_states)
        state_labels = isempty(fit.state_labels) ? Any[i for i in 1:fit.nstates] : fit.state_labels
    else
        throw(ArgumentError("tip_states are required to build tip priors when sampling from an existing MkFitResult"))
    end

    length(state_labels) == fit.nstates || throw(ArgumentError("tip_states and fit.nstates disagree"))
    priors = tip_priors_from_states(tree, encoded, fit.nstates)
    sampler = prepare_simmap_sampler(
        tree,
        priors,
        fit.transition_matrix;
        root_prior = fit.root_prior,
        root_prior_probs = root_prior_probs === nothing ? fit.root_prior_probs : root_prior_probs,
        nparams = fit.nparams,
        state_labels = state_labels,
    )
    return simmap_samples(sampler; nsim = nsim, rng = rng)
end

function simmap_samples(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    nsim::Integer = 1,
    root_prior::Symbol = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    nparams::Union{Nothing, Integer} = nothing,
    state_labels::AbstractVector = String[],
    rng::AbstractRNG = Random.default_rng(),
)
    sampler = prepare_simmap_sampler(
        tree,
        tip_priors,
        Q;
        root_prior = root_prior,
        root_prior_probs = root_prior_probs,
        nparams = nparams,
        state_labels = state_labels,
    )
    return simmap_samples(sampler; nsim = nsim, rng = rng)
end

function simmap_sample(tree::CompactTree, fit::MkFitResult; kwargs...)
    return first(simmap_samples(tree, fit; nsim = 1, kwargs...))
end

function simmap_sample(tree::CompactTree; kwargs...)
    samples = simmap_samples(tree; nsim = 1, kwargs...)
    return first(samples)
end

maps(args...; kwargs...) = simmap_samples(args...; kwargs...)
mapsample(args...; kwargs...) = simmap_sample(args...; kwargs...)
