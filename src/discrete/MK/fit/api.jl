"""
    fit_mk(tree, nstates; kwargs...)

Fit an Mk model by maximum likelihood under one of the exported rate-model
parameterizations (`ER`, `SYM`, `SUEDE`, `SRD`, or `ARD`). Tip information may
be supplied either as integer states or as a dense tip-prior matrix.
"""
function fit_mk(
    tree::CompactTree,
    nstates::Integer;
    tip_states=nothing,
    tip_priors=nothing,
    rate_model::Symbol = :ARD,
    root_prior::Union{Symbol, AbstractVector{<:Real}} = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_rates::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_transition_matrix::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    Ntrials::Integer = 1,
    Nscouts::Union{Nothing, Integer} = nothing,
    Nthreads::Integer = 1,
    optim_method::Symbol = :LBFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    lower_rate::Float64 = 1e-8,
    rng::AbstractRNG = Random.default_rng(),
)
    return _mk_fit_with_nstates(
        tree,
        nstates;
        tip_states = tip_states,
        tip_priors = tip_priors,
        rate_model = rate_model,
        root_prior = root_prior,
        root_prior_probs = root_prior_probs,
        guess_rates = guess_rates,
        guess_transition_matrix = guess_transition_matrix,
        Ntrials = Ntrials,
        Nscouts = Nscouts,
        Nthreads = Nthreads,
        optim_method = optim_method,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_rate = lower_rate,
        rng = rng,
    )
end

function _reject_fit_no_nstates_tip_priors(tip_priors)
    tip_priors === nothing || throw(ArgumentError("tip_priors requires explicit nstates; use fit_mk(tree, nstates; tip_priors=...)"))
    return nothing
end

function fit_mk(
    tree::CompactTree;
    tip_states = nothing,
    tip_priors = nothing,
    state_order = nothing,
    rate_model::Symbol = :ARD,
    root_prior::Union{Symbol, AbstractVector{<:Real}} = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_rates::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_transition_matrix::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    Ntrials::Integer = 1,
    Nscouts::Union{Nothing, Integer} = nothing,
    Nthreads::Integer = 1,
    optim_method::Symbol = :LBFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    lower_rate::Float64 = 1e-8,
    rng::AbstractRNG = Random.default_rng(),
)
    _reject_fit_no_nstates_tip_priors(tip_priors)
    _require_order_sensitive_state_order(rate_model, state_order)

    if tip_states isa AbstractDict
        encoded, state_labels = _encode_tip_states_from_mapping(tree, tip_states; state_order = state_order)
    elseif tip_states isa AbstractVector{<:Pair}
        encoded, state_labels = _encode_tip_states_from_pairs(tree, tip_states; state_order = state_order)
    else
        throw(ArgumentError("tip_states must be a tip-label mapping or vector of label=>state pairs when nstates is omitted"))
    end

    res = fit_mk(
        tree,
        length(state_labels);
        tip_states = encoded,
        rate_model = rate_model,
        root_prior = root_prior,
        root_prior_probs = root_prior_probs,
        guess_rates = guess_rates,
        guess_transition_matrix = guess_transition_matrix,
        Ntrials = Ntrials,
        Nscouts = Nscouts,
        Nthreads = Nthreads,
        optim_method = optim_method,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_rate = lower_rate,
        rng = rng,
    )

    return _with_mk_fit_state_metadata(res, state_labels)
end
