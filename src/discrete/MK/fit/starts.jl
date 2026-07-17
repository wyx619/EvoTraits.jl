function _default_mk_guess(tree::CompactTree, nstates::Integer; rate_model::Symbol = :ARD)
    mean_edge = isempty(tree.edge_length) ? 1.0 : mean(tree.edge_length)
    scale = max(mean_edge, 1e-6)
    rate = nstates / (scale * max(log(tree.ntips) / log(2.0), 1.0))
    return fill(rate, mk_nrates(rate_model, nstates))
end

function _prepare_root_prior(root_prior, root_prior_probs, Q, nstates)
    if root_prior isa AbstractVector
        return (:custom, normalize_probability_vector(root_prior))
    elseif root_prior === :custom
        root_prior_probs === nothing && throw(ArgumentError("root_prior_probs required when root_prior=:custom"))
        return (:custom, normalize_probability_vector(root_prior_probs))
    elseif root_prior === :stationary
        return (:stationary, stationary_distribution(Q))
    elseif root_prior === :flat || root_prior === :likelihoods || root_prior === :max_likelihood
        return (root_prior, root_prior_probs)
    else
        throw(ArgumentError("Unsupported root_prior=$root_prior"))
    end
end

function _tip_priors_input(tree::CompactTree, nstates::Integer; tip_states=nothing, tip_priors=nothing)
    if (tip_states === nothing) == (tip_priors === nothing)
        throw(ArgumentError("Provide exactly one of tip_states or tip_priors"))
    end
    if tip_states !== nothing
        return tip_priors_from_states(tree, tip_states, nstates)
    end
    priors = _validate_tip_priors(tree, tip_priors)
    size(priors, 2) == nstates || throw(ArgumentError("tip_priors and nstates disagree"))
    return priors
end

function _scout_objective(tree, priors, nstates, trial_rates, root_prior, root_prior_probs, rate_model)
    Q = mk_rates_to_Q(trial_rates, nstates; rate_model = rate_model)
    root_mode, root_vec = _prepare_root_prior(root_prior, root_prior_probs, Q, nstates)
    fit = mk_loglikelihood(tree, priors, Q; root_prior=root_mode, root_prior_probs=root_vec, nparams=length(trial_rates))
    return fit.loglik
end

function _generate_ard_starts(
    tree::CompactTree,
    priors::AbstractMatrix{<:Real},
    nstates::Integer;
    rate_model::Symbol = :ARD,
    root_prior::Symbol,
    root_prior_probs=nothing,
    guess_rates::Union{Nothing, AbstractVector{<:Real}}=nothing,
    Ntrials::Integer=1,
    Nscouts::Union{Nothing, Integer}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    threaded::Bool=false,
)
    default_start = guess_rates === nothing ? _default_mk_guess(tree, nstates; rate_model = rate_model) : Float64.(guess_rates)
    length(default_start) == mk_nrates(rate_model, nstates) || throw(ArgumentError("guess_rates has wrong length"))
    if Ntrials <= 1
        return [default_start]
    end

    nscouts = something(Nscouts, min(10_000, 10 * length(default_start) * Ntrials))
    starts_pool = Vector{Vector{Float64}}(undef, nscouts)
    objectives = fill(-Inf, nscouts)
    power_range = 6.0

    if threaded
        Threads.@threads for k in 1:nscouts
            local_rng = MersenneTwister(hash((k, nstates, tree.ntips)))
            width = ((k / nscouts)^2) * power_range / 2
            start = similar(default_start)
            for i in eachindex(default_start)
                start[i] = default_start[i] * 10.0 ^ ((2rand(local_rng) - 1) * width)
            end
            starts_pool[k] = start
            objectives[k] = _scout_objective(tree, priors, nstates, start, root_prior, root_prior_probs, rate_model)
        end
    else
        for k in 1:nscouts
            local_rng = MersenneTwister(hash((k, nstates, tree.ntips)))
            width = ((k / nscouts)^2) * power_range / 2
            start = similar(default_start)
            for i in eachindex(default_start)
                start[i] = default_start[i] * 10.0 ^ ((2rand(local_rng) - 1) * width)
            end
            starts_pool[k] = start
            objectives[k] = _scout_objective(tree, priors, nstates, start, root_prior, root_prior_probs, rate_model)
        end
    end

    order = sortperm(objectives; rev=true)
    selected = Vector{Vector{Float64}}()
    push!(selected, default_start)
    for idx in order
        length(selected) >= Ntrials && break
        isfinite(objectives[idx]) || continue
        push!(selected, starts_pool[idx])
    end
    while length(selected) < Ntrials
        push!(selected, default_start)
    end
    return selected
end
