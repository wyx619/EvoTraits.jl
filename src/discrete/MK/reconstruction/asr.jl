"""
    asr_mk(tree, nstates; kwargs...)

Ancestral state reconstruction for discrete Mk models. This function first fits
the Mk model by maximum likelihood via `fit_mk`, then computes marginal
ancestral likelihoods for all internal nodes.

Tip information may be supplied either as integer states or as a dense
tip-prior matrix. The transition matrix is estimated internally and never
accepted as a user-facing argument.
"""
function asr_mk(
    tree::CompactTree,
    nstates::Integer;
    tip_states = nothing,
    tip_priors = nothing,
    rate_model::Symbol = :ER,
    root_prior = :auto,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    reroot::Bool = true,
    Ntrials::Integer = 1,
    Nscouts::Union{Nothing, Integer} = nothing,
    Nthreads::Integer = 1,
    optim_method::Symbol = :LBFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    lower_rate::Float64 = 1e-8,
    rng::AbstractRNG = Random.default_rng(),
)
    priors = _tip_priors_input(tree, nstates; tip_states=tip_states, tip_priors=tip_priors)
    resolved_root, resolved_root_vec = _resolve_asr_root_prior(root_prior, root_prior_probs, priors)

    fit = fit_mk(
        tree,
        nstates;
        tip_states = tip_states,
        tip_priors = tip_priors,
        rate_model = rate_model,
        root_prior = resolved_root,
        root_prior_probs = resolved_root_vec,
        Ntrials = Ntrials,
        Nscouts = Nscouts,
        Nthreads = Nthreads,
        optim_method = optim_method,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_rate = lower_rate,
        rng = rng,
    )

    fit.success || return MkASRResult(
        success = false,
        nstates = nstates,
        root_prior = resolved_root,
        rate_model = rate_model,
        reroot = reroot,
        fit = fit,
    )

    Q = fit.transition_matrix
    cache = mk_pruning_cache(
        tree,
        priors,
        Q;
        root_prior = fit.root_prior,
        root_prior_probs = resolved_root_vec,
        nparams = fit.nparams,
    )

    cache.success || return MkASRResult(
        success = false,
        loglik = cache.loglik,
        aic = cache.aic,
        nstates = cache.nstates,
        nparams = cache.nparams,
        root_prior = fit.root_prior,
        rate_model = rate_model,
        rates = fit.rates,
        transition_matrix = Q,
        reroot = reroot,
        fit = fit,
    )

    internal_nodes = [n for n in tree.postorder_internal if !tree.is_tip[n]]
    node_ids = Int[]
    ancestral_likelihoods = zeros(Float64, 0, nstates)

    if reroot
        ancestral_likelihoods = _mk_rerooted_ancestral_likelihoods(tree, priors, Q, fit.root_prior, resolved_root_vec, cache)
        node_ids = Int[n for n in tree.postorder_internal if !tree.is_tip[n]]
    else
        ancestral_likelihoods = _mk_local_ancestral_likelihoods(tree, cache, internal_nodes)
        node_ids = Int[n for n in internal_nodes]
    end

    ancestral_states = _mk_ancestral_states(ancestral_likelihoods)

    return MkASRResult(
        success = true,
        loglik = cache.loglik,
        aic = cache.aic,
        nparams = cache.nparams,
        nstates = cache.nstates,
        root_prior = fit.root_prior,
        rate_model = rate_model,
        rates = fit.rates,
        transition_matrix = Q,
        node_ids = node_ids,
        ancestral_likelihoods = ancestral_likelihoods,
        ancestral_states = ancestral_states,
        reroot = reroot,
        fit = fit,
    )
end

function _resolve_asr_root_prior(root_prior, root_prior_probs, priors)
    if root_prior === :auto
        return (:max_likelihood, nothing)
    elseif root_prior === :empirical
        p = vec(sum(priors; dims = 1))
        p ./= sum(p)
        return (:custom, p)
    elseif root_prior isa AbstractVector
        return (:custom, normalize_probability_vector(root_prior))
    else
        return (root_prior, root_prior_probs)
    end
end

function _mk_local_ancestral_likelihoods(tree::CompactTree, cache::MkPruningCache, internal_nodes::AbstractVector{<:Integer})
    nstates = cache.nstates
    n_nodes = length(internal_nodes)
    out = zeros(Float64, n_nodes, nstates)
    for (i, node) in enumerate(internal_nodes)
        row = @view cache.logpost[node, :]
        row_sum = 0.0
        @inbounds for s in 1:nstates
            v = exp(row[s])
            out[i, s] = v
            row_sum += v
        end
        row_sum > 0.0 || continue
        @inbounds for s in 1:nstates
            out[i, s] /= row_sum
        end
    end
    return out
end

function _mk_rerooted_ancestral_likelihoods(
    tree::CompactTree,
    priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    root_prior::Symbol,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}},
    cache::MkPruningCache,
)
    nstates = cache.nstates
    internal_nodes = [n for n in tree.postorder_internal if !tree.is_tip[n]]
    n_nodes = length(internal_nodes)
    out = zeros(Float64, n_nodes, nstates)

    evals, V, Vinv = _mk_eigen_cache(Q)
    P = zeros(Float64, nstates, nstates)

    log_up = zeros(Float64, tree.nnodes, nstates)
    for node in internal_nodes
        @inbounds for s in 1:nstates
            log_up[node, s] = 0.0
        end
    end

    root = tree.root
    if !tree.is_tip[root]
        root_prior_vec = _root_prior_vector(Q, root_prior, root_prior_probs)
        if root_prior_vec !== nothing
            @inbounds for s in 1:nstates
                log_up[root, s] = log(max(root_prior_vec[s], EVOTRAITS_TINY))
            end
        end
    end

    for node in tree.preorder
        tree.is_tip[node] && continue
        parent = tree.parent_of_edge[tree.first_child_edge[node]]
        parent == node && continue

        @inbounds for s in 1:nstates
            log_up[node, s] = -Inf
        end

        first_edge = tree.first_child_edge[node]
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            _fill_transition_matrix!(P, tree.edge_length[edge], evals, V, Vinv)

            @inbounds for s in 1:nstates
                acc = -Inf
                for t in 1:nstates
                    v = log(max(P[s, t], EVOTRAITS_TINY)) + log_up[node, t]
                    acc = logaddexp2(acc, v)
                end
                log_up[child, s] = acc
            end
        end
    end

    for (i, node) in enumerate(internal_nodes)
        @inbounds for s in 1:nstates
            v = exp(cache.logpost[node, s] + log_up[node, s])
            out[i, s] = v
        end
        row_sum = sum(@view out[i, :])
        row_sum > 0.0 || continue
        @inbounds for s in 1:nstates
            out[i, s] /= row_sum
        end
    end

    return out
end

function _mk_ancestral_states(ancestral_likelihoods::Matrix{Float64})
    n_nodes = size(ancestral_likelihoods, 1)
    states = Vector{Int32}(undef, n_nodes)
    for i in 1:n_nodes
        states[i] = Int32(argmax(@view ancestral_likelihoods[i, :]))
    end
    return states
end

function _with_mk_state_metadata(res::MkASRResult, state_labels::Vector{Any})
    state_encoding = Dict{Any, Int}()
    for (i, state) in enumerate(state_labels)
        state_encoding[state] = i
    end
    ancestral_state_labels = Any[state_labels[Int(state)] for state in res.ancestral_states]

    return MkASRResult(;
        success = res.success,
        loglik = res.loglik,
        aic = res.aic,
        nparams = res.nparams,
        nstates = res.nstates,
        root_prior = res.root_prior,
        rate_model = res.rate_model,
        rates = res.rates,
        transition_matrix = res.transition_matrix,
        node_ids = res.node_ids,
        ancestral_likelihoods = res.ancestral_likelihoods,
        ancestral_states = res.ancestral_states,
        reroot = res.reroot,
        fit = res.fit,
        state_labels = state_labels,
        state_encoding = state_encoding,
        ancestral_state_labels = ancestral_state_labels,
    )
end

function _reject_no_nstates_tip_priors(tip_priors)
    tip_priors === nothing || throw(ArgumentError("tip_priors requires explicit nstates; use asr_mk(tree, nstates; tip_priors=...)"))
    return nothing
end

function asr_mk(tree::CompactTree; tip_states = nothing, tip_priors = nothing, state_order = nothing, rate_model::Symbol = :ER, kwargs...)
    _reject_no_nstates_tip_priors(tip_priors)
    _require_order_sensitive_state_order(rate_model, state_order)

    if tip_states isa AbstractDict
        encoded, state_labels = _encode_tip_states_from_mapping(tree, tip_states; state_order = state_order)
    elseif tip_states isa AbstractVector{<:Pair}
        encoded, state_labels = _encode_tip_states_from_pairs(tree, tip_states; state_order = state_order)
    else
        throw(ArgumentError("tip_states must be a tip-label mapping or vector of label=>state pairs when nstates is omitted"))
    end

    res = asr_mk(tree, length(state_labels); tip_states = encoded, rate_model = rate_model, kwargs...)
    return _with_mk_state_metadata(res, state_labels)
end
