"""
    mk_pruning_cache(tree, tip_priors, Q; kwargs...)

Run the Mk postorder pruning pass and return the reusable cache needed for
likelihood inspection, endpoint sampling, and downstream stochastic mapping.
"""
function mk_pruning_cache(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior::Symbol = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    nparams::Union{Nothing, Integer} = nothing,
)
    priors = _validate_tip_priors(tree, tip_priors)
    Qf = _validate_rate_matrix(Q)
    nstates = size(Qf, 1)
    size(priors, 2) == nstates || throw(ArgumentError("tip_priors and Q disagree on number of states"))
    root_prior_vec = _root_prior_vector(Qf, root_prior, root_prior_probs)

    node_priors = zeros(Float64, tree.nnodes, nstates)
    for (row, node_id) in enumerate(tree.tip_ids)
        @views node_priors[node_id, :] .= priors[row, :]
    end

    evals, V, Vinv = _mk_eigen_cache(Qf)
    logpost = zeros(Float64, tree.nnodes, nstates)
    P = zeros(Float64, nstates, nstates)
    Y = zeros(Float64, nstates)
    logY = zeros(Float64, nstates)
    ll_shift = 0.0

    foreach_postorder_internal(tree) do node
        @views fill!(logpost[node, :], 0.0)
        foreach_child_edge(tree, node) do edge, child
            _fill_transition_matrix!(P, tree.edge_length[edge], evals, V, Vinv)
            child_is_tip = tree.is_tip[child]
            child_vec = child_is_tip ? @view(node_priors[child, :]) : @view(logpost[child, :])

            if child_is_tip
                mul!(Y, P, child_vec)
                @inbounds for s in 1:nstates
                    logpost[node, s] += log(max(Y[s], EVOTRAITS_TINY))
                end
            else
                @inbounds for s in 1:nstates
                    logY[s] = -Inf
                end
                @inbounds for parent_state in 1:nstates
                    acc = -Inf
                    for child_state in 1:nstates
                        acc = logaddexp2(acc, log(max(P[parent_state, child_state], EVOTRAITS_TINY)) + child_vec[child_state])
                    end
                    logY[parent_state] = acc
                end
                @inbounds for s in 1:nstates
                    logpost[node, s] += logY[s]
                end
            end
        end

        scale = 0.0
        @inbounds for s in 1:nstates
            scale += exp(logpost[node, s])
        end
        scale > 0.0 || return MkPruningCache(success=false, loglik=-Inf, aic=Inf, nstates=nstates, nparams=(nparams === nothing ? 0 : Int(nparams)), root_prior=root_prior, scaling_shift=ll_shift)
        log_scale = log(scale)
        @inbounds for s in 1:nstates
            logpost[node, s] -= log_scale
        end
        ll_shift += log_scale
    end

    root = tree.root
    root_loglik =
        if tree.is_tip[root]
            root_vec = @view node_priors[root, :]
            if root_prior === :custom || root_prior === :stationary || root_prior === :flat
                log(sum(root_vec .* root_prior_vec))
            elseif root_prior === :max_likelihood
                log(maximum(root_vec))
            else
                denom = sum(root_vec)
                log(sum(abs2, root_vec) / denom)
            end
        else
            root_vec = @view logpost[root, :]
            if root_prior === :custom || root_prior === :stationary || root_prior === :flat
                total = 0.0
                for s in 1:nstates
                    total += exp(root_vec[s]) * root_prior_vec[s]
                end
                log(total)
            elseif root_prior === :max_likelihood
                maximum(root_vec)
            else
                denom = 0.0
                numer = 0.0
                for s in 1:nstates
                    v = exp(root_vec[s])
                    numer += v * v
                    denom += v
                end
                log(numer / denom)
            end
        end

    final_loglik = root_loglik + ll_shift
    nfree = Int(something(nparams, _default_mk_nparams(Qf)))
    return MkPruningCache(
        success = isfinite(final_loglik),
        loglik = final_loglik,
    aic = compute_aic(final_loglik, nfree),
        nstates = nstates,
        nparams = nfree,
        root_prior = root_prior,
        scaling_shift = ll_shift,
        node_priors = node_priors,
        logpost = logpost,
        transition_matrix = Qf,
    )
end
