function jointstates(likelihoods::AbstractMatrix{<:Real})
    states = Vector{Int32}(undef, size(likelihoods, 1))
    for i in axes(likelihoods, 1)
        states[i] = Int32(argmax(@view likelihoods[i, :]))
    end
    return states
end

function jointrootprior(Q::Matrix{Float64}, root_prior::Symbol, root_prior_probs)
    vec = _root_prior_vector(Q, root_prior, root_prior_probs)
    if vec === nothing
        return fill(1.0 / size(Q, 1), size(Q, 1))
    end
    return vec
end

function jointreconstruction(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior::Symbol,
    root_prior_probs = nothing,
)
    priors = _corhmm_validate_liks(tree, tip_priors)
    Qf = _validate_rate_matrix(Q)
    nstates = size(Qf, 1)
    evals, V, Vinv = _mk_eigen_cache(Qf)
    P = zeros(Float64, nstates, nstates)
    score = zeros(Float64, tree.nnodes, nstates)
    choice = zeros(Int32, tree.nnodes, nstates)

    for (row, node) in enumerate(tree.tip_ids)
        @inbounds for state in 1:nstates
            prior = priors[row, state]
            score[node, state] = prior > 0.0 ? log(prior) : -Inf
            choice[node, state] = Int32(state)
        end
    end

    for node in tree.postorder_internal
        for parent_state in 1:nstates
            total = 0.0
            for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
                child = tree.child_of_edge[edge]
                _fill_transition_matrix!(P, tree.edge_length[edge], evals, V, Vinv)
                best_state = 1
                best_val = -Inf
                for child_state in 1:nstates
                    transition = P[parent_state, child_state]
                    transition <= 0.0 && continue
                    child_val = log(transition) + score[child, child_state]
                    if child_val > best_val
                        best_val = child_val
                        best_state = child_state
                    end
                end
                total += best_val
                choice[child, parent_state] = Int32(best_state)
            end
            score[node, parent_state] = total
            choice[node, parent_state] = Int32(parent_state)
        end
    end

    root = tree.root
    root_prior_vec = _corhmm_root_vector(root_prior, Qf, @view score[root, :])
    best_root = 1
    best_loglik = -Inf
    for state in 1:nstates
        prior = root_prior_vec[state]
        prior <= 0.0 && continue
        val = log(prior) + score[root, state]
        if val > best_loglik
            best_loglik = val
            best_root = state
        end
    end

    node_states = zeros(Int32, tree.nnodes)
    node_states[root] = Int32(best_root)
    for node in tree.preorder
        tree.is_tip[node] && continue
        parent_state = Int(node_states[node])
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = tree.child_of_edge[edge]
            node_states[child] = choice[child, parent_state]
        end
    end

    internal_ids = Int[n for n in tree.preorder if !tree.is_tip[n]]
    tip_states = Int32[node_states[node] for node in tree.tip_ids]
    internal_states = Int32[node_states[node] for node in internal_ids]
    return (
        node_ids = internal_ids,
        tip_states = tip_states,
        internal_states = internal_states,
        joint_loglik = best_loglik,
    )
end

_corhmm_joint_states_from_marginal(args...; kwargs...) = jointstates(args...; kwargs...)
_corhmm_root_prior_vector_for_joint(args...; kwargs...) = jointrootprior(args...; kwargs...)
_corhmm_pupko_joint_reconstruction(args...; kwargs...) = jointreconstruction(args...; kwargs...)
