@inline function _sample_categorical!(rng::AbstractRNG, probs::AbstractVector{Float64})
    total = 0.0
    @inbounds for i in eachindex(probs)
        total += probs[i]
    end
    total > 0.0 || return Int32(1)
    u = rand(rng) * total
    acc = 0.0
    @inbounds for i in eachindex(probs)
        acc += probs[i]
        if u <= acc
            return Int32(i)
        end
    end
    return Int32(length(probs))
end

"""
    sample_mk_endpoints(tree, tip_priors, Q; kwargs...)

Sample the root state together with start and end states for every branch under
the fitted Mk process. This is the endpoint-sampling stage that powers
stochastic mapping.
"""
function sample_mk_endpoints(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior::Symbol = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    nparams::Union{Nothing, Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    cache = mk_pruning_cache(tree, tip_priors, Q; root_prior=root_prior, root_prior_probs=root_prior_probs, nparams=nparams)
    cache.success || return MkEndpointSample(success=false, nstates=cache.nstates, loglik=cache.loglik)

    Qf = cache.transition_matrix
    nstates = cache.nstates
    root_prior_vec = _mk_effective_root_prior(tree, cache; root_prior = root_prior, root_prior_probs = root_prior_probs)
    evals, V, Vinv = _mk_eigen_cache(Qf)
    node_states = fill(Int32(0), tree.nnodes)
    edge_start_states = fill(Int32(0), tree.nedges)
    edge_end_states = fill(Int32(0), tree.nedges)
    probs = zeros(Float64, nstates)
    P = zeros(Float64, nstates, nstates)

    root = tree.root
    if tree.is_tip[root]
        @views probs .= cache.node_priors[root, :]
    else
        @inbounds for s in 1:nstates
            probs[s] = exp(cache.logpost[root, s])
        end
    end
    @inbounds for s in 1:nstates
        probs[s] *= root_prior_vec[s]
    end
    node_states[root] = _sample_categorical!(rng, probs)

    for node in tree.preorder
        tree.is_tip[node] && continue
        parent_state = node_states[node]
        first_edge = tree.first_child_edge[node]
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            _fill_transition_matrix!(P, tree.edge_length[edge], evals, V, Vinv)
            @inbounds for s in 1:nstates
                evidence = tree.is_tip[child] ? cache.node_priors[child, s] : exp(cache.logpost[child, s])
                probs[s] = P[parent_state, s] * evidence
            end
            child_state = _sample_categorical!(rng, probs)
            node_states[child] = child_state
            edge_start_states[edge] = parent_state
            edge_end_states[edge] = child_state
        end
    end

    return MkEndpointSample(
        success = true,
        nstates = nstates,
        root_state = node_states[root],
        node_states = node_states,
        edge_start_states = edge_start_states,
        edge_end_states = edge_end_states,
        loglik = cache.loglik,
    )
end

function _sample_mk_endpoints_from_cache(
    tree::CompactTree,
    cache::MkPruningCache,
    evals::AbstractVector{<:Number},
    V::Matrix{ComplexF64},
    Vinv::Matrix{ComplexF64};
    root_prior::Symbol = cache.root_prior,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    cache.success || return MkEndpointSample(success = false, nstates = cache.nstates, loglik = cache.loglik)

    Qf = cache.transition_matrix
    nstates = cache.nstates
    root_prior_vec = _mk_effective_root_prior(tree, cache; root_prior = root_prior, root_prior_probs = root_prior_probs)
    node_states = fill(Int32(0), tree.nnodes)
    edge_start_states = fill(Int32(0), tree.nedges)
    edge_end_states = fill(Int32(0), tree.nedges)
    probs = zeros(Float64, nstates)
    P = zeros(Float64, nstates, nstates)

    root = tree.root
    if tree.is_tip[root]
        @views probs .= cache.node_priors[root, :]
    else
        @inbounds for s in 1:nstates
            probs[s] = exp(cache.logpost[root, s])
        end
    end
    @inbounds for s in 1:nstates
        probs[s] *= root_prior_vec[s]
    end
    node_states[root] = _sample_categorical!(rng, probs)

    for node in tree.preorder
        tree.is_tip[node] && continue
        parent_state = node_states[node]
        first_edge = tree.first_child_edge[node]
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            _fill_transition_matrix!(P, tree.edge_length[edge], evals, V, Vinv)
            @inbounds for s in 1:nstates
                evidence = tree.is_tip[child] ? cache.node_priors[child, s] : exp(cache.logpost[child, s])
                probs[s] = P[parent_state, s] * evidence
            end
            child_state = _sample_categorical!(rng, probs)
            node_states[child] = child_state
            edge_start_states[edge] = parent_state
            edge_end_states[edge] = child_state
        end
    end

    return MkEndpointSample(
        success = true,
        nstates = nstates,
        root_state = node_states[root],
        node_states = node_states,
        edge_start_states = edge_start_states,
        edge_end_states = edge_end_states,
        loglik = cache.loglik,
    )
end
