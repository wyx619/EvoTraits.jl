function normalizerows!(X::Matrix{Float64})
    for i in axes(X, 1)
        s = sum(@view X[i, :])
        s > 0.0 || continue
        @views X[i, :] ./= s
    end
    return X
end

function scaledliks(tree::CompactTree, cache::CorHMMPruningCache, node_ids::AbstractVector{<:Integer})
    out = zeros(Float64, length(node_ids), cache.nstates)
    for (i, node) in enumerate(node_ids)
        @views out[i, :] .= cache.node_liks[node, :]
    end
    return normalizerows!(out)
end

function internalnodes(tree::CompactTree)
    return Int[n for n in tree.preorder if !tree.is_tip[n]]
end

function fitbranchlengths(fit::CorHMMFitResult)
    isempty(fit.branch_lengths) ? _corhmm_branch_lengths(fit.tree) : fit.branch_lengths
end

function tipliks(tree::CompactTree, cache::CorHMMPruningCache)
    out = zeros(Float64, tree.ntips, cache.nstates)
    for (i, node) in enumerate(tree.tip_ids)
        @views out[i, :] .= cache.node_liks[node, :]
    end
    return normalizerows!(out)
end

function rtipstates(fit::CorHMMFitResult, mode::Symbol)
    if mode === :none
        return zeros(0, 0)
    elseif mode === :joint
        out = zeros(Float64, size(fit.tip_priors_hidden))
        for i in axes(out, 1)
            states = findall(>(0.0), @view fit.tip_priors_hidden[i, :])
            if length(states) == 1
                out[i, only(states)] = 1.0
            else
                @views out[i, :] .= fit.tip_priors_hidden[i, :]
            end
        end
        return out
    else
        return copy(fit.tip_priors_hidden)
    end
end

function marginalliks(
    tree::CompactTree,
    cache::CorHMMPruningCache,
    branch_lengths::AbstractVector{<:Real},
)
    nstates = cache.nstates
    Q = cache.transition_matrix
    evals, V, Vinv = _mk_eigen_cache(Q)
    P = zeros(Float64, nstates, nstates)
    up = zeros(Float64, tree.nnodes, nstates)
    root = tree.root
    @views up[root, :] .= cache.root_prior_probs
    root_sum = sum(@view up[root, :])
    root_sum > 0.0 && (@views up[root, :] ./= root_sum)

    tmp_parent = zeros(Float64, nstates)
    sister_msg = zeros(Float64, nstates)
    for node in tree.preorder
        tree.is_tip[node] && continue
        first_edge = tree.first_child_edge[node]
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            @views tmp_parent .= up[node, :]
            for sister_edge in first_edge:last_edge
                sister_edge == edge && continue
                sister = tree.child_of_edge[sister_edge]
                _fill_transition_matrix!(P, branch_lengths[sister_edge], evals, V, Vinv)
                mul!(sister_msg, P, @view cache.node_liks[sister, :])
                tmp_parent .*= sister_msg
            end
            _fill_transition_matrix!(P, branch_lengths[edge], evals, V, Vinv)
            mul!(@view(up[child, :]), transpose(P), tmp_parent)
            s = sum(@view up[child, :])
            s > 0.0 && (@views up[child, :] ./= s)
        end
    end

    node_ids = internalnodes(tree)
    out = zeros(Float64, length(node_ids), nstates)
    for (i, node) in enumerate(node_ids)
        @views out[i, :] .= cache.node_liks[node, :] .* up[node, :]
    end
    return normalizerows!(out)
end

_normalize_rows!(args...; kwargs...) = normalizerows!(args...; kwargs...)
_corhmm_scaled_ancestral_likelihoods(args...; kwargs...) = scaledliks(args...; kwargs...)
_corhmm_internal_node_ids(args...; kwargs...) = internalnodes(args...; kwargs...)
_corhmm_fit_branch_lengths(args...; kwargs...) = fitbranchlengths(args...; kwargs...)
_corhmm_tip_likelihoods(args...; kwargs...) = tipliks(args...; kwargs...)
_corhmm_r_tip_states(args...; kwargs...) = rtipstates(args...; kwargs...)
_corhmm_marginal_ancestral_likelihoods(args...; kwargs...) = marginalliks(args...; kwargs...)
