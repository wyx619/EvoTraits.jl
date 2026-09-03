@inline function _allow_zero_internal_edge_mismatch(tree::CompactTree, edge::Integer, segsum::Float64; atol::Float64 = 1e-8)
    child = Int(tree.child_of_edge[edge])
    tree.is_tip[child] && return false
    tree.edge_length[edge] == 0.0 || return false
    return abs(segsum) <= 10 * atol
end

function _validate_edge_segments(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}}; atol::Float64 = 1e-8)
    length(edge_segments) == tree.nedges || throw(ArgumentError("edge_segments must have $(tree.nedges) entries"))
    max_state = 0
    for edge in 1:tree.nedges
        segs = edge_segments[edge]
        isempty(segs) && throw(ArgumentError("edge_segments[$edge] is empty"))
        segsum = 0.0
        for seg in segs
            isfinite(seg.length) && seg.length >= 0.0 || throw(ArgumentError("edge_segments[$edge] contains invalid segment length"))
            seg.state >= 1 || throw(ArgumentError("edge_segments[$edge] contains invalid regime state"))
            segsum += seg.length
            max_state = max(max_state, Int(seg.state))
        end
        if !(isapprox(segsum, tree.edge_length[edge]; atol = atol) || _allow_zero_internal_edge_mismatch(tree, edge, segsum; atol = atol))
            throw(ArgumentError("edge_segments[$edge] lengths do not sum to branch length"))
        end
    end
    return max_state
end

function _root_regime_from_edge_segments(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}})
    root = Int(tree.root)
    first_edge = Int(tree.first_child_edge[root])
    first_edge > 0 || throw(ArgumentError("Root must have outgoing edges"))
    regime = Int(edge_segments[first_edge][1].state)
    for edge in tree.first_child_edge[root]:tree.last_child_edge[root]
        Int(edge_segments[edge][1].state) == regime || throw(ArgumentError("Root outgoing edges do not share a consistent initial regime"))
    end
    return regime
end

function _prepare_oum_edge_cache(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}}; atol::Float64 = 1e-8)
    nregimes = _validate_edge_segments(tree, edge_segments; atol = atol)
    root_regime = _root_regime_from_edge_segments(tree, edge_segments)
    edge_first_segment = fill(Int32(0), tree.nedges)
    edge_last_segment = fill(Int32(0), tree.nedges)
    segment_states = Int32[]
    segment_lengths = Float64[]

    for edge in 1:tree.nedges
        segs = edge_segments[edge]
        edge_first_segment[edge] = Int32(length(segment_states) + 1)
        for seg in segs
            push!(segment_states, seg.state)
            push!(segment_lengths, seg.length)
        end
        edge_last_segment[edge] = Int32(length(segment_states))
    end

    return OUMEdgeSegmentCache(
        nregimes,
        root_regime,
        edge_first_segment,
        edge_last_segment,
        segment_states,
        segment_lengths,
    )
end

function _ou_stationary_design_node_weights(
    tree::CompactTree,
    spec::OUSpec,
    bundle::OUParameterBundle,
    cache::OUMEdgeSegmentCache,
)
    weights = [zeros(Float64, cache.nregimes) for _ in 1:tree.nnodes]
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            copyto!(weights[child], weights[node])
            first_seg = Int(cache.edge_first_segment[edge])
            last_seg = Int(cache.edge_last_segment[edge])
            for seg_idx in first_seg:last_seg
                state = Int(cache.segment_states[seg_idx])
                alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, state)
                phi = exp(-alpha * cache.segment_lengths[seg_idx])
                weights[child] .*= phi
                weights[child][state] += 1.0 - phi
            end
        end
    end

    return weights
end

function _ou_stationary_design_node_means(
    tree::CompactTree,
    spec::OUSpec,
    bundle::OUParameterBundle,
    cache::OUMEdgeSegmentCache,
)
    weights = _ou_stationary_design_node_weights(tree, spec, bundle, cache)
    means = zeros(Float64, tree.nnodes)
    for node in 1:tree.nnodes
        denom = sum(weights[node])
        means[node] = abs(denom) <= 1e-12 ? 0.0 : dot(weights[node], bundle.theta) / denom
    end
    return means
end

function _ou_stationary_design_edge_mean(
    parent_weights::AbstractVector{<:Real},
    spec::OUSpec,
    bundle::OUParameterBundle,
    cache::OUMEdgeSegmentCache,
    edge::Integer,
    segment_index::Integer,
    fraction::Float64,
)
    weights = Vector{Float64}(parent_weights)
    first_seg = Int(cache.edge_first_segment[edge])
    last_seg = Int(cache.edge_last_segment[edge])
    stop_seg = first_seg + Int(segment_index) - 1
    stop_seg <= last_seg || throw(ArgumentError("segment index is outside edge cache"))
    @inbounds for seg_idx in first_seg:stop_seg
        length = cache.segment_lengths[seg_idx]
        seg_fraction = seg_idx == stop_seg ? fraction : 1.0
        state = Int(cache.segment_states[seg_idx])
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, state)
        phi = exp(-alpha * (length * seg_fraction))
        weights .*= phi
        weights[state] += 1.0 - phi
    end
    denom = sum(weights)
    return abs(denom) <= 1e-12 ? 0.0 : dot(weights, bundle.theta) / denom
end

@inline function _ou_edge_affine(cache, edge::Integer, spec::OUSpec, bundle::OUParameterBundle)
    a = 1.0
    b = 0.0
    v = 0.0
    first_seg = Int(cache.edge_first_segment[edge])
    last_seg = Int(cache.edge_last_segment[edge])
    @inbounds for seg_idx in first_seg:last_seg
        state = Int(cache.segment_states[seg_idx])
        seg_length = cache.segment_lengths[seg_idx]
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, state)
        sigma2 = _ou_regime_value(spec.sigma_mode, bundle.sigma2, state)
        theta = _ou_regime_value(spec.theta_mode, bundle.theta, state)
        phi = exp(-alpha * seg_length)
        a = phi * a
        b = phi * b + (1.0 - phi) * theta
        v = phi^2 * v + sigma2 * (1.0 - phi^2) / (2.0 * alpha)
    end
    return (a = a, b = b, v = v)
end

function _build_ou_edges_oum!(
    tree::CompactTree,
    cache::OUMEdgeSegmentCache,
    bundle::OUParameterBundle,
    edge_a::Vector{Float64},
    edge_b::Vector{Float64},
    edge_v::Vector{Float64},
)
    alpha = bundle.alpha[1]
    sigma2 = bundle.sigma2[1]
    @inbounds for edge in 1:tree.nedges
        total_length = tree.edge_length[edge]
        a = exp(-alpha * total_length)
        edge_a[edge] = a
        edge_v[edge] = sigma2 * (1.0 - a * a) / (2.0 * alpha)

        b = 0.0
        first_seg = Int(cache.edge_first_segment[edge])
        last_seg = Int(cache.edge_last_segment[edge])
        for seg_idx in first_seg:last_seg
            phi = exp(-alpha * cache.segment_lengths[seg_idx])
            theta = bundle.theta[Int(cache.segment_states[seg_idx])]
            b = phi * b + (1.0 - phi) * theta
        end
        edge_b[edge] = b
    end
    return (edge_a = edge_a, edge_b = edge_b, edge_v = edge_v)
end

function _build_ou_edges(
    tree::CompactTree,
    spec::OUSpec,
    bundle::OUParameterBundle;
    cache = nothing,
    workspace::Union{Nothing,OULikelihoodWorkspace} = nothing,
)
    if workspace === nothing
        edge_a = zeros(Float64, tree.nedges)
        edge_b = zeros(Float64, tree.nedges)
        edge_v = zeros(Float64, tree.nedges)
    else
        edge_a = workspace.edge_a
        edge_b = workspace.edge_b
        edge_v = workspace.edge_v
    end

    if cache !== nothing && spec.model === :OUM
        return _build_ou_edges_oum!(tree, cache, bundle, edge_a, edge_b, edge_v)
    end

    if cache === nothing
        spec.model === :OU1 || throw(ArgumentError("cache is required for $(spec.model)"))
        alpha = bundle.alpha[1]
        sigma2 = bundle.sigma2[1]
        theta = bundle.theta[1]
        edge_a .= exp.(-alpha .* tree.edge_length)
        edge_b .= (1.0 .- edge_a) .* theta
        edge_v .= sigma2 .* (1.0 .- edge_a .^ 2) ./ (2.0 * alpha)
    else
        for edge in 1:tree.nedges
            aff = _ou_edge_affine(cache, edge, spec, bundle)
            edge_a[edge] = aff.a
            edge_b[edge] = aff.b
            edge_v[edge] = aff.v
        end
    end

    return (edge_a = edge_a, edge_b = edge_b, edge_v = edge_v)
end
