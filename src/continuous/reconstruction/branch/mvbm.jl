function _mvbm_segments_kernel(
    model::Symbol,
    fit,
    segments::Vector{SimmapSegment},
    seg_start::Float64,
)
    p = fit.ntraits
    isempty(segments) && return (Phi = Matrix{Float64}(I, p, p), Q = zeros(Float64, p, p))
    if model === :mvBM1
        q = zeros(Float64, p, p)
        for seg in segments
            q .+= seg.length .* fit.sigma
        end
        return (Phi = Matrix{Float64}(I, p, p), Q = q)
    elseif model === :mvEB
        q = zeros(Float64, p, p)
        t = seg_start
        for seg in segments
            next_t = t + seg.length
            inc = _eb_branch_variance_increment(t, next_t, 1.0, fit.beta)
            q .+= inc .* fit.sigma
            t = next_t
        end
        return (Phi = Matrix{Float64}(I, p, p), Q = q)
    elseif model === :mvBMM
        q = zeros(Float64, p, p)
        for seg in segments
            q .+= seg.length .* fit.sigma[:, :, Int(seg.state)]
        end
        return (Phi = Matrix{Float64}(I, p, p), Q = q)
    end
    throw(ArgumentError("Unsupported multivariate BM branch kernel model $model"))
end

function _mvbm_segment_increment(
    model::Symbol,
    fit,
    seg::SimmapSegment,
    root_dist::Float64,
    seg_start::Float64,
    frac::Float64,
)
    p = fit.ntraits
    len = clamp(frac, 0.0, 1.0) * seg.length
    q = zeros(Float64, p, p)
    if len <= 0.0
        return q
    elseif model === :mvBM1
        q .+= len .* fit.sigma
    elseif model === :mvEB
        t0 = root_dist + seg_start
        inc = _eb_branch_variance_increment(t0, t0 + len, 1.0, fit.beta)
        q .+= inc .* fit.sigma
    elseif model === :mvBMM
        q .+= len .* fit.sigma[:, :, Int(seg.state)]
    else
        throw(ArgumentError("Unsupported multivariate BM branch kernel model $model"))
    end
    return (q + q') / 2
end

function _mvbm_segment_prefix_covariances(
    model::Symbol,
    fit,
    segments::Vector{SimmapSegment},
    root_dist::Float64,
)
    p = fit.ntraits
    prefix = [zeros(Float64, p, p) for _ in 1:(length(segments) + 1)]
    cum = 0.0
    for i in eachindex(segments)
        prefix[i + 1] .= prefix[i] .+ _mvbm_segment_increment(model, fit, segments[i], root_dist, cum, 1.0)
        cum += segments[i].length
    end
    return prefix
end

function _mvbm_branch_precalc(
    tree::CompactTree,
    fit::Union{MVContinuousBMResult, MVContinuousMultiBMResult},
    edge_segments::Vector{Vector{SimmapSegment}},
    p::Integer,
)
    edge_Phi = Array{Float64, 3}(undef, p, p, tree.nedges)
    edge_Q = Array{Float64, 3}(undef, p, p, tree.nedges)
    I_p = Matrix{Float64}(I, p, p)
    for edge in 1:tree.nedges
        edge_Phi[:, :, edge] .= I_p
        if fit.model === :mvBMM
            kernel = _mvbm_segments_kernel(fit.model, fit, edge_segments[edge], 0.0)
            edge_Q[:, :, edge] .= kernel.Q
        else
            parent = tree.parent_of_edge[edge]
            root_dist = tree.dist_from_root[parent]
            segments = edge_segments[edge]
            edge_Q[:, :, edge] .= zeros(Float64, p, p)
            if fit.model === :mvBM1
                for seg in segments
                    edge_Q[:, :, edge] .+= seg.length .* fit.sigma
                end
            else
                t = root_dist
                for seg in segments
                    next_t = t + seg.length
                    inc = _eb_branch_variance_increment(t, next_t, 1.0, fit.beta)
                    edge_Q[:, :, edge] .+= inc .* fit.sigma
                    t = next_t
                end
            end
        end
    end
    return edge_Phi, edge_Q
end

function _mvbm_shift_vector(fit::MVContinuousBMResult)
    return fit.model === :mvBM1 ? Vector{Float64}(fit.theta) : Vector{Float64}(vec(fit.theta))
end

function _mvbm_shift_vector(fit::MVContinuousMultiBMResult)
    return Vector{Float64}(vec(fit.theta))
end

function _mvbm_branch_posterior(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::Union{MVContinuousBMResult, MVContinuousMultiBMResult};
    edge_segments::Vector{Vector{SimmapSegment}},
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    edge_Phi, edge_Q = _mvbm_branch_precalc(tree, fit, edge_segments, p)
    shift_vec = _mvbm_shift_vector(fit)
    centered = data .- reshape(shift_vec, 1, p)
    post = _mv_recursive_posterior_cache(
        tree,
        centered,
        edge_Phi,
        edge_Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
    )
    post.success || return (success = false,)
    return (success = true, post = post, shift_vec = shift_vec, p = p)
end
