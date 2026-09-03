function _ou_segments_affine(
    spec::OUSpec,
    bundle::OUParameterBundle,
    segments::Vector{SimmapSegment},
)
    a = 1.0
    b = 0.0
    v = 0.0
    for seg in segments
        state = Int(seg.state)
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, state)
        sigma2 = _ou_regime_value(spec.sigma_mode, bundle.sigma2, state)
        theta = _ou_regime_value(spec.theta_mode, bundle.theta, state)
        phi = exp(-alpha * seg.length)
        a = phi * a
        b = phi * b + (1.0 - phi) * theta
        v = phi^2 * v + sigma2 * (1.0 - phi^2) / (2.0 * alpha)
    end
    return (a = a, b = b, v = v)
end

function _split_segment_point_segments(segments::Vector{SimmapSegment}, idx::Int, frac::Float64)
    prefix = SimmapSegment[]
    suffix = SimmapSegment[]
    for j in 1:(idx - 1)
        push!(prefix, segments[j])
    end
    split_len = frac * segments[idx].length
    remain_len = segments[idx].length - split_len
    split_len > 0.0 && push!(prefix, SimmapSegment(state = segments[idx].state, length = split_len))
    remain_len > 0.0 && push!(suffix, SimmapSegment(state = segments[idx].state, length = remain_len))
    for j in (idx + 1):length(segments)
        push!(suffix, segments[j])
    end
    return prefix, suffix
end

@inline function _branch_segment_affine(transform::Symbol, seg_start::Float64, seg_stop::Float64, seg::SimmapSegment, fit::ContinuousMultiRegimeResult, depth::Float64)
    state = Int(seg.state)
    sigma = Float64(fit.sigma2[state])
    if transform === :eb
        beta = Float64(fit.beta_regimes[state])
        return (a = 1.0, b = 0.0, v = _eb_branch_variance_increment(seg_start, seg_stop, sigma, beta))
    end
    throw(ArgumentError("Unsupported branch segment transform $transform"))
end

function _compose_affine_segments(parts)
    a = 1.0
    b = 0.0
    v = 0.0
    for part in parts
        a = part.a * a
        b = part.a * b + part.b
        v = part.a^2 * v + part.v
    end
    return (a = a, b = b, v = v)
end

function _bm_regime_segments_affine(
    tree::CompactTree,
    edge::Int,
    segments::Vector{SimmapSegment},
    fit::ContinuousMultiRegimeResult,
    transform::Symbol,
)
    parent = Int(tree.parent_of_edge[edge])
    seg_time = tree.dist_from_root[parent]
    parts = Vector{NamedTuple{(:a, :b, :v), Tuple{Float64, Float64, Float64}}}(undef, length(segments))
    depth = maximum(tree.dist_from_root[tree.tip_ids])
    for i in eachindex(segments)
        seg = segments[i]
        seg_stop = seg_time + seg.length
        parts[i] = _branch_segment_affine(transform, seg_time, seg_stop, seg, fit, depth)
        seg_time = seg_stop
    end
    return _compose_affine_segments(parts)
end

"""
    estim_branch_for_simmap(tree, trait, fit; edge_segments)

Perform simmap-aware branch posterior reconstruction for a fitted single-trait
multi-regime continuous model. The current implementation returns posterior
means, variances, and standard errors at the start, midpoint, and end of each
mapped branch segment.
"""
function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    fit::ContinuousMultiRegimeResult,
    simmap::SimmapSample,
)
    return estim_branch_for_simmap(tree, trait, fit; edge_segments = simmap.edge_segments)
end

function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    fit::ContinuousMultiRegimeResult;
    edge_segments::Vector{Vector{SimmapSegment}},
)
    fit.success || throw(ArgumentError("estim_branch_for_simmap requires a successful fitted model"))
    fit.model in (:EBM, :OUM, :OUMV, :OUMA, :OUMVA) ||
        throw(ArgumentError("estim_branch_for_simmap currently supports EBM, OUM, OUMV, OUMA, and OUMVA"))
    _validate_ultrametric_tree(tree)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    tip_index = zeros(Int, tree.nnodes)
    for (i, node) in enumerate(tree.tip_ids)
        tip_index[node] = i
    end
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    stationary_design = fit.root_mean_mode === :stationary_design
    stationary_node_means = stationary_design ? _ou_stationary_design_node_means(tree, ou_spec(fit.model), OUParameterBundle(theta = fit.theta_regimes, alpha = isempty(fit.alpha_regimes) ? [fit.alpha] : fit.alpha_regimes, sigma2 = fit.sigma2), cache) : nothing
    post =
        if fit.model in (:OUM, :OUMV, :OUMA, :OUMVA)
            spec = ou_spec(
                fit.model;
                root_mean_mode = fit.root_mean_mode,
                root_cov_mode = fit.root_cov_mode,
            )
            alpha_values = isempty(fit.alpha_regimes) ? [fit.alpha] : fit.alpha_regimes
            bundle = OUParameterBundle(theta = fit.theta_regimes, alpha = alpha_values, sigma2 = fit.sigma2)
            edges = _build_ou_edges(tree, spec, bundle; cache = cache)
            root = _ou_root_prior(spec, bundle; cache = cache)
            post_trait = stationary_design ? copy(tr) : tr
            if stationary_design
                for (i, tip) in enumerate(tree.tip_ids)
                    post_trait[i] -= stationary_node_means[Int(tip)]
                end
                fill!(edges.edge_b, 0.0)
            end
            _linear_gaussian_posterior_cache(
                tree,
                post_trait,
                edges.edge_a,
                edges.edge_b,
                edges.edge_v;
                root_prior_mean = root.mean,
                root_prior_var = root.var,
            )
        elseif fit.model === :EBM
            edge_v = _ebm_edge_variances(tree, edge_segments, fit.sigma2, fit.beta_regimes)
            _linear_gaussian_posterior_cache(tree, trait, ones(Float64, tree.nedges), zeros(Float64, tree.nedges), edge_v)
        end
    post.success || return ContinuousBranchPosteriorResult(model = fit.model, success = false)

    edge_ids = Int32[]
    parent_nodes = Int32[]
    child_nodes = Int32[]
    segment_indices = Int32[]
    segment_states = Int32[]
    branch_start = Float64[]
    branch_end = Float64[]
    branch_midpoint = Float64[]
    absolute_start = Float64[]
    absolute_end = Float64[]
    absolute_midpoint = Float64[]
    time_before_present_start = Float64[]
    time_before_present_end = Float64[]
    time_before_present_midpoint = Float64[]
    start_estimates = Float64[]
    midpoint_estimates = Float64[]
    end_estimates = Float64[]
    start_variances = Float64[]
    midpoint_variances = Float64[]
    end_variances = Float64[]

    for edge in 1:tree.nedges
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        segments = edge_segments[edge]
        cum = 0.0
        for seg_idx in 1:length(segments)
            seg = segments[seg_idx]
            seg_start = cum
            seg_end = cum + seg.length
            seg_mid = 0.5 * (seg_start + seg_end)
            root_dist = tree.dist_from_root[parent]
            tree_height = maximum(tree.dist_from_root[tree.tip_ids])

            local function _segment_affines(frac::Float64)
                prefix, suffix = _split_segment_point_segments(segments, seg_idx, frac)
                prefix_aff =
                    if fit.model in (:OUM, :OUMV, :OUMA, :OUMVA)
                        spec = ou_spec(
                            fit.model;
                            root_mean_mode = fit.root_mean_mode,
                            root_cov_mode = fit.root_cov_mode,
                        )
                        alpha_values = isempty(fit.alpha_regimes) ? [fit.alpha] : fit.alpha_regimes
                        bundle = OUParameterBundle(theta = fit.theta_regimes, alpha = alpha_values, sigma2 = fit.sigma2)
                        _ou_segments_affine(spec, bundle, prefix)
                    elseif fit.model === :EBM
                        _bm_regime_segments_affine(tree, edge, prefix, fit, :eb)
                    end
                suffix_aff =
                    if fit.model in (:OUM, :OUMV, :OUMA, :OUMVA)
                        spec = ou_spec(
                            fit.model;
                            root_mean_mode = fit.root_mean_mode,
                            root_cov_mode = fit.root_cov_mode,
                        )
                        alpha_values = isempty(fit.alpha_regimes) ? [fit.alpha] : fit.alpha_regimes
                        bundle = OUParameterBundle(theta = fit.theta_regimes, alpha = alpha_values, sigma2 = fit.sigma2)
                        _ou_segments_affine(spec, bundle, suffix)
                    elseif fit.model === :EBM
                        _bm_regime_segments_affine(tree, edge, suffix, fit, :eb)
                    end
                return prefix_aff, suffix_aff
            end

            local function _segment_point_posterior(frac::Float64)
                prefix_aff, suffix_aff = _segment_affines(frac)
                prefix_transition = stationary_design ? (a = prefix_aff.a, b = 0.0, v = prefix_aff.v) : prefix_aff
                suffix_transition = stationary_design ? (a = suffix_aff.a, b = 0.0, v = suffix_aff.v) : suffix_aff
                forward = _edge_predict_to_child(
                    post.edge_context_mean[edge],
                    post.edge_context_var[edge],
                    prefix_transition.a,
                    prefix_transition.b,
                    prefix_transition.v,
                )
                backward_info = _descendant_message(
                    post,
                    tr,
                    tip_index,
                    tree,
                    child,
                    suffix_transition.a,
                    suffix_transition.b,
                    suffix_transition.v,
                )
                backward_info.success || return (mean = NaN, var = NaN)
                backward = _information_to_gaussian(backward_info.precision, backward_info.linear)
                point = _gaussian_product(forward.mean, forward.var, backward.mean, backward.var)
                if stationary_design
                    point_mean = prefix_aff.a * stationary_node_means[parent] + prefix_aff.b
                    point = (mean = point.mean + point_mean, var = point.var)
                end
                return point
            end

            start_post = _segment_point_posterior(0.0)
            mid_post = _segment_point_posterior(0.5)
            end_post = _segment_point_posterior(1.0)
            all(isfinite, (start_post.mean, start_post.var, mid_post.mean, mid_post.var, end_post.mean, end_post.var)) ||
                return ContinuousBranchPosteriorResult(model = fit.model, success = false)

            push!(edge_ids, Int32(edge))
            push!(parent_nodes, Int32(parent))
            push!(child_nodes, Int32(child))
            push!(segment_indices, Int32(seg_idx))
            push!(segment_states, seg.state)
            push!(branch_start, seg_start)
            push!(branch_end, seg_end)
            push!(branch_midpoint, seg_mid)
            push!(absolute_start, root_dist + seg_start)
            push!(absolute_end, root_dist + seg_end)
            push!(absolute_midpoint, root_dist + seg_mid)
            push!(time_before_present_start, tree_height - (root_dist + seg_start))
            push!(time_before_present_end, tree_height - (root_dist + seg_end))
            push!(time_before_present_midpoint, tree_height - (root_dist + seg_mid))
            push!(start_estimates, start_post.mean)
            push!(midpoint_estimates, mid_post.mean)
            push!(end_estimates, end_post.mean)
            push!(start_variances, start_post.var)
            push!(midpoint_variances, mid_post.var)
            push!(end_variances, end_post.var)

            cum = seg_end
        end
    end

    return ContinuousBranchPosteriorResult(
        model = fit.model,
        success = true,
        edge_ids = edge_ids,
        parent_nodes = parent_nodes,
        child_nodes = child_nodes,
        segment_indices = segment_indices,
        segment_states = segment_states,
        branch_start = branch_start,
        branch_end = branch_end,
        branch_midpoint = branch_midpoint,
        time_from_root_start = absolute_start,
        time_from_root_end = absolute_end,
        time_from_root_midpoint = absolute_midpoint,
        time_before_present_start = time_before_present_start,
        time_before_present_end = time_before_present_end,
        time_before_present_midpoint = time_before_present_midpoint,
        absolute_start = absolute_start,
        absolute_end = absolute_end,
        absolute_midpoint = absolute_midpoint,
        start_estimates = start_estimates,
        midpoint_estimates = midpoint_estimates,
        end_estimates = end_estimates,
        start_variances = start_variances,
        midpoint_variances = midpoint_variances,
        end_variances = end_variances,
        start_se = sqrt.(start_variances),
        midpoint_se = sqrt.(midpoint_variances),
        end_se = sqrt.(end_variances),
    )
end
