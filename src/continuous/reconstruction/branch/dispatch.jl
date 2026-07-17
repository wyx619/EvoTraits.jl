"""
    estim_branch_for_simmap(tree, trait, fit; edge_segments)

Perform simmap-aware branch posterior reconstruction for fitted multivariate
OU-family models. The current implementation returns posterior means and
covariances at the start, midpoint, and end of each mapped branch segment.
"""
function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult,
    simmap::SimmapSample,
)
    return estim_branch_for_simmap(tree, trait, fit; edge_segments = simmap.edge_segments)
end

function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult;
    edge_segments::Vector{Vector{SimmapSegment}},
)
    fit.success || throw(ArgumentError("estim_branch_for_simmap requires a successful fitted model"))
    fit.model in (:mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA) ||
        throw(ArgumentError("estim_branch_for_simmap currently supports mvOUM, mvOUMV, mvOUMA, and mvOUMVA"))

    prep = _mvou_branch_posterior(tree, trait, fit; edge_segments = edge_segments)
    prep.success || return MVContinuousBranchPosteriorResult(model = fit.model, success = false)
    post = prep.post
    bundle = prep.bundle
    node_means = prep.node_means
    p = prep.p

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
    start_estimates = Vector{Vector{Float64}}()
    midpoint_estimates = Vector{Vector{Float64}}()
    end_estimates = Vector{Vector{Float64}}()
    start_covariances = Vector{Matrix{Float64}}()
    midpoint_covariances = Vector{Matrix{Float64}}()
    end_covariances = Vector{Matrix{Float64}}()

    for edge in 1:tree.nedges
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        segments = edge_segments[edge]
        start_node_mean = node_means[parent]
        kernel_cache = _mvou_edge_kernel_cache(fit.model, bundle, segments, p)
        mean_prefix = _mvou_edge_mean_prefix_cache(fit.model, start_node_mean, bundle, segments)
        cum = 0.0
        for seg_idx in eachindex(segments)
            seg = segments[seg_idx]
            seg_start = cum
            seg_end = cum + seg.length
            seg_mid = 0.5 * (seg_start + seg_end)
            root_dist = tree.dist_from_root[parent]
            tree_height = maximum(tree.dist_from_root[tree.tip_ids])

            local function _point(frac::Float64)
                split_len = clamp(frac, 0.0, 1.0) * seg.length
                remain_len = seg.length - split_len
                prefix_kernel =
                    split_len <= 0.0 ?
                    kernel_cache.prefix[seg_idx] :
                    _mv_kernel_compose(
                        kernel_cache.prefix[seg_idx],
                        _mvou_single_segment_kernel(
                            fit.model,
                            bundle,
                            SimmapSegment(state = seg.state, length = split_len),
                            p,
                        ),
                    )
                suffix_kernel =
                    remain_len <= 0.0 ?
                    kernel_cache.suffix[seg_idx + 1] :
                    _mv_kernel_compose(
                        _mvou_single_segment_kernel(
                            fit.model,
                            bundle,
                            SimmapSegment(state = seg.state, length = remain_len),
                            p,
                        ),
                        kernel_cache.suffix[seg_idx + 1],
                    )
                forward = _mvou_edge_predict_to_child(
                    post.edge_context_mean[edge],
                    post.edge_context_cov[edge],
                    prefix_kernel.Phi,
                    prefix_kernel.Q,
                )
                backward =
                    maximum(abs, suffix_kernel.Q) <= 1e-12 && maximum(abs, suffix_kernel.Phi - Matrix{Float64}(I, p, p)) <= 1e-12 ?
                    (
                        success = true,
                        precision = post.precision[child],
                        linear = post.linear[child],
                        logconst = post.logconst[child],
                    ) :
                    _mvou_info_to_parent(
                        post.precision[child],
                        post.linear[child],
                        post.logconst[child],
                        suffix_kernel.Phi,
                        suffix_kernel.Q,
                    )
                backward.success || return (success = false, mean = zeros(Float64, p), cov = zeros(Float64, p, p))
                combined = _mvou_context_with_info(forward.mean, forward.cov, backward.precision, backward.linear)
                eval_mean =
                    split_len <= 0.0 ?
                    mean_prefix[seg_idx] :
                    _mvou_segment_eval_mean(
                        fit.model,
                        mean_prefix[seg_idx],
                        bundle,
                        SimmapSegment[SimmapSegment(state = seg.state, length = split_len)],
                    )
                return (success = combined.success, mean = combined.mean .+ eval_mean, cov = combined.cov)
            end

            start_post = _point(0.0)
            mid_post = _point(0.5)
            end_post = _point(1.0)
            (start_post.success && mid_post.success && end_post.success) ||
                return MVContinuousBranchPosteriorResult(model = fit.model, success = false)

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
            _mvbranch_push_point!(start_estimates, start_covariances, start_post.mean, start_post.cov)
            _mvbranch_push_point!(midpoint_estimates, midpoint_covariances, mid_post.mean, mid_post.cov)
            _mvbranch_push_point!(end_estimates, end_covariances, end_post.mean, end_post.cov)

            cum = seg_end
        end
    end

    return MVContinuousBranchPosteriorResult(
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
        start_estimates = _mvbranch_pack_means(start_estimates, p),
        midpoint_estimates = _mvbranch_pack_means(midpoint_estimates, p),
        end_estimates = _mvbranch_pack_means(end_estimates, p),
        start_covariances = _mvbranch_pack_covariances(start_covariances, p),
        midpoint_covariances = _mvbranch_pack_covariances(midpoint_covariances, p),
        end_covariances = _mvbranch_pack_covariances(end_covariances, p),
    )
end

"""
    estim_branch_for_simmap(tree, trait, fit; edge_segments)

Simmap-aware branch posterior reconstruction for fitted multivariate BM-family
models. The current implementation returns posterior means and covariances at
the start, midpoint, and end of each mapped branch segment.
"""
function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousBMResult,
    simmap::SimmapSample,
)
    return estim_branch_for_simmap(tree, trait, fit; edge_segments = simmap.edge_segments)
end

function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousBMResult;
    edge_segments::Vector{Vector{SimmapSegment}},
)
    fit.success || throw(ArgumentError("estim_branch_for_simmap requires a successful fitted model"))
    fit.model in (:mvBM1, :mvEB) ||
        throw(ArgumentError("estim_branch_for_simmap currently supports mvBM1 and mvEB"))

    prep = _mvbm_branch_posterior(tree, trait, fit; edge_segments = edge_segments)
    prep.success || return MVContinuousBranchPosteriorResult(model = fit.model, success = false)
    post = prep.post
    shift_vec = prep.shift_vec
    p = prep.p

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
    start_estimates = Vector{Vector{Float64}}()
    midpoint_estimates = Vector{Vector{Float64}}()
    end_estimates = Vector{Vector{Float64}}()
    start_covariances = Vector{Matrix{Float64}}()
    midpoint_covariances = Vector{Matrix{Float64}}()
    end_covariances = Vector{Matrix{Float64}}()

    for edge in 1:tree.nedges
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        segments = edge_segments[edge]
        root_dist = tree.dist_from_root[parent]
        tree_height = maximum(tree.dist_from_root[tree.tip_ids])
        prefix_Q = _mvbm_segment_prefix_covariances(fit.model, fit, segments, root_dist)
        total_Q = prefix_Q[end]
        cum = 0.0
        for seg_idx in eachindex(segments)
            seg = segments[seg_idx]
            seg_start = cum
            seg_end = cum + seg.length
            seg_mid = 0.5 * (seg_start + seg_end)

            local function _point(frac::Float64)
                prefix_kernel_Q = prefix_Q[seg_idx] .+ _mvbm_segment_increment(fit.model, fit, seg, root_dist, seg_start, frac)
                suffix_kernel_Q = total_Q .- prefix_kernel_Q
                forward = _mv_identity_predict_to_child(
                    post.edge_context_mean[edge],
                    post.edge_context_cov[edge],
                    prefix_kernel_Q,
                )
                backward =
                    maximum(abs, suffix_kernel_Q) <= 1e-12 ?
                    (
                        success = true,
                        precision = post.precision[child],
                        linear = post.linear[child],
                        logconst = post.logconst[child],
                    ) :
                    _mvou_info_to_parent_identity(
                        post.precision[child],
                        post.linear[child],
                        post.logconst[child],
                        suffix_kernel_Q,
                    )
                backward.success || return (success = false, mean = zeros(Float64, p), cov = zeros(Float64, p, p))
                combined = _mvou_context_with_info(forward.mean, forward.cov, backward.precision, backward.linear)
                return (success = combined.success, mean = combined.mean .+ shift_vec, cov = combined.cov)
            end

            start_post = _point(0.0)
            mid_post = _point(0.5)
            end_post = _point(1.0)
            (start_post.success && mid_post.success && end_post.success) ||
                return MVContinuousBranchPosteriorResult(model = fit.model, success = false)

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
            _mvbranch_push_point!(start_estimates, start_covariances, start_post.mean, start_post.cov)
            _mvbranch_push_point!(midpoint_estimates, midpoint_covariances, mid_post.mean, mid_post.cov)
            _mvbranch_push_point!(end_estimates, end_covariances, end_post.mean, end_post.cov)

            cum = seg_end
        end
    end

    return MVContinuousBranchPosteriorResult(
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
        start_estimates = _mvbranch_pack_means(start_estimates, p),
        midpoint_estimates = _mvbranch_pack_means(midpoint_estimates, p),
        end_estimates = _mvbranch_pack_means(end_estimates, p),
        start_covariances = _mvbranch_pack_covariances(start_covariances, p),
        midpoint_covariances = _mvbranch_pack_covariances(midpoint_covariances, p),
        end_covariances = _mvbranch_pack_covariances(end_covariances, p),
    )
end

function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousMultiBMResult,
    simmap::SimmapSample,
)
    return estim_branch_for_simmap(tree, trait, fit; edge_segments = simmap.edge_segments)
end

function estim_branch_for_simmap(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousMultiBMResult;
    edge_segments::Vector{Vector{SimmapSegment}},
)
    fit.success || throw(ArgumentError("estim_branch_for_simmap requires a successful fitted model"))
    fit.model === :mvBMM ||
        throw(ArgumentError("estim_branch_for_simmap currently supports mvBMM"))

    prep = _mvbm_branch_posterior(tree, trait, fit; edge_segments = edge_segments)
    prep.success || return MVContinuousBranchPosteriorResult(model = fit.model, success = false)
    post = prep.post
    shift_vec = prep.shift_vec
    p = prep.p

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
    start_estimates = Vector{Vector{Float64}}()
    midpoint_estimates = Vector{Vector{Float64}}()
    end_estimates = Vector{Vector{Float64}}()
    start_covariances = Vector{Matrix{Float64}}()
    midpoint_covariances = Vector{Matrix{Float64}}()
    end_covariances = Vector{Matrix{Float64}}()

    for edge in 1:tree.nedges
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        segments = edge_segments[edge]
        root_dist = tree.dist_from_root[parent]
        tree_height = maximum(tree.dist_from_root[tree.tip_ids])
        prefix_Q = _mvbm_segment_prefix_covariances(fit.model, fit, segments, root_dist)
        total_Q = prefix_Q[end]
        cum = 0.0
        for seg_idx in eachindex(segments)
            seg = segments[seg_idx]
            seg_start = cum
            seg_end = cum + seg.length
            seg_mid = 0.5 * (seg_start + seg_end)

            local function _point(frac::Float64)
                prefix_kernel_Q = prefix_Q[seg_idx] .+ _mvbm_segment_increment(fit.model, fit, seg, root_dist, seg_start, frac)
                suffix_kernel_Q = total_Q .- prefix_kernel_Q
                forward = _mv_identity_predict_to_child(
                    post.edge_context_mean[edge],
                    post.edge_context_cov[edge],
                    prefix_kernel_Q,
                )
                backward =
                    maximum(abs, suffix_kernel_Q) <= 1e-12 ?
                    (
                        success = true,
                        precision = post.precision[child],
                        linear = post.linear[child],
                        logconst = post.logconst[child],
                    ) :
                    _mvou_info_to_parent_identity(
                        post.precision[child],
                        post.linear[child],
                        post.logconst[child],
                        suffix_kernel_Q,
                    )
                backward.success || return (success = false, mean = zeros(Float64, p), cov = zeros(Float64, p, p))
                combined = _mvou_context_with_info(forward.mean, forward.cov, backward.precision, backward.linear)
                return (success = combined.success, mean = combined.mean .+ shift_vec, cov = combined.cov)
            end

            start_post = _point(0.0)
            mid_post = _point(0.5)
            end_post = _point(1.0)
            (start_post.success && mid_post.success && end_post.success) ||
                return MVContinuousBranchPosteriorResult(model = fit.model, success = false)

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
            _mvbranch_push_point!(start_estimates, start_covariances, start_post.mean, start_post.cov)
            _mvbranch_push_point!(midpoint_estimates, midpoint_covariances, mid_post.mean, mid_post.cov)
            _mvbranch_push_point!(end_estimates, end_covariances, end_post.mean, end_post.cov)

            cum = seg_end
        end
    end

    return MVContinuousBranchPosteriorResult(
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
        start_estimates = _mvbranch_pack_means(start_estimates, p),
        midpoint_estimates = _mvbranch_pack_means(midpoint_estimates, p),
        end_estimates = _mvbranch_pack_means(end_estimates, p),
        start_covariances = _mvbranch_pack_covariances(start_covariances, p),
        midpoint_covariances = _mvbranch_pack_covariances(midpoint_covariances, p),
        end_covariances = _mvbranch_pack_covariances(end_covariances, p),
    )
end
