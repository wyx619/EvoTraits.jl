function _mvou_segments_kernel(
    model::Symbol,
    bundle::MVOUParameterBundle,
    segments::Vector{SimmapSegment},
    ;
    A_decomp::Symbol = :cholesky,
)
    p = isempty(bundle.A) ? size(bundle.A_regimes, 1) : size(bundle.A, 1)
    isempty(segments) && return _mvou_empty_kernel(p)

    if model === :mvOUM || model === :mvOU1
        precalc = MVOUPrecalc(branch_lengths = Float64[sum(seg.length for seg in segments)], A_decomp = A_decomp)
        branch = _mvou_branch_cache(precalc, bundle.A, bundle.Sigma)
        return (Phi = branch.Phi[:, :, 1], Q = branch.Q[:, :, 1])
    elseif model === :mvOUMV
        precalc = MVOUPrecalc(branch_lengths = Float64[sum(seg.length for seg in segments)], edge_segments = [segments], nregimes = size(bundle.Sigma_regimes, 3), A_decomp = A_decomp)
        branch = _mvoumv_branch_cache(precalc, bundle.A, bundle.Sigma_regimes)
        return (Phi = branch.Phi[:, :, 1], Q = branch.Q[:, :, 1])
    elseif model === :mvOUMA
        precalc = MVOUPrecalc(branch_lengths = Float64[sum(seg.length for seg in segments)], edge_segments = [segments], nregimes = size(bundle.A_regimes, 3), A_decomp = A_decomp)
        branch = _mvouma_branch_cache(precalc, bundle.A_regimes, bundle.Sigma)
        return (Phi = branch.Phi[:, :, 1], Q = branch.Q[:, :, 1])
    elseif model === :mvOUMVA
        precalc = MVOUPrecalc(branch_lengths = Float64[sum(seg.length for seg in segments)], edge_segments = [segments], nregimes = size(bundle.A_regimes, 3), A_decomp = A_decomp)
        branch = _mvoumva_branch_cache(precalc, bundle.A_regimes, bundle.Sigma_regimes)
        return (Phi = branch.Phi[:, :, 1], Q = branch.Q[:, :, 1])
    end

    throw(ArgumentError("Unsupported multivariate OU branch kernel model $model"))
end

function _mvou_segment_eval_mean(
    model::Symbol,
    start_mean::AbstractVector{<:Real},
    bundle::MVOUParameterBundle,
    segments::Vector{SimmapSegment},
)
    theta_matrix = reshape(bundle.theta, length(start_mean), :)
    if model === :mvOUM || model === :mvOUMV || model === :mvOU1
        return _mvou_edge_regime_mean(start_mean, bundle.A, theta_matrix, segments)
    elseif model === :mvOUMA || model === :mvOUMVA
        return _mvouma_edge_regime_mean(start_mean, bundle.A_regimes, theta_matrix, segments)
    end

    throw(ArgumentError("Unsupported multivariate OU branch mean model $model"))
end

function _mvou_asr_bundle(fit::MVContinuousOUResult)
    if fit.model === :mvOU1 || fit.model === :mvOUM
        return MVOUParameterBundle(theta = vec(fit.theta'), A = fit.A[:, :, 1], Sigma = fit.Sigma[:, :, 1])
    elseif fit.model === :mvOUMV
        return MVOUParameterBundle(
            theta = vec(fit.theta'),
            A = fit.A[:, :, 1],
            Sigma = fit.Sigma[:, :, 1],
            Sigma_regimes = fit.Sigma,
        )
    elseif fit.model === :mvOUMA
        return MVOUParameterBundle(
            theta = vec(fit.theta'),
            A = fit.A[:, :, 1],
            A_regimes = fit.A,
            Sigma = fit.Sigma[:, :, 1],
        )
    elseif fit.model === :mvOUMVA
        return MVOUParameterBundle(
            theta = vec(fit.theta'),
            A = fit.A[:, :, 1],
            A_regimes = fit.A,
            Sigma = fit.Sigma[:, :, 1],
            Sigma_regimes = fit.Sigma,
        )
    end
    throw(ArgumentError("Unsupported multivariate OU ASR model $(fit.model)"))
end

function _mvou_asr_node_means(
    tree::CompactTree,
    fit::MVContinuousOUResult,
    bundle::MVOUParameterBundle,
    edge_segments::Vector{Vector{SimmapSegment}},
    root_regime::Integer,
)
    if fit.model === :mvOU1
        theta = vec(fit.theta)
        return [copy(theta) for _ in 1:tree.nnodes]
    elseif fit.model in (:mvOUM, :mvOUMV)
        if fit.root_mean_mode === :root_regime_theta
            return _mvou_node_path_means(tree, bundle, edge_segments, root_regime)
        elseif fit.root_mean_mode === :stationary_design
            designs = _mvou_node_design_matrices(
                tree,
                bundle.A,
                edge_segments,
                fit.nregimes,
                fit.A_decomp,
            )
            _mvou_row_standardize_designs!(designs)
            return _mvou_node_means_from_design(tree, designs, reshape(bundle.theta, size(bundle.A, 1), :))
        end
        return _mvou_node_path_means_zero(
            tree,
            bundle.A,
            reshape(bundle.theta, size(bundle.A, 1), :),
            edge_segments,
        )
    elseif fit.model in (:mvOUMA, :mvOUMVA)
        if fit.root_mean_mode === :root_regime_theta
            return _mvouma_node_path_means(tree, bundle, edge_segments, root_regime)
        elseif fit.root_mean_mode === :stationary_design
            designs = _mvouma_node_design_matrices(
                tree,
                bundle.A_regimes,
                edge_segments,
                fit.nregimes,
                fit.A_decomp,
            )
            _mvou_row_standardize_designs!(designs)
            return _mvou_node_means_from_design(tree, designs, reshape(bundle.theta, size(bundle.A_regimes, 1), :))
        end
        return _mvouma_node_path_means_zero(
            tree,
            bundle.A_regimes,
            reshape(bundle.theta, size(bundle.A_regimes, 1), :),
            edge_segments,
        )
    end
    throw(ArgumentError("Unsupported multivariate OU ASR model $(fit.model)"))
end

function _mvou_asr_branch_cache(
    fit::MVContinuousOUResult,
    precalc::MVOUPrecalc,
)
    if fit.model === :mvOU1 || fit.model === :mvOUM
        return _mvou_branch_cache(precalc, fit.A[:, :, 1], fit.Sigma[:, :, 1])
    elseif fit.model === :mvOUMV
        return _mvoumv_branch_cache(precalc, fit.A[:, :, 1], fit.Sigma)
    elseif fit.model === :mvOUMA
        return _mvouma_branch_cache(precalc, fit.A, fit.Sigma[:, :, 1])
    elseif fit.model === :mvOUMVA
        return _mvoumva_branch_cache(precalc, fit.A, fit.Sigma)
    end
    throw(ArgumentError("Unsupported multivariate OU ASR model $(fit.model)"))
end

function _mvou_single_segment_kernel(
    model::Symbol,
    bundle::MVOUParameterBundle,
    seg::SimmapSegment,
    p::Integer;
    A_decomp::Symbol = :cholesky,
)
    seg.length <= 0.0 && return _mvou_empty_kernel(p)
    return _mvou_segments_kernel(model, bundle, SimmapSegment[seg]; A_decomp = A_decomp)
end

function _mvou_edge_kernel_cache(
    model::Symbol,
    bundle::MVOUParameterBundle,
    segments::Vector{SimmapSegment},
    p::Integer,
    ;
    A_decomp::Symbol = :cholesky,
)
    nseg = length(segments)
    segment_kernel = Vector{NamedTuple{(:Phi, :Q), Tuple{Matrix{Float64}, Matrix{Float64}}}}(undef, nseg)
    prefix = Vector{NamedTuple{(:Phi, :Q), Tuple{Matrix{Float64}, Matrix{Float64}}}}(undef, nseg + 1)
    suffix = Vector{NamedTuple{(:Phi, :Q), Tuple{Matrix{Float64}, Matrix{Float64}}}}(undef, nseg + 1)
    prefix[1] = _mvou_empty_kernel(p)
    for i in 1:nseg
        segment_kernel[i] = _mvou_single_segment_kernel(model, bundle, segments[i], p; A_decomp = A_decomp)
        prefix[i + 1] = _mv_kernel_compose(prefix[i], segment_kernel[i])
    end
    suffix[nseg + 1] = _mvou_empty_kernel(p)
    for i in nseg:-1:1
        suffix[i] = _mv_kernel_compose(segment_kernel[i], suffix[i + 1])
    end
    return (segment = segment_kernel, prefix = prefix, suffix = suffix)
end

function _mvou_edge_mean_prefix_cache(
    model::Symbol,
    start_mean::AbstractVector{<:Real},
    bundle::MVOUParameterBundle,
    segments::Vector{SimmapSegment},
)
    nseg = length(segments)
    means = Vector{Vector{Float64}}(undef, nseg + 1)
    means[1] = Vector{Float64}(start_mean)
    for i in 1:nseg
        means[i + 1] = _mvou_segment_eval_mean(model, means[i], bundle, SimmapSegment[segments[i]])
    end
    return means
end

function _mvou_branch_posterior(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult;
    edge_segments::Vector{Vector{SimmapSegment}},
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    spec = _mvou_spec_with_root(
        fit.model;
        A_decomp = fit.A_decomp,
        root_mean_mode = fit.root_mean_mode,
        root_cov_mode = fit.root_cov_mode,
    )
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    bundle = _mvou_asr_bundle(fit)
    node_means = _mvou_asr_node_means(tree, fit, bundle, edge_segments, precalc.root_regime)

    centered = copy(data)
    for (i, tip) in enumerate(tree.tip_ids)
        centered[i, :] .-= node_means[Int(tip)]
    end

    branch = _mvou_asr_branch_cache(fit, precalc)
    post = _mv_recursive_posterior_cache(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = _mvou_root_covariance(precalc, bundle),
    )
    post.success || return (success = false,)
    return (success = true, post = post, bundle = bundle, node_means = node_means, precalc = precalc, p = p)
end
