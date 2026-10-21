function _mvou1_tree_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    p = size(data, 2)
    size(bundle.A, 1) == size(bundle.A, 2) == p || throw(ArgumentError("A dimensions must match trait dimension"))
    size(bundle.Sigma, 1) == size(bundle.Sigma, 2) == p || throw(ArgumentError("Sigma dimensions must match trait dimension"))
    length(bundle.theta) == p || throw(ArgumentError("theta length must match trait dimension"))

    try
        if precalc.A_decomp === :cholesky
            cholesky(Symmetric(bundle.A))
        else
            vals = eigvals(Matrix{Float64}(bundle.A))
            all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
        end
        cholesky(Symmetric(bundle.Sigma))
    catch
        return (success = false, loglik = -Inf)
    end

    means = _mvou_node_means(tree, bundle, precalc)
    centered = copy(data)
    for (i, tip) in enumerate(tree.tip_ids)
        @views centered[i, :] .-= means[tip]
    end
    branch = _mvou_branch_cache(precalc, bundle.A, bundle.Sigma)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = _mvou_root_covariance(precalc, bundle),
        workspace = workspace,
        edge_Qinv = branch.Qinv,
        edge_logdet_Q = branch.logdet_Q,
    )
end


function _mvou1_tree_pruning_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult,
)
    data = _validate_multivariate_trait(tree, trait)
    fit.success || throw(ArgumentError("mvOU1 ASR requires a successful fit"))
    p = fit.ntraits
    size(data, 2) == p || throw(ArgumentError("Trait dimension does not match fit"))

    precalc = _mvou_precalc(
        tree,
        _mvou_spec_with_root(
            :mvOU1;
            A_decomp = fit.A_decomp,
            root_mean_mode = fit.root_mean_mode,
            root_cov_mode = fit.root_cov_mode,
        ),
    )
    bundle = _mvou_asr_bundle(fit)
    means = _mvou_node_means(tree, bundle, precalc)
    centered = copy(data)
    for (i, tip) in enumerate(tree.tip_ids)
        @views centered[i, :] .-= means[tip]
    end
    branch = _mvou_branch_cache(precalc, bundle.A, bundle.Sigma)
    asr = _mv_recursive_asr(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = _mvou_root_covariance(precalc, bundle),
        shift = zeros(p),
        model = :mvOU1,
    )
    asr.success || return asr
    for (i, node) in enumerate(asr.node_ids)
        @views asr.estimates[i, :] .+= means[node]
    end
    for node in 1:tree.nnodes
        @views asr.all_node_estimates[node, :] .+= means[node]
    end
    for (i, tip) in enumerate(asr.tip_ids)
        @views asr.tip_estimates[i, :] .+= means[tip]
    end
    return asr
end

