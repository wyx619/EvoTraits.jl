function _mvoum_tree_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    size(bundle.A, 1) == size(bundle.A, 2) == size(data, 2) || throw(ArgumentError("A dimensions must match trait dimension"))
    size(bundle.Sigma, 1) == size(bundle.Sigma, 2) == size(data, 2) || throw(ArgumentError("Sigma dimensions must match trait dimension"))
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
    path_means = _mvou_tip_path_means(tree, bundle, precalc)
    centered = data .- path_means
    branch = _mvou_branch_cache(precalc, bundle.A, bundle.Sigma)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, size(data, 2)),
        root_cov = zeros(Float64, size(data, 2), size(data, 2)),
        workspace = workspace,
        edge_Qinv = branch.Qinv,
        edge_logdet_Q = branch.logdet_Q,
    )
end

function _mvoumv_tree_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    size(bundle.A, 1) == size(bundle.A, 2) == p || throw(ArgumentError("A dimensions must match trait dimension"))
    size(bundle.Sigma_regimes, 1) == size(bundle.Sigma_regimes, 2) == p || throw(ArgumentError("Sigma_regimes dimensions must match trait dimension"))
    size(bundle.Sigma_regimes, 3) == precalc.nregimes || throw(ArgumentError("Sigma_regimes regime count must match precalc"))
    try
        if precalc.A_decomp === :cholesky
            cholesky(Symmetric(bundle.A))
        else
            vals = eigvals(Matrix{Float64}(bundle.A))
            all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
        end
        for r in 1:precalc.nregimes
            cholesky(Symmetric(bundle.Sigma_regimes[:, :, r]))
        end
    catch
        return (success = false, loglik = -Inf)
    end
    path_means = _mvou_tip_path_means(tree, bundle, precalc)
    centered = data .- path_means
    branch = _mvoumv_branch_cache(precalc, bundle.A, bundle.Sigma_regimes)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        workspace = workspace,
    )
end

function _mvou_tip_means_from_node_means(tree::CompactTree, node_means::Vector{Vector{Float64}}, p::Integer)
    means = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        means[i, :] .= node_means[Int(tip)]
    end
    return means
end

function _mvoumv_tree_pruning_profile_root_regime(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    try
        if precalc.A_decomp === :cholesky
            cholesky(Symmetric(bundle.A))
        else
            vals = eigvals(Matrix{Float64}(bundle.A))
            all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
        end
        for r in 1:precalc.nregimes
            cholesky(Symmetric(bundle.Sigma_regimes[:, :, r]))
        end
    catch
        return (success = false, loglik = -Inf)
    end
    node_means = _mvou_node_path_means(tree, bundle, precalc.edge_segments, precalc.root_regime)
    path_means = _mvou_tip_means_from_node_means(tree, node_means, p)
    centered = data .- path_means
    branch = _mvoumv_branch_cache(precalc, bundle.A, bundle.Sigma_regimes)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        workspace = workspace,
    )
end

function _mvouma_tree_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    size(bundle.Sigma, 1) == size(bundle.Sigma, 2) == p || throw(ArgumentError("Sigma dimensions must match trait dimension"))
    size(bundle.A_regimes, 1) == size(bundle.A_regimes, 2) == p || throw(ArgumentError("A_regimes dimensions must match trait dimension"))
    size(bundle.A_regimes, 3) == precalc.nregimes || throw(ArgumentError("A_regimes regime count must match precalc"))
    try
        cholesky(Symmetric(bundle.Sigma))
        for r in 1:precalc.nregimes
            if precalc.A_decomp === :cholesky
                cholesky(Symmetric(bundle.A_regimes[:, :, r]))
            else
                vals = eigvals(Matrix{Float64}(@view(bundle.A_regimes[:, :, r])))
                all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
            end
        end
    catch
        return (success = false, loglik = -Inf)
    end
    path_means = _mvouma_tip_path_means(tree, bundle, precalc)
    centered = data .- path_means
    branch = _mvouma_branch_cache(precalc, bundle.A_regimes, bundle.Sigma)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        workspace = workspace,
    )
end

function _mvouma_tree_pruning_profile_root_regime(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    try
        cholesky(Symmetric(bundle.Sigma))
        for r in 1:precalc.nregimes
            if precalc.A_decomp === :cholesky
                cholesky(Symmetric(bundle.A_regimes[:, :, r]))
            else
                vals = eigvals(Matrix{Float64}(@view(bundle.A_regimes[:, :, r])))
                all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
            end
        end
    catch
        return (success = false, loglik = -Inf)
    end
    node_means = _mvouma_node_path_means(tree, bundle, precalc.edge_segments, precalc.root_regime)
    path_means = _mvou_tip_means_from_node_means(tree, node_means, p)
    centered = data .- path_means
    branch = _mvouma_branch_cache(precalc, bundle.A_regimes, bundle.Sigma)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        workspace = workspace,
    )
end

function _mvoumva_tree_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    size(bundle.A_regimes, 1) == size(bundle.A_regimes, 2) == p || throw(ArgumentError("A_regimes dimensions must match trait dimension"))
    size(bundle.Sigma_regimes, 1) == size(bundle.Sigma_regimes, 2) == p || throw(ArgumentError("Sigma_regimes dimensions must match trait dimension"))
    size(bundle.A_regimes, 3) == size(bundle.Sigma_regimes, 3) == precalc.nregimes || throw(ArgumentError("Regime counts must match precalc"))
    try
        for r in 1:precalc.nregimes
            if precalc.A_decomp === :cholesky
                cholesky(Symmetric(bundle.A_regimes[:, :, r]))
            else
                vals = eigvals(Matrix{Float64}(@view(bundle.A_regimes[:, :, r])))
                all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
            end
            cholesky(Symmetric(bundle.Sigma_regimes[:, :, r]))
        end
    catch
        return (success = false, loglik = -Inf)
    end
    path_means = _mvouma_tip_path_means(tree, bundle, precalc)
    centered = data .- path_means
    branch = _mvoumva_branch_cache(precalc, bundle.A_regimes, bundle.Sigma_regimes)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        workspace = workspace,
    )
end

function _mvoumva_tree_pruning_profile_root_regime(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    try
        for r in 1:precalc.nregimes
            if precalc.A_decomp === :cholesky
                cholesky(Symmetric(bundle.A_regimes[:, :, r]))
            else
                vals = eigvals(Matrix{Float64}(@view(bundle.A_regimes[:, :, r])))
                all(z -> real(z) > 0.0, vals) || throw(ArgumentError("A must have eigenvalues with positive real parts"))
            end
            cholesky(Symmetric(bundle.Sigma_regimes[:, :, r]))
        end
    catch
        return (success = false, loglik = -Inf)
    end
    node_means = _mvouma_node_path_means(tree, bundle, precalc.edge_segments, precalc.root_regime)
    path_means = _mvou_tip_means_from_node_means(tree, node_means, p)
    centered = data .- path_means
    branch = _mvoumva_branch_cache(precalc, bundle.A_regimes, bundle.Sigma_regimes)
    return _mv_recursive_profile(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        workspace = workspace,
    )
end
