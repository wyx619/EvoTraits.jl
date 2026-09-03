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

    centered = data .- reshape(bundle.theta, 1, p)
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

    centered = data .- repeat(fit.theta, tree.ntips, 1)
    precalc = _mvou_precalc(
        tree,
        _mvou_spec_with_root(
            :mvOU1;
            A_decomp = fit.A_decomp,
            root_mean_mode = fit.root_mean_mode,
            root_cov_mode = fit.root_cov_mode,
        ),
    )
    bundle = MVOUParameterBundle(
        theta = vec(fit.theta),
        A = fit.A[:, :, 1],
        Sigma = fit.Sigma[:, :, 1],
    )
    branch = _mvou_branch_cache(precalc, bundle.A, bundle.Sigma)
    return _mv_recursive_asr(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = _mvou_root_covariance(precalc, bundle),
        shift = vec(fit.theta),
        model = :mvOU1,
    )
end

