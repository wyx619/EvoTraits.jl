"""
    mvou1_loglikelihood(tree, trait, A, Sigma, theta)

Evaluate the multivariate single-optimum Ornstein-Uhlenbeck (`mvOU1`)
log-likelihood for fixed `A`, `Sigma`, and `theta` using tree pruning.
"""
function mvou1_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    theta::AbstractVector{<:Real};
    A_decomp::Symbol = :cholesky,
    root_cov_mode::Symbol = :fixed,
    trait_names = nothing,
)
    spec = _mvou_spec_with_root(:mvOU1; A_decomp = A_decomp, root_cov_mode = root_cov_mode)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    theta_vec = Float64.(collect(theta))
    length(theta_vec) == p || throw(ArgumentError("theta length must match trait dimension"))
    precalc = _mvou_precalc(tree, spec)
    bundle = MVOUParameterBundle(theta = theta_vec, A = Matrix{Float64}(A), Sigma = Matrix{Float64}(Sigma))
    prof = _mvou_profile_dispatch(tree, data, bundle, precalc)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    nparams = Ablock + Sblock + p
    return MVContinuousOUResult(
        model = :mvOU1,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = 1,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = ["OU1"],
        theta = reshape(theta_vec, 1, p),
        A = reshape(Matrix{Float64}(A), p, p, 1),
        Sigma = reshape(Matrix{Float64}(Sigma), p, p, 1),
        A_decomp = spec.A_decomp,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
    )
end

"""
    fit_mvou1(tree, trait; kwargs...)

Fit the multivariate single-optimum Ornstein-Uhlenbeck (`mvOU1`) model by
maximum likelihood.
"""
function fit_mvou1(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-7,
    A_decomp::Symbol = :cholesky,
    root_cov_mode::Symbol = :fixed,
    trait_names = nothing,
)
    return _fit_mvou_recursive(
        tree,
        trait,
        _mvou_spec_with_root(:mvOU1; A_decomp = A_decomp, root_cov_mode = root_cov_mode);
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        trait_names = trait_names,
        regime_names = ["OU1"],
    )
end

function fit_mvou1(
    tree::CompactTree,
    data::AbstractDataFrame;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvou1(tree, trait; trait_names = trait_names, kwargs...)
end

