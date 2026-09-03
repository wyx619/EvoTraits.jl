"""
    mvoumv_loglikelihood(tree, trait, edge_segments, A, Sigma_regimes, theta_regimes)

Evaluate the multivariate multi-optimum, multi-variance Ornstein-Uhlenbeck
(`mvOUMV`) log-likelihood with regime-specific optimum vectors and
regime-specific scatter matrices under a shared pull matrix `A`.
"""
function mvoumv_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    A::AbstractMatrix{<:Real},
    Sigma_regimes::Array{<:Real, 3},
    theta_regimes::AbstractMatrix{<:Real};
    root_mean_mode::Symbol = :stationary_design,
    root_cov_mode::Symbol = :fixed,
    A_decomp::Symbol = :cholesky,
    trait_names = nothing,
    regime_names = nothing,
)
    spec = _mvou_spec_with_root(
        :mvOUMV;
        A_decomp = A_decomp,
        root_mean_mode = root_mean_mode,
        root_cov_mode = root_cov_mode,
    )
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    size(theta_regimes, 2) == p || throw(ArgumentError("theta_regimes column count must match trait dimension"))
    size(Sigma_regimes, 1) == size(Sigma_regimes, 2) == p || throw(ArgumentError("Sigma_regimes dimensions must match trait dimension"))
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    size(theta_regimes, 1) == precalc.nregimes || throw(ArgumentError("theta_regimes row count must match regime count"))
    size(Sigma_regimes, 3) == precalc.nregimes || throw(ArgumentError("Sigma_regimes count must match regime count"))
    bundle = MVOUParameterBundle(
        theta = vec(copy(theta_regimes')),
        A = Matrix{Float64}(A),
        Sigma = Matrix{Float64}(Sigma_regimes[:, :, 1]),
        Sigma_regimes = Float64.(Sigma_regimes),
    )
    prof = _mvou_profile_dispatch(tree, data, bundle, precalc)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    nparams = Ablock + Sblock * precalc.nregimes + p * precalc.nregimes
    return MVContinuousOUResult(
        model = :mvOUMV,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = precalc.nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, precalc.nregimes, "regime"),
        theta = Matrix{Float64}(theta_regimes),
        A = reshape(Matrix{Float64}(A), p, p, 1),
        Sigma = Float64.(Sigma_regimes),
        A_decomp = spec.A_decomp,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
    )
end

"""
    fit_mvoumv(tree, trait, edge_segments; kwargs...)

Fit the multivariate multi-optimum, multi-variance Ornstein-Uhlenbeck
(`mvOUMV`) model with regime-specific `theta` and regime-specific `Sigma`
under a shared pull matrix `A`.
"""
function fit_mvoumv(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    optimization::Symbol = :SBPLX_L_BFGS,
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    A_decomp::Symbol = :cholesky,
    root_mean_mode::Symbol = :stationary_design,
    root_cov_mode::Symbol = :fixed,
    trait_names = nothing,
    regime_names = nothing,
)
    return _fit_mvou_recursive(
        tree,
        trait,
        _mvou_spec_with_root(
            :mvOUMV;
            A_decomp = A_decomp,
            root_mean_mode = root_mean_mode,
            root_cov_mode = root_cov_mode,
        );
        edge_segments = edge_segments,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        trait_names = trait_names,
        regime_names = regime_names,
    )
end

function fit_mvoumv(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvoumv(tree, trait, simmap.edge_segments; trait_names = trait_names, regime_names = simmap.state_labels, kwargs...)
end

function fit_mvoumv(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    simmap::SimmapSample;
    kwargs...,
)
    return fit_mvoumv(tree, trait, simmap.edge_segments; regime_names = simmap.state_labels, kwargs...)
end

