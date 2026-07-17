"""
    mvouma_loglikelihood(tree, trait, edge_segments, A_regimes, Sigma, theta_regimes)

Evaluate the multivariate multi-optimum, multi-alpha Ornstein-Uhlenbeck
(`mvOUMA`) log-likelihood with regime-specific optimum vectors and
regime-specific pull matrices under a shared scatter matrix `Sigma`.
"""
function mvouma_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    A_regimes::Array{<:Real, 3},
    Sigma::AbstractMatrix{<:Real},
    theta_regimes::AbstractMatrix{<:Real};
    root_mean_mode::Symbol = :stationary_design,
    A_decomp::Symbol = :cholesky,
    trait_names = nothing,
    regime_names = nothing,
)
    base_spec = mvou_spec(:mvOUMA; A_decomp = A_decomp)
    spec = MVOUSpec(
        model = base_spec.model,
        theta_mode = base_spec.theta_mode,
        A_mode = base_spec.A_mode,
        Sigma_mode = base_spec.Sigma_mode,
        A_decomp = base_spec.A_decomp,
        root_mean_mode = root_mean_mode,
        root_cov_mode = base_spec.root_cov_mode,
    )
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    size(theta_regimes, 2) == p || throw(ArgumentError("theta_regimes column count must match trait dimension"))
    size(A_regimes, 1) == size(A_regimes, 2) == p || throw(ArgumentError("A_regimes dimensions must match trait dimension"))
    size(Sigma, 1) == size(Sigma, 2) == p || throw(ArgumentError("Sigma dimensions must match trait dimension"))
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    size(theta_regimes, 1) == precalc.nregimes || throw(ArgumentError("theta_regimes row count must match regime count"))
    size(A_regimes, 3) == precalc.nregimes || throw(ArgumentError("A_regimes count must match regime count"))
    bundle = MVOUParameterBundle(
        theta = vec(copy(theta_regimes')),
        A = Matrix{Float64}(A_regimes[:, :, 1]),
        A_regimes = Float64.(A_regimes),
        Sigma = Matrix{Float64}(Sigma),
    )
    prof =
        root_mean_mode === :stationary_design ?
        _mvou_profile_dispatch(tree, data, bundle, precalc) :
        root_mean_mode === :root_regime_theta ?
        _mvouma_tree_pruning_profile_root_regime(tree, data, bundle, precalc) :
        throw(ArgumentError("Unsupported mvOUMA root_mean_mode=$root_mean_mode"))
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    nparams = Ablock * precalc.nregimes + Sblock + p * precalc.nregimes
    return MVContinuousOUResult(
        model = :mvOUMA,
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
        A = Float64.(A_regimes),
        Sigma = reshape(Matrix{Float64}(Sigma), p, p, 1),
        A_decomp = spec.A_decomp,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
    )
end

"""
    fit_mvouma(tree, trait, edge_segments; kwargs...)

Fit the multivariate multi-optimum, multi-alpha Ornstein-Uhlenbeck (`mvOUMA`)
model with regime-specific `theta` and regime-specific `A` under a shared
scatter matrix `Sigma`.
"""
function fit_mvouma(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    optimization::Symbol = :SBPLX_L_BFGS,
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    A_decomp::Symbol = :cholesky,
    trait_names = nothing,
    regime_names = nothing,
)
    return _fit_mvou_recursive(
        tree,
        trait,
        mvou_spec(:mvOUMA; A_decomp = A_decomp);
        edge_segments = edge_segments,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        trait_names = trait_names,
        regime_names = regime_names,
    )
end

function fit_mvouma(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvouma(tree, trait, simmap.edge_segments; trait_names = trait_names, regime_names = simmap.state_labels, kwargs...)
end

function fit_mvouma(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    simmap::SimmapSample;
    kwargs...,
)
    return fit_mvouma(tree, trait, simmap.edge_segments; regime_names = simmap.state_labels, kwargs...)
end

