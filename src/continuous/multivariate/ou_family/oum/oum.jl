"""
    mvoum_loglikelihood(tree, trait, edge_segments, A, Sigma, theta_regimes)

Evaluate the multivariate multi-optimum Ornstein-Uhlenbeck (`mvOUM`)
log-likelihood with regime-specific optimum vectors, shared `A`, and shared
`Sigma`.
"""
function mvoum_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    theta_regimes::AbstractMatrix{<:Real};
    A_decomp::Symbol = :cholesky,
    root_mean_mode::Symbol = :stationary_design,
    root_cov_mode::Symbol = :fixed,
    trait_names = nothing,
    regime_names = nothing,
)
    spec = _mvou_spec_with_root(
        :mvOUM;
        A_decomp = A_decomp,
        root_mean_mode = root_mean_mode,
        root_cov_mode = root_cov_mode,
    )
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    size(theta_regimes, 2) == p || throw(ArgumentError("theta_regimes column count must match trait dimension"))
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    size(theta_regimes, 1) == precalc.nregimes || throw(ArgumentError("theta_regimes row count must match regime count"))
    bundle = MVOUParameterBundle(theta = vec(copy(theta_regimes')), A = Matrix{Float64}(A), Sigma = Matrix{Float64}(Sigma))
    prof = _mvou_profile_dispatch(tree, data, bundle, precalc)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    nparams = Ablock + Sblock + p * precalc.nregimes
    return MVContinuousOUResult(
        model = :mvOUM,
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
        Sigma = reshape(Matrix{Float64}(Sigma), p, p, 1),
        A_decomp = spec.A_decomp,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
    )
end

"""
    fit_mvoum(tree, trait, edge_segments; kwargs...)

Fit the multivariate multi-optimum Ornstein-Uhlenbeck (`mvOUM`) model with
regime-specific optimum vectors and shared `A` / `Sigma`.
"""
function fit_mvoum(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    optimization::Symbol = :SBPLX_L_BFGS,
    max_iterations::Integer = 300,
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
            :mvOUM;
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

function fit_mvoum(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvoum(tree, trait, simmap.edge_segments; trait_names = trait_names, regime_names = simmap.state_labels, kwargs...)
end

function fit_mvoum(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    simmap::SimmapSample;
    kwargs...,
)
    return fit_mvoum(tree, trait, simmap.edge_segments; regime_names = simmap.state_labels, kwargs...)
end

