"""
    mvoum_LL(tree, trait, edge_segments, A, Sigma, theta_regimes)

Evaluate the multivariate multi-optimum Ornstein-Uhlenbeck (`mvOUM`)
log-likelihood with regime-specific optimum vectors, shared `A`, and shared
`Sigma`.

Root means `:regime` (default) and
`:stationary` correspond to mvMORPH's `root=FALSE` and
`"stationary"`. Select zero or stationary root covariance with
`root_cov_mode=:fixed` (default) or `:random`. The normalized design divides
each full mean-design row by its sum; undefined sums raise an error. Its root value uses the t -> 0+
limit. It is distinct from legacy zero-root `:stationary_design`.
"""
function mvoum_LL(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    theta_regimes::AbstractMatrix{<:Real};
    A_decomp::Symbol = :cholesky,
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    theta0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    trait_names = nothing,
    regime_names = nothing,
)
    spec = _mvou_public_spec(
        :mvOUM;
        A_decomp = A_decomp,
        root_mean_mode = root_mean_mode,
        root_cov_mode = root_cov_mode,
    )
    _validate_ou_theta0(root_mean_mode, theta0)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    theta0 === nothing || length(theta0) == p || throw(ArgumentError("theta0 length must match trait dimension"))
    size(theta_regimes, 2) == p || throw(ArgumentError("theta_regimes column count must match trait dimension"))
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    size(theta_regimes, 1) == precalc.nregimes || throw(ArgumentError("theta_regimes row count must match regime count"))
    bundle = MVOUParameterBundle(theta = vec(copy(theta_regimes')), A = Matrix{Float64}(A), Sigma = Matrix{Float64}(Sigma), theta0 = theta0 === nothing ? nothing : Float64.(theta0))
    prof = _mvou_fixed_LL(tree, data, precalc, bundle)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    nparams = Ablock + Sblock + _mvou_mean_ncoefficients(spec, p, precalc.nregimes)
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
        theta0 = bundle.theta0,
    )
end

"""
    fit_mvoum(tree, trait, edge_segments; kwargs...)

Fit the multivariate multi-optimum Ornstein-Uhlenbeck (`mvOUM`) model with
regime-specific optimum vectors and shared `A` / `Sigma`.

Root options match [`mvoum_LL`](@ref). GLS profiles the regime optima;
inspect `mean_diagnostics` for rank deficiency.

Set `n_starts > 0` to add that many random starting points, each optimised
independently (in parallel across threads) and combined with the structured
candidates; `random_seed` makes the random starts reproducible. The default
`n_starts = 0` keeps the structured multi-start behaviour unchanged.
"""
function fit_mvoum(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    optimization::Symbol = :SBPLX_L_BFGS,
    max_iterations::Integer = 300,
    polish_iterations::Integer = 100,
    rel_tol::Float64 = 1e-5,
    A_decomp::Symbol = :cholesky,
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    trait_names = nothing,
    regime_names = nothing,
    n_starts::Integer = 0,
    random_seed = nothing,
)
    return _fit_mvou_recursive(
        tree,
        trait,
        _mvou_public_spec(
            :mvOUM;
            A_decomp = A_decomp,
            root_mean_mode = root_mean_mode,
            root_cov_mode = root_cov_mode,
        );
        edge_segments = edge_segments,
        optimization = optimization,
        max_iterations = max_iterations,
        polish_iterations = polish_iterations,
        rel_tol = rel_tol,
        trait_names = trait_names,
        regime_names = regime_names,
        n_starts = n_starts,
        random_seed = random_seed,
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

