"""
    mvou1_LL(tree, trait, A, Sigma, theta)

Evaluate the multivariate single-optimum Ornstein-Uhlenbeck (`mvOU1`)
log-likelihood for fixed `A`, `Sigma`, and `theta` using tree pruning.

Root means `:regime` (default) and
`:stationary` correspond to mvMORPH's `root=FALSE` and
`"stationary"`. In OU1 the normalized design equals the default. Select zero
or stationary root covariance with `root_cov_mode=:fixed` (default) or
`:random`.
"""
function mvou1_LL(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    theta::AbstractVector{<:Real};
    A_decomp::Symbol = :cholesky,
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    theta0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    trait_names = nothing,
)
    spec = _mvou_public_spec(:mvOU1; A_decomp, root_mean_mode, root_cov_mode)
    _validate_ou_theta0(root_mean_mode, theta0)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    theta0 === nothing || length(theta0) == p || throw(ArgumentError("theta0 length must match trait dimension"))
    theta_vec = Float64.(collect(theta))
    length(theta_vec) == p || throw(ArgumentError("theta length must match trait dimension"))
    precalc = _mvou_precalc(tree, spec)
    bundle = MVOUParameterBundle(theta = theta_vec, A = Matrix{Float64}(A), Sigma = Matrix{Float64}(Sigma), theta0 = theta0 === nothing ? nothing : Float64.(theta0))
    prof = _mvou_fixed_LL(tree, data, precalc, bundle)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    nparams = Ablock + Sblock + _mvou_mean_ncoefficients(spec, p, 1)
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
        theta0 = bundle.theta0,
    )
end

"""
    fit_mvou1(tree, trait; kwargs...)

Fit the multivariate single-optimum Ornstein-Uhlenbeck (`mvOU1`) model by
maximum likelihood.

Root options match [`mvou1_LL`](@ref). GLS profiles the optimum;
the root mean is the same vector. The result records `mean_diagnostics`.

Set `n_starts > 0` to add that many random starting points, each optimised
independently (in parallel across threads) and combined with the structured
candidates; `random_seed` makes the random starts reproducible. The default
`n_starts = 0` keeps the structured multi-start behaviour unchanged.
"""
function fit_mvou1(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-7,
    A_decomp::Symbol = :cholesky,
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    trait_names = nothing,
    n_starts::Integer = 0,
    random_seed = nothing,
)
    return _fit_mvou_recursive(
        tree,
        trait,
        _mvou_public_spec(:mvOU1; A_decomp, root_mean_mode, root_cov_mode);
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        trait_names = trait_names,
        regime_names = ["OU1"],
        n_starts = n_starts,
        random_seed = random_seed,
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

