"""
    oum_LL(tree, trait, edge_segments, alpha, sigma2, theta_regimes; kwargs...)

Evaluate OUM at fixed parameters. `root_mean_mode=:regime` (default)
and `:stationary` correspond to mvMORPH's `root=FALSE`
and `"stationary"`. Covariance modes `:fixed` (default) and `:random`
correspond to `vcv="fixedRoot"` and `"randomRoot"`. The normalized design is
distinct from legacy zero-root `:stationary_design`, which is only retained
for restoring saved results.
"""
function oum_LL(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    alpha::Real,
    sigma2::Real,
    theta_regimes::AbstractVector{<:Real},
    ;
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    theta0::Union{Nothing,Real} = nothing,
    trait_name = nothing,
    regime_names = nothing,
)
    spec = _ou_public_spec(:OUM; root_mean_mode, root_cov_mode)
    _validate_ou_theta0(root_mean_mode, theta0)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    length(theta_regimes) == cache.nregimes || throw(ArgumentError("theta_regimes length must match regime count"))
    bundle = OUParameterBundle(theta = Float64.(theta_regimes), alpha = [Float64(alpha)], sigma2 = [Float64(sigma2)], theta0 = theta0)
    prof = _ou_LL(tree, trait, spec, bundle; cache = cache)
    return _ou_multiregime_result(spec, bundle, prof, cache.nregimes; trait_name = trait_name, regime_names = regime_names)
end

"""
    fit_oum(tree, trait, edge_segments; kwargs...)

Fit OUM using the root options documented for [`oum_LL`](@ref).
Regime optima are profiled by GLS. Inspect `mean_diagnostics`
before interpreting individual means on an ultrametric tree.
"""
function fit_oum(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    trait_name = nothing,
    regime_names = nothing,
)
    spec = _ou_public_spec(:OUM; root_mean_mode, root_cov_mode)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    fit = _ou_fit(
        tree,
        trait,
        spec;
        cache = cache,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )
    return _ou_multiregime_result(spec, fit.bundle, fit.profile, fit.nregimes; result = fit.result, trait_name = trait_name, regime_names = regime_names)
end

function fit_oum(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    trait_col = nothing,
    kwargs...,
)
    return _fit_univariate_simmap(fit_oum, tree, data, simmap; trait_col = trait_col, kwargs...)
end

function fit_oum(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample;
    trait_name = nothing,
    kwargs...,
)
    return _fit_univariate_simmap(fit_oum, tree, trait, simmap; trait_name = trait_name, kwargs...)
end

function oum_LL(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample,
    alpha::Real,
    sigma2::Real,
    theta_regimes::AbstractVector{<:Real};
    trait_col = nothing,
    kwargs...,
)
    return _LL_univariate_simmap(oum_LL, tree, data, simmap, alpha, sigma2, theta_regimes; trait_col = trait_col, kwargs...)
end

function oum_LL(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample,
    alpha::Real,
    sigma2::Real,
    theta_regimes::AbstractVector{<:Real};
    trait_name = nothing,
    kwargs...,
)
    return _LL_univariate_simmap(oum_LL, tree, trait, simmap, alpha, sigma2, theta_regimes; trait_name = trait_name, kwargs...)
end
