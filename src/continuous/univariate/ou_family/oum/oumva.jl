"""
    oumva_LL(tree, trait, edge_segments, alpha, sigma2, theta; kwargs...)

Evaluate the OUMVA likelihood with `root_cov_mode=:fixed`. The default
`root_mean_mode=:regime` uses the root regime optimum as the root
mean.
"""
function oumva_LL(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    alpha_regimes::AbstractVector{<:Real},
    sigma2_regimes::AbstractVector{<:Real},
    theta_regimes::AbstractVector{<:Real},
    ;
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    theta0::Union{Nothing,Real} = nothing,
    trait_name = nothing,
    regime_names = nothing,
)
    spec = _ou_public_spec(:OUMVA; root_mean_mode, root_cov_mode)
    _validate_ou_theta0(root_mean_mode, theta0)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    length(theta_regimes) == cache.nregimes || throw(ArgumentError("theta_regimes length must match regime count"))
    length(alpha_regimes) == cache.nregimes || throw(ArgumentError("alpha_regimes length must match regime count"))
    length(sigma2_regimes) == cache.nregimes || throw(ArgumentError("sigma2_regimes length must match regime count"))
    bundle = OUParameterBundle(
        theta = Float64.(theta_regimes),
        alpha = Float64.(alpha_regimes),
        sigma2 = Float64.(sigma2_regimes),
        theta0 = theta0,
    )
    prof = _ou_LL(tree, trait, spec, bundle; cache = cache)
    return _ou_multiregime_result(spec, bundle, prof, cache.nregimes; trait_name = trait_name, regime_names = regime_names)
end

"""
    fit_oumva(tree, trait, edge_segments; kwargs...)

Fit OUMVA, profiling the regime optima by GLS. Use
`root_mean_mode=:regime` with `root_cov_mode=:fixed`.
Inspect `mean_diagnostics` for identifiability separately from `converged`.
"""
function fit_oumva(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    max_iterations::Integer = 800,
    rel_tol::Float64 = 1e-5,
    root_mean_mode::Symbol = :regime,
    root_cov_mode::Symbol = :fixed,
    trait_name = nothing,
    regime_names = nothing,
)
    spec = _ou_public_spec(:OUMVA; root_mean_mode, root_cov_mode)
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

function fit_oumva(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_oumva(
        tree,
        trait,
        simmap.edge_segments;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function fit_oumva(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample;
    trait_name = nothing,
    kwargs...,
)
    return fit_oumva(
        tree,
        trait,
        simmap.edge_segments;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function oumva_LL(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample,
    alpha_regimes::AbstractVector{<:Real},
    sigma2_regimes::AbstractVector{<:Real},
    theta_regimes::AbstractVector{<:Real};
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return oumva_LL(
        tree,
        trait,
        simmap.edge_segments,
        alpha_regimes,
        sigma2_regimes,
        theta_regimes;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function oumva_LL(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample,
    alpha_regimes::AbstractVector{<:Real},
    sigma2_regimes::AbstractVector{<:Real},
    theta_regimes::AbstractVector{<:Real};
    trait_name = nothing,
    kwargs...,
)
    return oumva_LL(
        tree,
        trait,
        simmap.edge_segments,
        alpha_regimes,
        sigma2_regimes,
        theta_regimes;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end
