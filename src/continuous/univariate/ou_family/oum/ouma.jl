function ouma_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    alpha_regimes::AbstractVector{<:Real},
    sigma2::Real,
    theta_regimes::AbstractVector{<:Real},
    ;
    trait_name = nothing,
    regime_names = nothing,
)
    spec = ou_spec(:OUMA)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    length(theta_regimes) == cache.nregimes || throw(ArgumentError("theta_regimes length must match regime count"))
    length(alpha_regimes) == cache.nregimes || throw(ArgumentError("alpha_regimes length must match regime count"))
    bundle = OUParameterBundle(
        theta = Float64.(theta_regimes),
        alpha = Float64.(alpha_regimes),
        sigma2 = [Float64(sigma2)],
    )
    prof = _ou_loglikelihood(tree, trait, spec, bundle; cache = cache)
    return _ou_multiregime_result(spec, bundle, prof, cache.nregimes; trait_name = trait_name, regime_names = regime_names)
end

function fit_ouma(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    max_iterations::Integer = 600,
    rel_tol::Float64 = 1e-5,
    trait_name = nothing,
    regime_names = nothing,
)
    spec = ou_spec(:OUMA)
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

function fit_ouma(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_ouma(
        tree,
        trait,
        simmap.edge_segments;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function fit_ouma(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample;
    trait_name = nothing,
    kwargs...,
)
    return fit_ouma(
        tree,
        trait,
        simmap.edge_segments;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function ouma_loglikelihood(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample,
    alpha_regimes::AbstractVector{<:Real},
    sigma2::Real,
    theta_regimes::AbstractVector{<:Real};
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return ouma_loglikelihood(
        tree,
        trait,
        simmap.edge_segments,
        alpha_regimes,
        sigma2,
        theta_regimes;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function ouma_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample,
    alpha_regimes::AbstractVector{<:Real},
    sigma2::Real,
    theta_regimes::AbstractVector{<:Real};
    trait_name = nothing,
    kwargs...,
)
    return ouma_loglikelihood(
        tree,
        trait,
        simmap.edge_segments,
        alpha_regimes,
        sigma2,
        theta_regimes;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end
