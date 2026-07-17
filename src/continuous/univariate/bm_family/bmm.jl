function _validate_mapped_edge(tree::CompactTree, mapped_edge::AbstractMatrix{<:Real}; atol::Float64=1e-8)
    size(mapped_edge, 1) == tree.nedges || throw(ArgumentError("mapped_edge must have $(tree.nedges) rows"))
    size(mapped_edge, 2) >= 2 || throw(ArgumentError("BMM requires at least 2 regimes"))
    mapped = Matrix{Float64}(mapped_edge)
    any(x -> !isfinite(x) || x < 0.0, mapped) && throw(ArgumentError("mapped_edge contains negative or non-finite values"))
    for edge in 1:tree.nedges
        isapprox(sum(@view(mapped[edge, :])), tree.edge_length[edge]; atol=atol) || throw(ArgumentError("mapped_edge row $edge does not sum to branch length"))
    end
    return mapped
end

function _bmm_edge_variances(mapped_edge::AbstractMatrix{<:Real}, sigma2::AbstractVector{<:Real})
    size(mapped_edge, 2) == length(sigma2) || throw(ArgumentError("Number of sigma2 parameters must match number of regimes"))
    any(x -> !isfinite(x) || x <= 0.0, sigma2) && return fill(NaN, size(mapped_edge, 1))
    return vec(mapped_edge * Float64.(sigma2))
end

function _bmm_profile(tree::CompactTree, trait::AbstractVector{<:Real}, mapped_edge::AbstractMatrix{<:Real}, sigma2::AbstractVector{<:Real})
    _validate_binary_tree(tree)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    edge_variances = _bmm_edge_variances(mapped, sigma2)
    return _bm_pruning_profile_from_edge_variances(tree, tr, edge_variances)
end

"""
    bmm_loglikelihood(tree, trait, mapped_edge, sigma2)

Evaluate the multi-regime Brownian-motion (`BMM`) log-likelihood for a fixed
vector of regime-specific diffusion rates `sigma2`. `mapped_edge` must contain
per-edge regime durations whose rows sum to the original branch lengths.
"""
function bmm_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    mapped_edge::AbstractMatrix{<:Real},
    sigma2::AbstractVector{<:Real};
    trait_name = nothing,
    regime_names = nothing,
)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    prof = _bmm_profile(tree, trait, mapped, sigma2)
    nregimes = size(mapped, 2)
    return ContinuousMultiRegimeResult(
        model = :BMM,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nregimes + 1) : Inf,
        nparams = nregimes + 1,
        trait_name = _continuous_checked_trait_name(trait_name),
        regime_names = _continuous_checked_regime_names(regime_names, nregimes),
        sigma2 = Float64.(sigma2),
        theta = prof.root_state,
        root_state = prof.root_state,
        nregimes = nregimes,
    )
end

"""
    fit_bmm(tree, trait, mapped_edge; kwargs...)

Fit the multi-regime Brownian-motion (`BMM`) model with one `sigma2` per regime.
The function reuses the shared Brownian pruning kernel after converting mapped
regime durations into per-edge variance increments.
"""
function fit_bmm(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    mapped_edge::AbstractMatrix{<:Real};
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    trait_name = nothing,
    regime_names = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    nregimes = size(mapped, 2)

    sigma0 = fill(max(var(filter(!isnan, tr)), 1e-8), nregimes)
    init = log.(sigma0)

    objective = function (logsigmas)
        sigma2 = exp.(Float64.(logsigmas))
        prof = _bmm_profile(tree, tr, mapped, sigma2)
        return prof.success ? -prof.loglik : Inf
    end

    result = _continuous_optimize_objective(
        objective,
        init;
        method = :L_BFGS,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )

    fitted_sigma2 = exp.(_continuous_result_minimizer(result))
    prof = _bmm_profile(tree, tr, mapped, fitted_sigma2)

    return ContinuousMultiRegimeResult(
        model = :BMM,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nregimes + 1) : Inf,
        nparams = nregimes + 1,
        trait_name = _continuous_checked_trait_name(trait_name),
        regime_names = _continuous_checked_regime_names(regime_names, nregimes),
        sigma2 = collect(fitted_sigma2),
        theta = prof.root_state,
        root_state = prof.root_state,
        nregimes = nregimes,
        converged = _continuous_result_converged(result),
        iterations = _continuous_result_iterations(result),
        f_calls = _continuous_result_f_calls(result),
    )
end

function fit_bmm(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_bmm(
        tree,
        trait,
        simmap.mapped_edge;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, size(simmap.mapped_edge, 2)),
        kwargs...,
    )
end

function fit_bmm(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample;
    trait_name = nothing,
    kwargs...,
)
    return fit_bmm(
        tree,
        trait,
        simmap.mapped_edge;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, size(simmap.mapped_edge, 2)),
        kwargs...,
    )
end

function bmm_loglikelihood(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample,
    sigma2::AbstractVector{<:Real};
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return bmm_loglikelihood(
        tree,
        trait,
        simmap.mapped_edge,
        sigma2;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, size(simmap.mapped_edge, 2)),
        kwargs...,
    )
end

function bmm_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample,
    sigma2::AbstractVector{<:Real};
    trait_name = nothing,
    kwargs...,
)
    return bmm_loglikelihood(
        tree,
        trait,
        simmap.mapped_edge,
        sigma2;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, size(simmap.mapped_edge, 2)),
        kwargs...,
    )
end

