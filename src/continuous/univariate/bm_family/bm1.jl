function _bm_pruning_profile_from_edge_variances(tree::CompactTree, trait::AbstractVector{<:Real}, edge_variances::AbstractVector{<:Real})
    _validate_binary_tree(tree)
    y = _validate_univariate_trait_allow_missing(tree, trait)
    length(edge_variances) == tree.nedges || throw(ArgumentError("Expected $(tree.nedges) edge variances, got $(length(edge_variances))"))
    any(v -> !isfinite(v) || v < 0.0, edge_variances) && return (success = false, loglik = -Inf, root_state = NaN, root_var = NaN)
    if any(isnan, y)
        prof = _linear_gaussian_loglik(
            tree,
            y,
            ones(Float64, tree.nedges),
            zeros(Float64, tree.nedges),
            Float64.(edge_variances),
        )
        return (success = prof.success, loglik = prof.loglik, root_state = prof.root_state, root_var = NaN)
    end
    edge_var = Float64.(edge_variances)

    means = zeros(Float64, tree.nnodes)
    spreads = zeros(Float64, tree.nnodes)
    loglik = 0.0

    for (i, node) in enumerate(tree.tip_ids)
        means[node] = y[i]
        spreads[node] = 0.0
    end

    for node in tree.postorder_internal
        child1 = tree.children[node][1]
        child2 = tree.children[node][2]

        v1 = spreads[child1] + edge_var[tree.first_child_edge[node]]
        v2 = spreads[child2] + edge_var[tree.last_child_edge[node]]

        denom = v1 + v2
        denom > 0.0 || return (success = false, loglik = -Inf, root_state = NaN, root_var = NaN)

        contrast = means[child1] - means[child2]
        loglik += -0.5 * (log(2 * pi * denom) + (contrast * contrast) / denom)

        invsum = 1.0 / denom
        means[node] = (means[child1] * v2 + means[child2] * v1) * invsum
        spreads[node] = (v1 * v2) * invsum
    end

    root = tree.root
    root_var = spreads[root]
    root_var > 0.0 || return (success = false, loglik = -Inf, root_state = NaN, root_var = NaN)
    loglik += -0.5 * log(2 * pi * root_var)

    return (success = isfinite(loglik), loglik = loglik, root_state = means[root], root_var = root_var)
end

function _bm1_profile(tree::CompactTree, trait::AbstractVector{<:Real}, sigma2::Float64)
    sigma2 > 0.0 || return (success = false, loglik = -Inf, root_state = NaN, root_var = NaN)
    return _bm_pruning_profile_from_edge_variances(tree, trait, sigma2 .* tree.edge_length)
end

"""
    bm1_loglikelihood(tree, trait, sigma2)

Evaluate the single-trait Brownian-motion (`BM1`) log-likelihood on a
tree. The returned `ContinuousFitResult` profiles the root state and reports
the corresponding AIC with two free parameters (`sigma2` and root).
"""
function bm1_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    sigma2::Real;
    trait_name = nothing,
)
    profile = _bm1_profile(tree, trait, Float64(sigma2))
    return ContinuousFitResult(
        model = :BM1,
        success = profile.success,
        loglik = profile.loglik,
        aic = profile.success ? compute_aic(profile.loglik, 2) : Inf,
        nparams = 2,
        trait_name = _continuous_checked_trait_name(trait_name),
        sigma2 = Float64(sigma2),
        theta = profile.root_state,
        root_state = profile.root_state,
    )
end

"""
    fit_bm1(tree, trait; kwargs...)

Fit the single-trait Brownian-motion (`BM1`) model by maximum likelihood using
the pruning kernel. Keyword arguments control iteration limits, tolerance, and
optional trait naming.
"""
function fit_bm1(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    trait_name = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)

    observed = filter(!isnan, tr)
    var_guess = max(var(observed), 1e-8)
    init = [log(var_guess)]

    objective = function (logsigma)
        sigma2 = exp(only(logsigma))
        prof = _bm1_profile(tree, tr, sigma2)
        return prof.success ? -prof.loglik : Inf
    end

    result = _continuous_optimize_objective(
        objective,
        init;
        method = :L_BFGS,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )

    fitted_sigma2 = exp(_continuous_result_minimizer(result)[1])
    profile = _bm1_profile(tree, tr, fitted_sigma2)

    return ContinuousFitResult(
        model = :BM1,
        success = profile.success,
        loglik = profile.loglik,
        aic = profile.success ? compute_aic(profile.loglik, 2) : Inf,
        nparams = 2,
        trait_name = _continuous_checked_trait_name(trait_name),
        sigma2 = fitted_sigma2,
        theta = profile.root_state,
        root_state = profile.root_state,
        converged = _continuous_result_converged(result),
        iterations = _continuous_result_iterations(result),
        f_calls = _continuous_result_f_calls(result),
    )
end

function fit_bm1(
    tree::CompactTree,
    data::AbstractDataFrame;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_bm1(tree, trait; trait_name = trait_name, kwargs...)
end

function bm1_loglikelihood(
    tree::CompactTree,
    data::AbstractDataFrame,
    sigma2::Real;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return bm1_loglikelihood(tree, trait, sigma2; trait_name = trait_name, kwargs...)
end

