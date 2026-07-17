@inline function _eb_beta_from_raw(raw::Float64, low::Float64, up::Float64)
    low < up || throw(ArgumentError("EB bounds require low < up"))
    return low + (up - low) / (1.0 + exp(-raw))
end

@inline function _eb_branch_variance_increment(parent_time::Float64, child_time::Float64, sigma2::Float64, beta::Float64)
    dt = child_time - parent_time
    dt >= 0.0 || return NaN
    if abs(beta) < 1e-12
        return sigma2 * dt
    end
    return sigma2 * (exp(beta * child_time) - exp(beta * parent_time)) / beta
end

function _eb_profile(tree::CompactTree, trait::AbstractVector{<:Real}, sigma2::Float64, beta::Float64)
    sigma2 > 0.0 || return (success = false, loglik = -Inf, root_state = NaN)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    y = _validate_univariate_trait_allow_missing(tree, trait)

    if any(isnan, y)
        edge_variances = Vector{Float64}(undef, tree.nedges)
        for edge in 1:tree.nedges
            parent = tree.parent_of_edge[edge]
            child = tree.child_of_edge[edge]
            edge_variances[edge] = _eb_branch_variance_increment(tree.dist_from_root[parent], tree.dist_from_root[child], sigma2, beta)
        end
        any(x -> !isfinite(x) || x < 0.0, edge_variances) && return (success = false, loglik = -Inf, root_state = NaN)
        prof = _linear_gaussian_loglik(
            tree,
            y,
            ones(Float64, tree.nedges),
            zeros(Float64, tree.nedges),
            edge_variances,
        )
        return (success = prof.success, loglik = prof.loglik, root_state = prof.root_state)
    end

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
        parent_time = tree.dist_from_root[node]
        child_time1 = tree.dist_from_root[child1]
        child_time2 = tree.dist_from_root[child2]

        inc1 = _eb_branch_variance_increment(parent_time, child_time1, sigma2, beta)
        inc2 = _eb_branch_variance_increment(parent_time, child_time2, sigma2, beta)
        (isfinite(inc1) && isfinite(inc2) && inc1 >= 0.0 && inc2 >= 0.0) || return (success = false, loglik = -Inf, root_state = NaN)

        v1 = spreads[child1] + inc1
        v2 = spreads[child2] + inc2
        denom = v1 + v2
        denom > 0.0 || return (success = false, loglik = -Inf, root_state = NaN)

        contrast = means[child1] - means[child2]
        loglik += -0.5 * (log(2 * pi * denom) + (contrast * contrast) / denom)

        invsum = 1.0 / denom
        means[node] = (means[child1] * v2 + means[child2] * v1) * invsum
        spreads[node] = (v1 * v2) * invsum
    end

    root = tree.root
    root_var = spreads[root]
    root_var > 0.0 || return (success = false, loglik = -Inf, root_state = NaN)
    loglik += -0.5 * log(2 * pi * root_var)

    return (success = isfinite(loglik), loglik = loglik, root_state = means[root])
end

function eb_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    sigma2::Real,
    beta::Real;
    trait_name = nothing,
)
    profile = _eb_profile(tree, trait, Float64(sigma2), Float64(beta))
    return ContinuousFitResult(
        model = :EB,
        success = profile.success,
        loglik = profile.loglik,
        aic = profile.success ? compute_aic(profile.loglik, 3) : Inf,
        nparams = 3,
        trait_name = _continuous_checked_trait_name(trait_name),
        sigma2 = Float64(sigma2),
        beta = Float64(beta),
        theta = profile.root_state,
        root_state = profile.root_state,
    )
end

function fit_eb(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    low::Union{Nothing, Real} = nothing,
    up::Real = 0.0,
    trait_name = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    tree_height = maximum(tree.dist_from_root)
    low_beta = low === nothing ? log(1e-5) / max(tree_height, 1e-8) : Float64(low)
    up_beta = Float64(up)
    low_beta < up_beta || throw(ArgumentError("EB requires low < up"))

    observed = filter(!isnan, tr)
    sigma20 = max(var(observed), 1e-8)
    beta0 = -1.0 / max(tree_height, 1e-8)
    beta_start = clamp(beta0, low_beta + 1e-8, up_beta - 1e-8)
    p0 = (beta_start - low_beta) / (up_beta - low_beta)
    init = [log(sigma20), log(p0 / (1.0 - p0))]

    objective = function (par)
        sigma2 = exp(par[1])
        beta = _eb_beta_from_raw(par[2], low_beta, up_beta)
        prof = _eb_profile(tree, tr, sigma2, beta)
        return prof.success ? -prof.loglik : Inf
    end

    result = _continuous_optimize_objective(
        objective,
        init;
        method = :L_BFGS,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )

    fitted = _continuous_result_minimizer(result)
    sigma2 = exp(fitted[1])
    beta = _eb_beta_from_raw(fitted[2], low_beta, up_beta)
    profile = _eb_profile(tree, tr, sigma2, beta)

    return ContinuousFitResult(
        model = :EB,
        success = profile.success,
        loglik = profile.loglik,
        aic = profile.success ? compute_aic(profile.loglik, 3) : Inf,
        nparams = 3,
        trait_name = _continuous_checked_trait_name(trait_name),
        sigma2 = sigma2,
        beta = beta,
        theta = profile.root_state,
        root_state = profile.root_state,
        converged = _continuous_result_converged(result),
        iterations = _continuous_result_iterations(result),
        f_calls = _continuous_result_f_calls(result),
    )
end

function fit_eb(
    tree::CompactTree,
    data::AbstractDataFrame;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_eb(tree, trait; trait_name = trait_name, kwargs...)
end

function eb_loglikelihood(
    tree::CompactTree,
    data::AbstractDataFrame,
    sigma2::Real,
    beta::Real;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return eb_loglikelihood(tree, trait, sigma2, beta; trait_name = trait_name, kwargs...)
end

