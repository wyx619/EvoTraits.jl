function _mveb_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    beta::Float64,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    Sigma_mat = Matrix{Float64}(Sigma)
    p = size(data, 2)
    size(Sigma_mat, 1) == size(Sigma_mat, 2) == p || throw(ArgumentError("Sigma dimensions must match trait dimension"))
    issymmetric(Sigma_mat) || return (success = false, loglik = -Inf, theta = Float64[])
    try
        cholesky(Symmetric(Sigma_mat))
    catch
        return (success = false, loglik = -Inf, theta = Float64[])
    end
    edge_Q = workspace === nothing ? Array{Float64, 3}(undef, p, p, tree.nedges) : workspace.edge_Q
    size(edge_Q) == (p, p, tree.nedges) || throw(ArgumentError("edge_Q workspace has incompatible dimensions"))
    @inbounds for e in 1:tree.nedges
        parent = tree.parent_of_edge[e]
        child = tree.child_of_edge[e]
        inc = _eb_branch_variance_increment(tree.dist_from_root[parent], tree.dist_from_root[child], 1.0, beta)
        for j in 1:p, i in 1:p
            edge_Q[i, j, e] = inc * Sigma_mat[i, j]
        end
    end
    return _mvbm_edge_q_profile(tree, data, edge_Q, Sigma_mat; workspace = workspace)
end

function _mveb_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousBMResult,
)
    data = _validate_multivariate_trait(tree, trait)
    fit.success || throw(ArgumentError("mvEB ASR requires a successful fit"))
    p = fit.ntraits
    centered = data .- reshape(fit.theta, 1, p)
    edge_Phi = Array{Float64, 3}(undef, p, p, tree.nedges)
    edge_Q = Array{Float64, 3}(undef, p, p, tree.nedges)
    I2 = Matrix{Float64}(I, p, p)
    for e in 1:tree.nedges
        edge_Phi[:, :, e] .= I2
        parent = tree.parent_of_edge[e]
        child = tree.child_of_edge[e]
        inc = _eb_branch_variance_increment(tree.dist_from_root[parent], tree.dist_from_root[child], 1.0, fit.beta)
        edge_Q[:, :, e] .= inc .* fit.sigma
    end
    return _mv_recursive_asr(
        tree,
        centered,
        edge_Phi,
        edge_Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        shift = fit.theta,
        model = :mvEB,
    )
end

@inline function _bounded_param_from_raw(raw::Float64, lower::Float64, upper::Float64)
    lower < upper || throw(ArgumentError("bounded parameter requires lower < upper"))
    return lower + (upper - lower) / (1.0 + exp(-raw))
end

@inline function _bounded_param_to_raw(value::Float64, lower::Float64, upper::Float64)
    clamped = clamp(value, lower + 1e-8, upper - 1e-8)
    p = (clamped - lower) / (upper - lower)
    return log(p / (1.0 - p))
end

@inline function _mv_transform_result(model_sym::Symbol, profile, Sigma::AbstractMatrix{<:Real}, p::Integer, param_name::Symbol, param::Float64; nobs::Integer, converged::Bool = false, iterations::Int = 0, f_calls::Int = 0, trait_names = nothing)
    param_name === :beta || throw(ArgumentError("Unsupported transformed mvBM parameter $param_name"))
    nparams = div(p * (p + 1), 2) + p + 1
    return MVContinuousBMResult(
        model = model_sym,
        backend = :tree_pruning,
        success = profile.success,
        loglik = profile.loglik,
        aic = profile.success ? compute_aic(profile.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = Int(nobs),
        ntraits = p,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        sigma = Matrix{Float64}(Sigma),
        theta = profile.theta,
        beta = param,
        converged = converged,
        iterations = iterations,
        f_calls = f_calls,
    )
end

"""
    mveb_loglikelihood(tree, trait, Sigma, beta)

Evaluate the multivariate early-burst (`mvEB`) log-likelihood for a fixed shared
trait covariance matrix `Sigma` and a shared global `beta`.
"""
function mveb_loglikelihood(tree::CompactTree, trait::AbstractMatrix{<:Real}, Sigma::AbstractMatrix{<:Real}, beta::Real; trait_names = nothing)
    _validate_ultrametric_tree(tree)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    prof = _mveb_profile(tree, data, Sigma, Float64(beta))
    return _mv_transform_result(:mvEB, prof, Sigma, p, :beta, Float64(beta); nobs = count(!isnan, data), trait_names = trait_names)
end

"""
    fit_mveb(tree, trait; kwargs...)

Fit the multivariate early-burst (`mvEB`) model by maximum likelihood with a
shared trait covariance matrix and a shared global `beta`.
"""
function fit_mveb(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 1000,
    rel_tol::Float64 = 1e-6,
    guess_sigma::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    guess_beta::Union{Nothing, Real} = nothing,
    low::Union{Nothing, Real} = nothing,
    up::Real = 0.0,
    trait_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    p = size(data, 2)
    tree_height = maximum(tree.dist_from_root)
    low_beta = low === nothing ? log(1e-5) / max(tree_height, 1e-8) : Float64(low)
    up_beta = Float64(up)
    sigma0 = guess_sigma === nothing ? (_mv_complete_rows_cov(data) + 1e-6 * Matrix{Float64}(I, p, p)) : Matrix{Float64}(guess_sigma)
    sigma0 = (sigma0 + sigma0') / 2 + 1e-8 * Matrix{Float64}(I, p, p)
    beta0 = guess_beta === nothing ? -1.0 / max(tree_height, 1e-8) : clamp(Float64(guess_beta), low_beta + 1e-8, up_beta - 1e-8)
    p0 = vcat(_mvbm1_pack_sigma(sigma0), _bounded_param_to_raw(beta0, low_beta, up_beta))
    workspace = _mv_profile_workspace(tree, p)
    objective = function (pars)
        Sigma = _mvbm1_unpack_sigma(pars[1:(div(p * (p + 1), 2))], p)
        beta = _bounded_param_from_raw(pars[end], low_beta, up_beta)
        prof = _mveb_profile(tree, data, Sigma, beta, workspace)
        return prof.success ? -prof.loglik : Inf
    end
    result = _mvbm_optimize_objective(
        objective,
        p0;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = vcat(_mvbm_sigma_lower_bounds(p, 1), -Inf),
    )
    minimizer = _mvbm_result_minimizer(result)
    Sigma = _mvbm1_unpack_sigma(minimizer[1:(div(p * (p + 1), 2))], p)
    beta = _bounded_param_from_raw(minimizer[end], low_beta, up_beta)
    prof = mveb_loglikelihood(tree, data, Sigma, beta; trait_names = trait_names)
    return MVContinuousBMResult(
        model = prof.model,
        backend = prof.backend,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.aic,
        nparams = prof.nparams,
        nobs = prof.nobs,
        ntraits = prof.ntraits,
        trait_names = prof.trait_names,
        sigma = prof.sigma,
        theta = prof.theta,
        beta = prof.beta,
        converged = _mvbm_result_converged(result),
        iterations = _mvbm_result_iterations(result),
        f_calls = _mvbm_result_f_calls(result),
    )
end

function fit_mveb(
    tree::CompactTree,
    data::AbstractDataFrame;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mveb(tree, trait; trait_names = trait_names, kwargs...)
end

