function _mvbm1_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
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
        scale = tree.edge_length[e]
        for j in 1:p, i in 1:p
            edge_Q[i, j, e] = scale * Sigma_mat[i, j]
        end
    end
    return _mvbm_edge_q_profile(tree, data, edge_Q, Sigma_mat; workspace = workspace)
end

function _mvbm1_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousBMResult,
)
    data = _validate_multivariate_trait(tree, trait)
    fit.success || throw(ArgumentError("mvBM1 ASR requires a successful fit"))
    p = fit.ntraits
    centered = data .- reshape(fit.theta, 1, p)
    edge_Phi = Array{Float64, 3}(undef, p, p, tree.nedges)
    edge_Q = Array{Float64, 3}(undef, p, p, tree.nedges)
    I2 = Matrix{Float64}(I, p, p)
    for e in 1:tree.nedges
        edge_Phi[:, :, e] .= I2
        edge_Q[:, :, e] .= tree.edge_length[e] .* fit.sigma
    end
    return _mv_recursive_asr(
        tree,
        centered,
        edge_Phi,
        edge_Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        shift = fit.theta,
        model = :mvBM1,
    )
end

"""
    mvbm1_loglikelihood(tree, trait, Sigma)

Evaluate the multivariate Brownian-motion (`mvBM1`) log-likelihood for a fixed
shared trait diffusion covariance matrix `Sigma`. The root mean vector is
profiled by generalized least squares.
"""
function mvbm1_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real};
    trait_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    prof = _mvbm1_profile(tree, data, Sigma)
    p = size(data, 2)
    nsigma = div(p * (p + 1), 2)
    nparams = nsigma + p
    return MVContinuousBMResult(
        model = :mvBM1,
        backend = :tree_pruning,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        sigma = prof.success ? prof.sigma : zeros(p, p),
        theta = prof.theta,
    )
end

"""
    fit_mvbm1(tree, trait; kwargs...)

Fit the multivariate Brownian-motion (`mvBM1`) model by maximum likelihood.
The current production path uses tree-pruning Gaussian messages with
a profiled root mean vector and a Cholesky parameterization for the shared trait
covariance matrix.
"""
function fit_mvbm1(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 1000,
    rel_tol::Float64 = 1e-6,
    guess_sigma::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    trait_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    _, p = size(data)

    sigma0 =
        if guess_sigma === nothing
            _mv_complete_rows_cov(data) + 1e-6 * Matrix{Float64}(I, p, p)
        else
            Matrix{Float64}(guess_sigma)
        end
    size(sigma0, 1) == p == size(sigma0, 2) || throw(ArgumentError("guess_sigma dimensions must match trait matrix"))
    sigma0 = Matrix{Float64}((sigma0 + sigma0') / 2)
    sigma0 .+= 1e-8 * Matrix{Float64}(I, p, p)
    p0 = _mvbm1_pack_sigma(sigma0)
    workspace = _mv_profile_workspace(tree, p)

    objective = function (pars)
        Sigma = _mvbm1_unpack_sigma(pars, p)
        prof = _mvbm1_profile(tree, data, Sigma, workspace)
        return prof.success ? -prof.loglik : Inf
    end

    result = _mvbm_optimize_objective(
        objective,
        p0;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = _mvbm_sigma_lower_bounds(p, 1),
    )

    fitted_sigma = _mvbm1_unpack_sigma(_mvbm_result_minimizer(result), p)
    prof = _mvbm1_profile(tree, data, fitted_sigma)
    nsigma = div(p * (p + 1), 2)
    nparams = nsigma + p
    return MVContinuousBMResult(
        model = :mvBM1,
        backend = :tree_pruning,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        sigma = fitted_sigma,
        theta = prof.theta,
        converged = _mvbm_result_converged(result),
        iterations = _mvbm_result_iterations(result),
        f_calls = _mvbm_result_f_calls(result),
    )
end

function fit_mvbm1(
    tree::CompactTree,
    data::AbstractDataFrame;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvbm1(tree, trait; trait_names = trait_names, kwargs...)
end

