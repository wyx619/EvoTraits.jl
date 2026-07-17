function _mvbmm_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    Sigmas::AbstractVector{<:AbstractMatrix},
    mapped_edge::AbstractMatrix{<:Real};
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
    return_sigma::Bool = true,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    nregimes = size(mapped, 2)
    length(Sigmas) == nregimes || throw(ArgumentError("Sigmas must match regime count"))
    p = size(data, 2)
    Sigma_mats =
        all(S -> S isa Matrix{Float64}, Sigmas) ?
        Sigmas :
        [Matrix{Float64}(S) for S in Sigmas]
    return _mvbmm_profile_validated(tree, data, Sigma_mats, mapped; workspace = workspace, return_sigma = return_sigma)
end

function _mvbmm_profile_validated(
    tree::CompactTree,
    data::Matrix{Float64},
    Sigma_mats::AbstractVector{<:AbstractMatrix{Float64}},
    mapped::AbstractMatrix{<:Real};
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
    return_sigma::Bool = true,
)
    p = size(data, 2)
    nregimes = size(mapped, 2)
    length(Sigma_mats) == nregimes || throw(ArgumentError("Sigmas must match regime count"))
    all(size(S, 1) == p && size(S, 2) == p for S in Sigma_mats) || throw(ArgumentError("All Sigmas must match trait dimension"))
    for S in Sigma_mats
        issymmetric(S) || return (success = false, loglik = -Inf, theta = zeros(p, nregimes))
        try
            cholesky(Symmetric(S))
        catch
            return (success = false, loglik = -Inf, theta = zeros(p, nregimes))
        end
    end

    sigma_array = return_sigma ? cat(Sigma_mats...; dims = 3) : nothing
    edge_Q = workspace === nothing ? Array{Float64, 3}(undef, p, p, tree.nedges) : workspace.edge_Q
    size(edge_Q) == (p, p, tree.nedges) || throw(ArgumentError("edge_Q workspace has incompatible dimensions"))
    @inbounds for e in 1:tree.nedges
        for j in 1:p, i in 1:p
            v = 0.0
            for r in 1:nregimes
                v += Float64(mapped[e, r]) * Sigma_mats[r][i, j]
            end
            edge_Q[i, j, e] = v
        end
        for j in 1:p, i in (j + 1):p
            v = 0.5 * (edge_Q[i, j, e] + edge_Q[j, i, e])
            edge_Q[i, j, e] = v
            edge_Q[j, i, e] = v
        end
    end
    prof = _mvbm_edge_q_profile(tree, data, edge_Q, sigma_array; workspace = workspace)
    return (
        success = prof.success,
        loglik = prof.loglik,
        theta = reshape(prof.theta, 1, p),
        sigma = prof.sigma,
    )
end

function _mvbmm_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousMultiBMResult,
    mapped_edge::AbstractMatrix{<:Real},
)
    data = _validate_multivariate_trait(tree, trait)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    fit.success || throw(ArgumentError("mvBMM ASR requires a successful fit"))
    p = fit.ntraits
    size(data, 2) == p || throw(ArgumentError("Trait dimension does not match fit"))
    nregimes = fit.nregimes
    edge_Phi = Array{Float64, 3}(undef, p, p, tree.nedges)
    edge_Q = Array{Float64, 3}(undef, p, p, tree.nedges)
    I_p = Matrix{Float64}(I, p, p)
    for e in 1:tree.nedges
        edge_Phi[:, :, e] .= I_p
        fill!(view(edge_Q, :, :, e), 0.0)
        for r in 1:nregimes
            edge_Q[:, :, e] .+= Float64(mapped[e, r]) .* fit.sigma[:, :, r]
        end
        edge_Q[:, :, e] .= (edge_Q[:, :, e] .+ edge_Q[:, :, e]') ./ 2
    end
    centered = data .- reshape(vec(fit.theta), 1, p)
    return _mv_recursive_asr(
        tree,
        centered,
        edge_Phi,
        edge_Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        shift = vec(fit.theta),
        model = :mvBMM,
    )
end

"""
    mvbmm_loglikelihood(tree, trait, mapped_edge, Sigmas)

Evaluate the multivariate multi-regime Brownian-motion (`mvBMM`) log-likelihood
for fixed regime-specific trait covariance matrices `Sigmas`. The root mean
vectors are profiled by generalized least squares.
"""
function mvbmm_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    mapped_edge::AbstractMatrix{<:Real},
    Sigmas::AbstractVector{<:AbstractMatrix};
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    prof = _mvbmm_profile(tree, data, Sigmas, mapped_edge)
    p = size(data, 2)
    nregimes = size(mapped_edge, 2)
    nsigma = nregimes * (div(p * (p + 1), 2))
    nparams = nsigma + p
    return MVContinuousMultiBMResult(
        model = :mvBMM,
        backend = :tree_pruning,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, nregimes, "regime"),
        sigma = prof.success ? prof.sigma : zeros(p, p, nregimes),
        theta = prof.theta,
    )
end

"""
    fit_mvbmm(tree, trait, mapped_edge; kwargs...)

Fit the multivariate multi-regime Brownian-motion (`mvBMM`) model by maximum
likelihood using one trait covariance matrix per regime and profiled
regime-specific root mean vectors.
"""
function fit_mvbmm(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    mapped_edge::AbstractMatrix{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 1000,
    rel_tol::Float64 = 1e-6,
    guess_sigma::Union{Nothing, AbstractVector{<:AbstractMatrix}} = nothing,
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    nt, p = size(data)
    nregimes = size(mapped, 2)

    sigma0 =
        if guess_sigma === nothing
            empirical = _mv_complete_rows_cov(data)
            empirical = (empirical + empirical') / 2 + 1e-6 * Matrix{Float64}(I, p, p)
            [copy(empirical) for _ in 1:nregimes]
        else
            length(guess_sigma) == nregimes || throw(ArgumentError("guess_sigma must have one matrix per regime"))
            [Matrix{Float64}(S) for S in guess_sigma]
        end
    for S in sigma0
        size(S, 1) == p == size(S, 2) || throw(ArgumentError("All guess_sigma matrices must match trait dimension"))
        S .= (S + S') / 2
        S .+= 1e-8 * Matrix{Float64}(I, p, p)
    end
    p0 = _mvbm_pack_sigmas(sigma0)
    workspace = _mv_profile_workspace(tree, p)
    sigma_workspace = [zeros(Float64, p, p) for _ in 1:nregimes]
    sigma_factor_work = zeros(Float64, p, p)

    objective = function (pars)
        _mvbm_unpack_sigmas!(sigma_workspace, pars, p, nregimes, sigma_factor_work)
        prof = _mvbmm_profile_validated(tree, data, sigma_workspace, mapped; workspace = workspace, return_sigma = false)
        return prof.success ? -prof.loglik : Inf
    end

    result = _mvbm_optimize_objective(
        objective,
        p0;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = _mvbm_sigma_lower_bounds(p, nregimes),
    )

    fitted_sigmas = _mvbm_unpack_sigmas(_mvbm_result_minimizer(result), p, nregimes)
    prof = _mvbmm_profile(tree, data, fitted_sigmas, mapped)
    nsigma = nregimes * (div(p * (p + 1), 2))
    nparams = nsigma + p
    return MVContinuousMultiBMResult(
        model = :mvBMM,
        backend = :tree_pruning,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, nregimes, "regime"),
        sigma = prof.sigma,
        theta = prof.theta,
        converged = _mvbm_result_converged(result),
        iterations = _mvbm_result_iterations(result),
        f_calls = _mvbm_result_f_calls(result),
    )
end

function fit_mvbmm(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvbmm(tree, trait, simmap.mapped_edge; trait_names = trait_names, regime_names = simmap.state_labels, kwargs...)
end

function fit_mvbmm(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    simmap::SimmapSample;
    kwargs...,
)
    return fit_mvbmm(tree, trait, simmap.mapped_edge; regime_names = simmap.state_labels, kwargs...)
end

