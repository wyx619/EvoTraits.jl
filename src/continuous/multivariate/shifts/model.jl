function _mvshift_sr_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    Sigmas::AbstractVector{<:AbstractMatrix};
    age = nothing,
    mapped_edge = nothing,
    precalc = nothing,
    trait_names = nothing,
    theta::Union{Nothing, AbstractVector{<:Real}} = nothing,
    Sigma_decomp::Symbol = :cholesky,
)
    length(Sigmas) == 2 || throw(ArgumentError("SR requires early and late diffusion matrices"))
    _shift_sigma_nparams(size(trait, 2), Sigma_decomp)
    if Sigma_decomp !== :cholesky
        foreach(S -> _shift_sigma(S, size(trait, 2), Sigma_decomp), Sigmas)
    end
    mapping = _shift_mapping(tree, size(trait, 2); age = age, mapped_edge = mapped_edge, precalc = precalc)
    if theta !== nothing
        data = _validate_multivariate_trait(tree, trait)
        p = size(data, 2)
        length(theta) == p || throw(ArgumentError("theta must match trait dimension"))
        mean = Vector{Float64}(theta)
        all(isfinite, mean) || throw(ArgumentError("theta must be finite"))
        workspace = _mv_profile_workspace(tree, p)
        prof = _mvbmm_profile(tree, data, Sigmas, mapping.mapped_edge; workspace = workspace)
        # Evaluate the root message at the supplied mean, without a second tree traversal.
        root = Int(tree.root)
        ll = prof.success ? workspace.logconst[root] -
            0.5 * dot(mean, workspace.precision[root] * mean) +
            dot(workspace.linear[root], mean) : -Inf
        success = prof.success && isfinite(ll)
        nparams = p + 2 * _shift_sigma_nparams(p, Sigma_decomp)
        fit = MVContinuousMultiBMResult(
            model = :mvBMM, backend = :tree_pruning, success = success, loglik = ll,
            aic = success ? compute_aic(ll, nparams) : Inf,
            nparams = nparams, nobs = count(!isnan, data), ntraits = p, nregimes = 2,
            trait_names = _mv_checked_names(trait_names, p, "trait"),
            regime_names = ["early", "late"],
            sigma = success ? prof.sigma : zeros(p, p, 2), theta = reshape(mean, 1, p))
        return merge(_shift_sr_result(fit, mapping, count(!isnan, data)), (Sigma_decomp = Sigma_decomp,))
    end
    fit = mvbmm_loglikelihood(tree, trait, mapping.mapped_edge, Sigmas;
        trait_names = trait_names, regime_names = ["early", "late"])
    if Sigma_decomp !== :cholesky
        values = (; (name => getfield(fit, name) for name in fieldnames(typeof(fit)))...)
        nparams = size(trait, 2) + 2 * _shift_sigma_nparams(size(trait, 2), Sigma_decomp)
        fit = MVContinuousMultiBMResult(; merge(values, (nparams = nparams,
            aic = fit.success ? compute_aic(fit.loglik, nparams) : Inf))...)
    end
    return merge(_shift_sr_result(fit, mapping, count(!isnan, trait)), (Sigma_decomp = Sigma_decomp,))
end

"""
    _mvshift_process_loglikelihood(tree, trait, A, sigma; model=:OUBM, ...)

Internal fixed-parameter OU/BM and EB shift evaluation. `sigma` is the OU
diffusion when present, otherwise BM; independent models use `sig` for the
remaining BM or EB component, regardless of period order.
The observed trait is the sum of two independent process components and a
constant mean. This is not a sequential switch of a single process.
"""
function _mvshift_process_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    A::Union{Nothing, AbstractMatrix{<:Real}},
    sigma::AbstractMatrix{<:Real};
    model::Symbol = :OUBM,
    sig::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    age = nothing,
    mapped_edge = nothing,
    precalc = nothing,
    theta::Union{Nothing, AbstractVector{<:Real}} = nothing,
    A_decomp::Symbol = :cholesky,
    Sigma_decomp::Symbol = :cholesky,
    beta::Union{Nothing, Real} = nothing,
    beta_bounds = (-2.0, 2.0),
    trait_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    spec = _shift_process_spec(model)
    mapping = _shift_mapping(tree, p; age = age, mapped_edge = mapped_edge, precalc = precalc)
    factor = if spec.ou_period > 0
        A === nothing && throw(ArgumentError("$(spec.model) requires A"))
        _shift_ou_factor(A, p, A_decomp)
    else
        A === nothing || throw(ArgumentError("$(spec.model) has no OU component; omit A"))
        nothing
    end
    beta_value = if spec.eb_period > 0
        beta === nothing && throw(ArgumentError("$(spec.model) requires beta"))
        isfinite(beta) || throw(ArgumentError("beta must be finite"))
        _shift_effective_beta(beta, _shift_beta_bounds(beta_bounds))
    else
        beta === nothing || throw(ArgumentError("$(spec.model) has no EB component; omit beta"))
        0.0
    end
    primary = _shift_sigma(sigma, p, Sigma_decomp)
    if spec.independent
        sig === nothing && throw(ArgumentError("$(spec.model) requires a second diffusion matrix sig"))
        secondary = _shift_sigma(sig, p, Sigma_decomp)
    else
        sig === nothing || throw(ArgumentError("$(spec.model) shares diffusion; omit sig or use an independent model"))
        secondary = primary
    end
    mean = theta === nothing ? nothing : Vector{Float64}(theta)
    if mean !== nothing
        length(mean) == p && all(isfinite, mean) || throw(ArgumentError("theta must be finite and match trait dimension"))
    end
    names = _mv_checked_names(trait_names, p, "trait")
    prof = _shift_oubm_profile(tree, data, mapping, spec, factor, primary, secondary; theta = mean, beta = beta_value)
    nsigma = _shift_sigma_nparams(p, Sigma_decomp)
    nalpha = spec.ou_period > 0 ? _shift_A_nparams(p, A_decomp) : 0
    nparams = p + nalpha + nsigma * (spec.independent ? 2 : 1) + (spec.eb_period > 0 ? 1 : 0)
    nobs = count(!isnan, data)
    return (
        model = spec.model, input_model = model, backend = :tree_pruning,
        success = prof.success, converged = false, loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        aicc = prof.success ? compute_aicc(prof.loglik, nparams, nobs) : Inf,
        nparams = nparams, nobs = nobs, ntraits = p, trait_names = names,
        A = A === nothing ? nothing : Matrix{Float64}(A), A_decomp = spec.ou_period > 0 ? A_decomp : nothing,
        beta = spec.eb_period > 0 ? beta_value : nothing,
        input_beta = spec.eb_period > 0 ? Float64(beta) : nothing,
        beta_bounds = spec.eb_period > 0 ? beta_bounds : nothing,
        sigma = primary, sig = copy(secondary), Sigma_decomp = Sigma_decomp, theta = copy(prof.theta),
        root_cov_mode = :fixed, age = mapping.age, boundary = mapping.boundary,
        height = mapping.height, mapped_edge = copy(mapping.mapped_edge),
    )
end

function _mvshift_oubm_loglikelihood(tree::CompactTree, trait::AbstractMatrix{<:Real},
    A::AbstractMatrix{<:Real}, sigma::AbstractMatrix{<:Real}; model::Symbol = :OUBM, kwargs...)
    _shift_oubm_spec(model)
    return _mvshift_process_loglikelihood(tree, trait, A, sigma; model = model, kwargs...)
end

function _mvshift_eb_loglikelihood(tree::CompactTree, trait::AbstractMatrix{<:Real},
    sigma::AbstractMatrix{<:Real}; model::Symbol = :EBBM, A = nothing, beta::Real, kwargs...)
    spec = _shift_process_spec(model)
    spec.eb_period > 0 || throw(ArgumentError("An EB shift model is required"))
    return _mvshift_process_loglikelihood(tree, trait, A, sigma; model = model, beta = beta, kwargs...)
end

function _fit_mvshift_sr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    age = nothing,
    mapped_edge = nothing,
    precalc = nothing,
    Sigma_decomp::Symbol = :cholesky,
    kwargs...,
)
    mapping = _shift_mapping(tree, size(trait, 2); age = age, mapped_edge = mapped_edge, precalc = precalc)
    if Sigma_decomp !== :cholesky
        return _fit_mvshift_sr_constrained(tree, trait, mapping; Sigma_decomp = Sigma_decomp, kwargs...)
    end
    fit = fit_mvbmm(tree, trait, mapping.mapped_edge;
        regime_names = ["early", "late"], kwargs...)
    return merge(_shift_sr_result(fit, mapping, count(!isnan, trait)), (Sigma_decomp = Sigma_decomp,))
end
