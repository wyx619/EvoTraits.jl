function _shift_sr_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    sigma2::AbstractVector{<:Real};
    age = nothing,
    mapped_edge = nothing,
    precalc = nothing,
    trait_name = nothing,
    theta::Union{Nothing, Real} = nothing,
    Sigma_decomp::Symbol = :cholesky,
)
    _shift_sigma_nparams(1, Sigma_decomp)
    length(sigma2) == 2 || throw(ArgumentError("SR requires early and late diffusion rates"))
    mapping = _shift_mapping(tree, 1; age = age, mapped_edge = mapped_edge, precalc = precalc)
    if theta !== nothing
        mean = Float64(theta)
        isfinite(mean) || throw(ArgumentError("theta must be finite"))
        variances = _bmm_edge_variances(mapping.mapped_edge, sigma2)
        prof = _linear_gaussian_loglik(tree, trait, ones(tree.nedges), zeros(tree.nedges), variances;
            root_prior_mean = mean, root_prior_var = 0.0, profile_root = false)
        fit = ContinuousMultiRegimeResult(
            model = :BMM, success = prof.success, loglik = prof.loglik,
            aic = prof.success ? compute_aic(prof.loglik, 3) : Inf,
            nparams = 3, trait_name = _continuous_checked_trait_name(trait_name),
            regime_names = ["early", "late"], sigma2 = Float64.(sigma2),
            theta = mean, root_state = mean, nregimes = 2)
        return merge(_shift_sr_result(fit, mapping, count(!isnan, trait)), (Sigma_decomp = Sigma_decomp,))
    end
    fit = bmm_loglikelihood(tree, trait, mapping.mapped_edge, sigma2;
        trait_name = trait_name, regime_names = ["early", "late"])
    return merge(_shift_sr_result(fit, mapping, count(!isnan, trait)), (Sigma_decomp = Sigma_decomp,))
end

function _shift_eb_loglikelihood(tree::CompactTree, trait::AbstractVector{<:Real}, sigma2::Real;
    alpha::Union{Nothing, Real} = nothing, sig::Union{Nothing, Real} = nothing,
    theta::Union{Nothing, Real} = nothing, trait_name = nothing, kwargs...)
    data = _validate_univariate_trait_allow_missing(tree, trait)
    matrix(x) = x === nothing ? nothing : reshape([Float64(x)], 1, 1)
    result = _mvshift_eb_loglikelihood(tree, reshape(data, :, 1), matrix(sigma2);
        A = matrix(alpha), sig = matrix(sig), theta = theta === nothing ? nothing : [Float64(theta)],
        trait_names = [_continuous_checked_trait_name(trait_name)], kwargs...)
    return merge(result, (theta = only(result.theta), alpha = alpha, sigma2 = Float64(sigma2), sig = only(result.sig)))
end

function _shift_oubm_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    alpha::Real,
    sigma2::Real;
    sig::Union{Nothing, Real} = nothing,
    theta::Union{Nothing, Real} = nothing,
    trait_name = nothing,
    kwargs...,
)
    data = _validate_univariate_trait_allow_missing(tree, trait)
    result = _mvshift_oubm_loglikelihood(tree, reshape(data, :, 1),
        reshape([Float64(alpha)], 1, 1), reshape([Float64(sigma2)], 1, 1);
        sig = sig === nothing ? nothing : reshape([Float64(sig)], 1, 1),
        theta = theta === nothing ? nothing : [Float64(theta)],
        trait_names = [_continuous_checked_trait_name(trait_name)], kwargs...)
    return merge(result, (theta = only(result.theta), alpha = Float64(alpha),
        sigma2 = Float64(sigma2), sig = only(result.sig)))
end

function _fit_shift_sr(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    age = nothing,
    mapped_edge = nothing,
    precalc = nothing,
    Sigma_decomp::Symbol = :cholesky,
    kwargs...,
)
    _shift_sigma_nparams(1, Sigma_decomp)
    mapping = _shift_mapping(tree, 1; age = age, mapped_edge = mapped_edge, precalc = precalc)
    fit = fit_bmm(tree, trait, mapping.mapped_edge;
        regime_names = ["early", "late"], kwargs...)
    return merge(_shift_sr_result(fit, mapping, count(!isnan, trait)), (Sigma_decomp = Sigma_decomp,))
end

function _fit_shift_oubm(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    guess_alpha::Union{Nothing, Real} = nothing,
    guess_sigma2::Union{Nothing, Real} = nothing,
    guess_sig::Union{Nothing, Real} = nothing,
    trait_name = nothing,
    kwargs...,
)
    data = _validate_univariate_trait_allow_missing(tree, trait)
    matrix(x) = x === nothing ? nothing : reshape([Float64(x)], 1, 1)
    fit = _fit_mvshift_oubm(tree, reshape(data, :, 1);
        guess_A = matrix(guess_alpha), guess_sigma = matrix(guess_sigma2),
        guess_sig = matrix(guess_sig), trait_names = [_continuous_checked_trait_name(trait_name)], kwargs...)
    return merge(fit, (alpha = only(fit.A), sigma2 = only(fit.sigma),
        sig = only(fit.sig), theta = only(fit.theta)))
end
function _fit_shift_eb(tree::CompactTree, trait::AbstractVector{<:Real};
    guess_alpha::Union{Nothing, Real} = nothing,
    guess_sigma2::Union{Nothing, Real} = nothing,
    guess_sig::Union{Nothing, Real} = nothing,
    trait_name = nothing, kwargs...)
    data = _validate_univariate_trait_allow_missing(tree, trait)
    matrix(x) = x === nothing ? nothing : reshape([Float64(x)], 1, 1)
    fit = _fit_mvshift_eb(tree, reshape(data, :, 1);
        guess_A = matrix(guess_alpha), guess_sigma = matrix(guess_sigma2),
        guess_sig = matrix(guess_sig), trait_names = [_continuous_checked_trait_name(trait_name)], kwargs...)
    return merge(fit, (alpha = fit.A === nothing ? nothing : only(fit.A),
        sigma2 = only(fit.sigma), sig = only(fit.sig), theta = only(fit.theta)))
end
