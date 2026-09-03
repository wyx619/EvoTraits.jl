"""
    ou1_loglikelihood(tree, trait, alpha, sigma2, theta)

Evaluate the single-optimum Ornstein-Uhlenbeck (`OU1`) log-likelihood under the
unified OU engine. `alpha`, `sigma2`, and `theta` are treated as fixed inputs,
and the returned result records the current OU root-mode metadata.
"""
function ou1_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    alpha::Real,
    sigma2::Real,
    theta::Real,
    ;
    root_cov_mode::Symbol = :fixed,
    trait_name = nothing,
)
    spec = ou_spec(:OU1; root_cov_mode = root_cov_mode)
    bundle = OUParameterBundle(theta = [Float64(theta)], alpha = [Float64(alpha)], sigma2 = [Float64(sigma2)])
    prof = _ou_loglikelihood(tree, trait, spec, bundle)
    return ContinuousFitResult(
        model = :OU1,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, 3) : Inf,
        nparams = 3,
        trait_name = _continuous_checked_trait_name(trait_name),
        root_treatment = spec.root_cov_mode,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
        sigma2 = bundle.sigma2[1],
        alpha = bundle.alpha[1],
        theta = bundle.theta[1],
        root_state = prof.root_state,
    )
end

"""
    fit_ou1(tree, trait; kwargs...)

Fit the single-trait `OU1` model by maximum likelihood using the unified OU
engine. Keyword arguments provide optimizer controls and optional trait naming.
"""
function fit_ou1(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    max_iterations::Integer = 300,
    rel_tol::Float64 = 1e-5,
    root_cov_mode::Symbol = :fixed,
    trait_name = nothing,
)
    spec = ou_spec(:OU1; root_cov_mode = root_cov_mode)
    fit = _ou_fit(
        tree,
        trait,
        spec;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )
    theta = fit.bundle.theta[1]
    return ContinuousFitResult(
        model = :OU1,
        success = fit.profile.success,
        loglik = fit.profile.loglik,
        aic = fit.profile.success ? compute_aic(fit.profile.loglik, 3) : Inf,
        nparams = 3,
        trait_name = _continuous_checked_trait_name(trait_name),
        root_treatment = spec.root_cov_mode,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
        sigma2 = fit.bundle.sigma2[1],
        alpha = fit.bundle.alpha[1],
        theta = theta,
        root_state = fit.profile.root_state,
        converged = _continuous_result_converged(fit.result),
        iterations = _continuous_result_iterations(fit.result),
        f_calls = _continuous_result_f_calls(fit.result),
    )
end

function fit_ou1(
    tree::CompactTree,
    data::AbstractDataFrame;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_ou1(tree, trait; trait_name = trait_name, kwargs...)
end

function ou1_loglikelihood(
    tree::CompactTree,
    data::AbstractDataFrame,
    alpha::Real,
    sigma2::Real,
    theta::Real;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return ou1_loglikelihood(tree, trait, alpha, sigma2, theta; trait_name = trait_name, kwargs...)
end

