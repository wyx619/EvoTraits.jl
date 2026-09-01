function _ou_fit(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec;
    cache = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    nregimes = cache === nothing ? 1 : cache.nregimes

    # geiger's OU fit uses root="max": alpha and sigma2 are optimized while
    # the root state is profiled at every objective evaluation. Keep this
    # special path local to OU1 so multi-regime models retain their current
    # fixed-root semantics and parameter layout.
    if spec.model === :OU1
        return _ou1_profiled_fit(tree, tr; max_iterations = max_iterations, rel_tol = rel_tol)
    end

    init = _ou_initial_params(spec, nregimes; tree = tree, trait = tr)
    context = _ou_prepare_context(tree, tr, spec, cache)
    objective = function (par)
        bundle = _ou_unpack_params(spec, par, nregimes)
        prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
        return prof.success ? -prof.loglik : Inf
    end
    objective_factory = _ -> _ou_context_objective(context, OULikelihoodWorkspace(tree))
    lower_bounds = vcat(
        fill(1e-8, spec.alpha_mode === :shared ? 1 : nregimes),
        fill(1e-8, spec.sigma_mode === :shared ? 1 : nregimes),
        fill(-Inf, spec.theta_mode === :shared ? 1 : nregimes),
        spec.root_mean_mode === :free_theta0 ? [-Inf] : Float64[],
    )
    result = _ou_optimize_from_initial(
        objective,
        init,
        spec,
        nregimes,
        tree,
        tr,
        cache;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
        objective_factory = objective_factory,
    )

    minimizer = _continuous_result_minimizer(result)
    bundle = _ou_unpack_params(spec, minimizer, nregimes)
    prof = _ou_context_loglikelihood(context, bundle, OULikelihoodWorkspace(tree))
    return (bundle = bundle, profile = prof, result = result, nregimes = nregimes)
end

const _OU1_LOG_PARAMETER_LOWER = -500.0
const _OU1_LOG_ALPHA_UPPER = 5.0
const _OU1_LOG_SIGMA2_UPPER = 7.0

function _ou1_profiled_candidates(trait::AbstractVector{<:Real})
    observed = filter(!isnan, Float64.(trait))
    observed_var = max(var(observed), 1e-8)
    alpha_logs = [-8.0, -4.0, -1.0, 0.0, _OU1_LOG_ALPHA_UPPER]
    sigma_scales = [-1.0, 0.0, 1.0]
    candidates = Vector{Float64}[]
    for alpha_log in alpha_logs
        for sigma_scale in sigma_scales
            sigma_log = log(observed_var) + sigma_scale * log(10.0)
            push!(candidates, [alpha_log, clamp(sigma_log, _OU1_LOG_PARAMETER_LOWER, _OU1_LOG_SIGMA2_UPPER)])
        end
    end
    return candidates
end

function _ou1_profiled_fit(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    max_iterations::Integer,
    rel_tol::Float64,
)
    objective = function (par)
        alpha_log = clamp(Float64(par[1]), _OU1_LOG_PARAMETER_LOWER, _OU1_LOG_ALPHA_UPPER)
        sigma_log = clamp(Float64(par[2]), _OU1_LOG_PARAMETER_LOWER, _OU1_LOG_SIGMA2_UPPER)
        prof = _ou1_profiled_likelihood(tree, trait, exp(alpha_log), exp(sigma_log))
        return prof.success ? -prof.loglik : Inf
    end

    result = _continuous_two_stage_multistart_serial(
        objective,
        _ou1_profiled_candidates(trait);
        max_iterations = max_iterations,
        polish_iterations = 60,
        rel_tol = rel_tol,
        lower_bounds = [_OU1_LOG_PARAMETER_LOWER, _OU1_LOG_PARAMETER_LOWER],
    )

    minimizer = _continuous_result_minimizer(result)
    alpha_log = clamp(Float64(minimizer[1]), _OU1_LOG_PARAMETER_LOWER, _OU1_LOG_ALPHA_UPPER)
    sigma_log = clamp(Float64(minimizer[2]), _OU1_LOG_PARAMETER_LOWER, _OU1_LOG_SIGMA2_UPPER)
    fixed_profile = _ou1_profiled_likelihood(tree, trait, exp(alpha_log), exp(sigma_log))
    fixed_bundle = OUParameterBundle(theta = [fixed_profile.root_state], alpha = [exp(alpha_log)], sigma2 = [exp(sigma_log)])
    return (bundle = fixed_bundle, profile = fixed_profile, result = result, nregimes = 1)
end

function _ou1_profiled_likelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    alpha::Float64,
    sigma2::Float64,
)
    spec = ou_spec(:OU1)
    observed = filter(!isnan, Float64.(trait))
    center = mean(observed)
    step = max(sqrt(var(observed)), 1.0)
    eval_at = function (theta)
        bundle = OUParameterBundle(theta = [theta], alpha = [alpha], sigma2 = [sigma2])
        return _ou_loglikelihood(tree, trait, spec, bundle)
    end

    at_center = eval_at(center)
    at_plus = eval_at(center + step)
    at_minus = eval_at(center - step)
    at_center.success && at_plus.success && at_minus.success ||
        return (success = false, loglik = -Inf, root_state = NaN)

    curvature = (2.0 * at_center.loglik - at_plus.loglik - at_minus.loglik) / (step * step)
    curvature > 0.0 && isfinite(curvature) || return (success = false, loglik = -Inf, root_state = NaN)
    slope = (at_plus.loglik - at_minus.loglik) / (2.0 * step)
    theta = center + slope / curvature
    prof = eval_at(theta)
    return (success = prof.success, loglik = prof.loglik, root_state = theta)
end

function _ou_fit_with_starts(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec;
    cache = nothing,
    method::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    start_alpha::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_sigma2::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_theta::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)

    nregimes = cache === nothing ? 1 : cache.nregimes
    init = _ou_initial_params_from_starts(
        spec,
        nregimes;
        tree = tree,
        trait = tr,
        start_alpha = start_alpha,
        start_sigma2 = start_sigma2,
        start_theta = start_theta,
        start_theta_regimes = start_theta_regimes,
    )
    objective = function (par)
        bundle = _ou_unpack_params(spec, par, nregimes)
        prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
        return prof.success ? -prof.loglik : Inf
    end
    lower_bounds = vcat(
        fill(1e-8, spec.alpha_mode === :shared ? 1 : nregimes),
        fill(1e-8, spec.sigma_mode === :shared ? 1 : nregimes),
        fill(-Inf, spec.theta_mode === :shared ? 1 : nregimes),
        spec.root_mean_mode === :free_theta0 ? [-Inf] : Float64[],
    )
    result =
        if method in (:SBPLX_L_BFGS, :LN_SBPLX_L_BFGS, :TWO_STAGE)
            _ou_optimize_from_initial(
                objective,
                init,
                spec,
                nregimes,
                tree,
                tr,
                cache;
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        elseif method === :LN_SBPLX
            _continuous_optimize_objective(
                objective,
                init;
                method = :LN_SBPLX,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        else
            method === :L_BFGS || throw(ArgumentError("Unsupported internal method=$method"))
            _continuous_optimize_objective(
                objective,
                init;
                method = :L_BFGS,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end

    minimizer = _continuous_result_minimizer(result)
    bundle = _ou_unpack_params(spec, minimizer, nregimes)
    prof = _ou_loglikelihood(tree, tr, spec, bundle; cache = cache)
    return (bundle = bundle, profile = prof, result = result, nregimes = nregimes)
end
