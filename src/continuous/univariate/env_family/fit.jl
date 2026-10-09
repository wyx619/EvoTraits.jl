function _env_model(model)
    model in (:EnvExp,:EnvLin) || throw(ArgumentError("model must be :EnvExp or :EnvLin"))
    return model
end

function _env_profile(tree, data, cache, variances, sigma2, beta, model)
    _env_edge_variances!(variances,cache,sigma2,beta,model) ||
        return (success=false,loglik=-Inf,root_state=NaN,root_var=NaN)
    return _bm_pruning_profile_from_edge_variances(tree,data,variances)
end

function _env_result(tree,data,cache,variances,profile,sigma2,beta,model;
    trait_name=nothing,optimized=nothing,optimization=:none)
    nobs = count(!isnan,data)
    return EnvFitResult(;model,success=profile.success,loglik=profile.loglik,
        aic=profile.success ? compute_aic(profile.loglik,3) : Inf,
        aicc=profile.success ? compute_aicc(profile.loglik,3,nobs) : Inf,
        nobs,trait_name=_continuous_checked_trait_name(trait_name),sigma2,beta,
        theta=profile.root_state,root_state=profile.root_state,root_var=profile.root_var,
        converged=optimized !== nothing && profile.success && _continuous_result_converged(optimized),
        iterations=optimized === nothing ? 0 : _continuous_result_iterations(optimized),
        f_calls=optimized === nothing ? 0 : _continuous_result_f_calls(optimized),
        optimization,quadrature_order=cache.order,quadrature_breakpoints=copy(cache.breakpoints),height=cache.height,
        environment=cache.environment,
        edge_variances=profile.success ? copy(variances) : fill(NaN,tree.nedges))
end

"""
    t_env_loglikelihood(tree, trait, environment, sigma2, beta;
        model=:EnvExp, quadrature_order=16, trait_name=nothing)

Evaluate environmental BM using existing tree-pruning likelihoods, profiling
the root mean. Environment is callable with age before present, or an EnvCurve.
EnvExp: rate = sigma2*exp(beta*environment(age)).
EnvLin: rate = sigma2+(beta-sigma2)*environment(age), matching RPANDA source.
Both EnvLin endpoint rates must be positive; scaling environment to [0,1]
is recommended. Invalid candidate rates return success=false, loglik=-Inf.
Requires a bifurcating ultrametric tree. NaN denotes missing traits.
Callable environments are checked at quadrature nodes, not over their entire
domain; users must ensure adequate quadrature resolution for their functions.
For known sharp features, provide `quadrature_breakpoints` as ages before
present to split each branch at those ages. Increasing order alone may miss
narrow pulses. EnvCurve knots are always used in addition to supplied breaks.
EnvLin integrates EnvCurve and SmoothEnvCurve analytically; callable histories
retain numerical quadrature.
"""
function t_env_loglikelihood(tree::CompactTree,trait::AbstractVector{<:Real},
    environment,sigma2::Real,beta::Real;model::Symbol=:EnvExp,
    quadrature_order::Integer=16,quadrature_breakpoints::AbstractVector{<:Real}=Float64[],trait_name=nothing)
    _env_model(model)
    data = _validate_univariate_trait_allow_missing(tree,trait)
    cache = _env_precalc(tree,environment;quadrature_order,quadrature_breakpoints,model)
    variances = zeros(tree.nedges)
    sigma, coefficient = Float64(sigma2),Float64(beta)
    profile = _env_profile(tree,data,cache,variances,sigma,coefficient,model)
    return _env_result(tree,data,cache,variances,profile,sigma,coefficient,model;trait_name)
end

"""
    fit_t_env(tree, trait, environment; model=:EnvExp,
        optimization=:LN_SBPLX, max_iterations=2000, rel_tol=1e-6,
        guess_sigma2=nothing, guess_beta=nothing, quadrature_order=16,
        quadrature_breakpoints=Float64[])

Fit environmental BM. Quadrature values and weights are shared across identical
age intervals and cached once. EnvLin uses per-edge analytical environment
integrals for EnvCurve/SmoothEnvCurve, with no quadrature-node cache.
Sigma2 and the EnvLin endpoint beta are optimized
on log scale; EnvExp beta is unconstrained. The default starting rates are
sample trait variance / tree height, with zero environmental effect.
Smoothing may be performed explicitly with smooth_env before fitting;
no automatic smoothing, measurement error, custom rate function or ASR is included.
See t_env_loglikelihood for the statistical and quadrature conventions.
"""
function fit_t_env(tree::CompactTree,trait::AbstractVector{<:Real},environment;
    model::Symbol=:EnvExp,optimization::Symbol=:LN_SBPLX,
    max_iterations::Integer=2000,rel_tol::Real=1e-6,
    guess_sigma2::Union{Nothing,Real}=nothing,guess_beta::Union{Nothing,Real}=nothing,
    quadrature_order::Integer=16,quadrature_breakpoints::AbstractVector{<:Real}=Float64[],trait_name=nothing)
    _env_model(model)
    max_iterations > 0 && isfinite(rel_tol) && rel_tol > 0 || throw(ArgumentError("Invalid optimization budget or tolerance"))
    data = _validate_univariate_trait_allow_missing(tree,trait)
    observed = filter(!isnan,data)
    length(observed) >= 2 || throw(ArgumentError("Fitting requires at least two observed trait values"))
    cache = _env_precalc(tree,environment;quadrature_order,quadrature_breakpoints,model)
    sigma = guess_sigma2 === nothing ? max(var(observed)/cache.height,1e-8) : Float64(guess_sigma2)
    beta = guess_beta === nothing ? (model === :EnvExp ? 0.0 : sigma) : Float64(guess_beta)
    isfinite(sigma) && sigma > 0 && isfinite(beta) && (model !== :EnvLin || beta > 0) ||
        throw(ArgumentError("Starting rates must be positive and finite; EnvExp beta must be finite"))
    variances = zeros(tree.nedges)
    initial = [log(sigma),model === :EnvExp ? beta : log(beta)]
    unpack(pars) = (exp(pars[1]),model === :EnvExp ? pars[2] : exp(pars[2]))
    objective = pars -> begin
        sigma, beta = unpack(pars)
        profile = _env_profile(tree,data,cache,variances,sigma,beta,model)
        return profile.success ? -profile.loglik : Inf
    end
    isfinite(objective(initial)) || throw(ArgumentError("Starting parameters yield an invalid environmental likelihood"))
    optimized = _continuous_optimize_objective(objective,initial;
        method=optimization,max_iterations,rel_tol=Float64(rel_tol))
    sigma,beta = unpack(_continuous_result_minimizer(optimized))
    profile = _env_profile(tree,data,cache,variances,sigma,beta,model)
    return _env_result(tree,data,cache,variances,profile,sigma,beta,model;
        trait_name,optimized,optimization)
end

function fit_t_env(tree::CompactTree,data::AbstractDataFrame,environment;trait_col=nothing,kwargs...)
    trait,trait_name = _continuous_align_dataframe_trait(tree,data;trait_col)
    return fit_t_env(tree,trait,environment;trait_name,kwargs...)
end
