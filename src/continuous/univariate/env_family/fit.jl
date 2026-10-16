function _env_model(model)
    model in (:EnvExp,:EnvLin) || throw(ArgumentError("model must be :EnvExp or :EnvLin"))
    return model
end

function _env_standard_errors(tree, error)
    error === nothing && return Float64[], Int[]
    values = if ismissing(error) || (error isa Real && isnan(error))
        zeros(tree.ntips)
    elseif error isa AbstractVector
        length(error) == tree.ntips || throw(ArgumentError("Expected $(tree.ntips) tip standard errors"))
        error
    elseif error isa AbstractDict
        named = Dict{String,Any}()
        for (taxon, value) in error
            label = string(taxon)
            haskey(named,label) && throw(ArgumentError("Duplicate standard-error taxon: $label"))
            named[label] = value
        end
        all(label -> haskey(named,label),tree.tip_labels) ||
            throw(ArgumentError("Standard errors must include every tree tip"))
        [named[label] for label in tree.tip_labels]
    else
        throw(ArgumentError("error must be nothing, missing, NaN, a vector in tree-tip order, or a taxon-keyed dictionary"))
    end
    all(x -> x isa Real && isfinite(x) && x >= 0,values) ||
        throw(ArgumentError("Tip standard errors must be finite and nonnegative; use error=missing to estimate nuisance without known errors"))
    standard_errors = Float64.(values)
    all(x -> isfinite(x*x),standard_errors) || throw(ArgumentError("Tip standard-error variances overflow Float64"))
    tip_index = zeros(Int,tree.nnodes)
    tip_index[tree.tip_ids] .= eachindex(tree.tip_ids)
    edges = Vector{Int}(undef,tree.ntips)
    for edge in 1:tree.nedges
        index = tip_index[tree.child_of_edge[edge]]
        index > 0 && (edges[index] = edge)
    end
    return standard_errors,edges
end

function _env_profile(tree, data, cache, variances, sigma2, beta, model;
    tip_edges=(),error_variances=(),nuisance_variance=0.0)
    _env_edge_variances!(variances,cache,sigma2,beta,model) ||
        return (success=false,loglik=-Inf,root_state=NaN,root_var=NaN)
    isfinite(nuisance_variance) && nuisance_variance >= 0 ||
        return (success=false,loglik=-Inf,root_state=NaN,root_var=NaN)
    for i in eachindex(tip_edges)
        edge = tip_edges[i]
        variance = variances[edge]+error_variances[i]+nuisance_variance
        isfinite(variance) || return (success=false,loglik=-Inf,root_state=NaN,root_var=NaN)
        variances[edge] = variance
    end
    return _bm_pruning_profile_from_edge_variances(tree,data,variances)
end

function _env_result(tree,data,cache,variances,profile,sigma2,beta,model;
    trait_name=nothing,optimized=nothing,optimization=:none,
    standard_errors=Float64[],nuisance=0.0)
    nobs = count(!isnan,data)
    nparams = isempty(standard_errors) ? 3 : 4
    return EnvFitResult(;model,success=profile.success,loglik=profile.loglik,
        aic=profile.success ? compute_aic(profile.loglik,nparams) : Inf,
        aicc=profile.success ? compute_aicc(profile.loglik,nparams,nobs) : Inf,
        nobs,nparams,trait_name=_continuous_checked_trait_name(trait_name),sigma2,beta,
        nuisance,nuisance_variance=nuisance*nuisance,standard_errors=copy(standard_errors),
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
        model=:EnvExp, error=nothing, nuisance=0.0,
        quadrature_order=16, trait_name=nothing)

Evaluate environmental BM using existing tree-pruning likelihoods, profiling
the root mean. Environment is callable with age before present, or an EnvCurve.
EnvExp: rate = sigma2*exp(beta*environment(age)).
EnvLin: rate = sigma2+(beta-sigma2)*environment(age), matching RPANDA source.
Both EnvLin endpoint rates must be positive; scaling environment to [0,1]
is recommended. Invalid candidate rates return success=false, loglik=-Inf.
Requires a bifurcating ultrametric tree. NaN denotes missing traits.
`error` supplies known tip standard errors as a vector in tree-tip order or a
taxon-keyed dictionary. `missing` or scalar `NaN` denotes zero known errors.
`nuisance` is a fixed, nonnegative common terminal standard deviation.
Each terminal branch receives variance `error[i]^2 + nuisance^2`; internal
branches are unchanged. With error enabled (or positive nuisance), the model
counts four parameters, including nuisance, even when evaluating its zero
boundary. Otherwise it retains three parameters. No error is estimated by
this fixed-parameter interface. `edge_variances` includes terminal errors.
Callable environments are checked at quadrature nodes, not over their entire
domain; users must ensure adequate quadrature resolution for their functions.
For known sharp features, provide `quadrature_breakpoints` as ages before
present to split each branch at those ages. Increasing order alone may miss
narrow pulses. EnvCurve knots are always used in addition to supplied breaks.
EnvLin integrates EnvCurve and SmoothEnvCurve analytically; callable histories
retain numerical quadrature.
"""
function t_env_loglikelihood(tree::CompactTree,trait::AbstractVector{<:Real},
    environment,sigma2::Real,beta::Real;model::Symbol=:EnvExp,error=nothing,nuisance::Real=0.0,
    quadrature_order::Integer=16,quadrature_breakpoints::AbstractVector{<:Real}=Float64[],trait_name=nothing)
    _env_model(model)
    nu = Float64(nuisance)
    isfinite(nu) && nu >= 0 && isfinite(nu*nu) || throw(ArgumentError("nuisance must be a finite, nonnegative standard deviation"))
    data = _validate_univariate_trait_allow_missing(tree,trait)
    standard_errors,tip_edges = _env_standard_errors(tree,error === nothing && nu > 0 ? missing : error)
    error_variances = standard_errors.^2
    cache = _env_precalc(tree,environment;quadrature_order,quadrature_breakpoints,model)
    variances = zeros(tree.nedges)
    sigma, coefficient = Float64(sigma2),Float64(beta)
    profile = _env_profile(tree,data,cache,variances,sigma,coefficient,model;
        tip_edges,error_variances,nuisance_variance=nu*nu)
    return _env_result(tree,data,cache,variances,profile,sigma,coefficient,model;
        trait_name,standard_errors,nuisance=nu)
end

"""
    fit_t_env(tree, trait, environment; model=:EnvExp, error=nothing,
        optimization=:LN_SBPLX, max_iterations=2000, rel_tol=1e-6,
        guess_sigma2=nothing, guess_beta=nothing, quadrature_order=16,
        quadrature_breakpoints=Float64[])

Fit environmental BM. Quadrature values and weights are shared across identical
age intervals and cached once. EnvLin uses per-edge analytical environment
integrals for EnvCurve/SmoothEnvCurve, with no quadrature-node cache.
Sigma2 and the EnvLin endpoint beta are optimized
on log scale; EnvExp beta is unconstrained. The default starting rates are
sample trait variance / tree height, with zero environmental effect.
`error=nothing` retains the model without measurement error. As in RPANDA,
`error=missing` (or scalar `NaN`) estimates a common nuisance standard deviation
without known errors; a finite nonnegative vector in tree-tip order or a
taxon-keyed dictionary supplies known standard errors and also estimates
nuisance. Each tip receives variance `error[i]^2 + nuisance^2` in addition to
the environmental process. Nuisance is optimized as a squared standard
deviation, allowing an exact zero boundary, and adds one free parameter.
For DataFrame input, `error=:se` selects a standard-error column and aligns
it by taxon; specify `trait_col` when multiple numeric columns are present.
Smoothing may be performed explicitly with smooth_env before fitting;
no automatic smoothing, custom rate function or ASR is included.
See t_env_loglikelihood for the statistical and quadrature conventions.
"""
function fit_t_env(tree::CompactTree,trait::AbstractVector{<:Real},environment;
    model::Symbol=:EnvExp,error=nothing,optimization::Symbol=:LN_SBPLX,
    max_iterations::Integer=2000,rel_tol::Real=1e-6,
    guess_sigma2::Union{Nothing,Real}=nothing,guess_beta::Union{Nothing,Real}=nothing,
    quadrature_order::Integer=16,quadrature_breakpoints::AbstractVector{<:Real}=Float64[],trait_name=nothing)
    _env_model(model)
    max_iterations > 0 && isfinite(rel_tol) && rel_tol > 0 || throw(ArgumentError("Invalid optimization budget or tolerance"))
    data = _validate_univariate_trait_allow_missing(tree,trait)
    standard_errors,tip_edges = _env_standard_errors(tree,error)
    error_variances = standard_errors.^2
    estimate_nuisance = !isempty(standard_errors)
    observed = filter(!isnan,data)
    length(observed) >= 2 || throw(ArgumentError("Fitting requires at least two observed trait values"))
    cache = _env_precalc(tree,environment;quadrature_order,quadrature_breakpoints,model)
    sigma = guess_sigma2 === nothing ? max(var(observed)/cache.height,1e-8) : Float64(guess_sigma2)
    beta = guess_beta === nothing ? (model === :EnvExp ? 0.0 : sigma) : Float64(guess_beta)
    isfinite(sigma) && sigma > 0 && isfinite(beta) && (model !== :EnvLin || beta > 0) ||
        throw(ArgumentError("Starting rates must be positive and finite; EnvExp beta must be finite"))
    variances = zeros(tree.nedges)
    initial = [log(sigma),model === :EnvExp ? beta : log(beta)]
    estimate_nuisance && push!(initial,0.1)
    unpack(pars) = (exp(pars[1]),model === :EnvExp ? pars[2] : exp(pars[2]),
        estimate_nuisance ? abs(pars[3]) : 0.0)
    objective = pars -> begin
        sigma, beta, nuisance = unpack(pars)
        profile = _env_profile(tree,data,cache,variances,sigma,beta,model;
            tip_edges,error_variances,nuisance_variance=nuisance*nuisance)
        return profile.success ? -profile.loglik : Inf
    end
    isfinite(objective(initial)) || throw(ArgumentError("Starting parameters yield an invalid environmental likelihood"))
    optimized = _continuous_optimize_objective(objective,initial;
        method=optimization,max_iterations,rel_tol=Float64(rel_tol))
    sigma,beta,nuisance = unpack(_continuous_result_minimizer(optimized))
    profile = _env_profile(tree,data,cache,variances,sigma,beta,model;
        tip_edges,error_variances,nuisance_variance=nuisance*nuisance)
    return _env_result(tree,data,cache,variances,profile,sigma,beta,model;
        trait_name,optimized,optimization,standard_errors,nuisance)
end

function fit_t_env(tree::CompactTree,data::AbstractDataFrame,environment;
    trait_col=nothing,error=nothing,kwargs...)
    trait,trait_name = _continuous_align_dataframe_trait(tree,data;trait_col)
    if error isa Union{Symbol,AbstractString}
        column = Symbol(error)
        column in propertynames(data) || throw(ArgumentError("Standard-error column `$column` was not found"))
        rows = _aligned_dataframe_rows(tree,data,first(propertynames(data)))
        error = data[rows,column]
    end
    return fit_t_env(tree,trait,environment;trait_name,error,kwargs...)
end
