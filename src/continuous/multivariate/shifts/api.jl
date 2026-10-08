"""
    ShiftFitResult

Result shared by scalar and multivariate time-shift models. `sigma` and `sig`
are matrices (1x1 for scalar traits); for SR they are early and late diffusion.
For OU models `sigma` is OU diffusion, otherwise BM, and `sig` is the other
component. `theta` is a constant mean vector, not ancestral reconstruction.
Fixed evaluations have termination=:FIXED and converged=false.
With scale_height=true, time metadata and rate parameters use normalized units;
source_height records the original tree height and time_scale=1/source_height.
"""
Base.@kwdef struct ShiftFitResult
    model::Symbol
    input_model::Symbol
    backend::Symbol = :tree_pruning
    success::Bool
    converged::Bool
    loglik::Float64
    aic::Float64
    aicc::Float64
    nparams::Int
    nobs::Int
    ntraits::Int
    trait_names::Vector{String}
    A::Union{Nothing,Matrix{Float64}}
    A_decomp::Union{Nothing,Symbol}
    sigma::Matrix{Float64}
    sig::Matrix{Float64}
    Sigma_decomp::Symbol
    theta::Vector{Float64}
    beta::Union{Nothing,Float64} = nothing
    input_beta::Union{Nothing,Float64} = nothing
    beta_bounds::Union{Nothing,Tuple{Float64,Float64}} = nothing
    root_mean_mode::Symbol = :constant
    root_cov_mode::Symbol = :fixed
    age::Float64
    boundary::Float64
    height::Float64
    source_height::Float64
    time_scale::Float64 = 1.0
    scale_height::Bool = false
    mapped_edge::Matrix{Float64}
    optimization::Union{Nothing,Symbol} = nothing
    termination::Symbol = :FIXED
    iterations::Int = 0
    f_calls::Int = 0
    max_iterations::Union{Nothing,Int} = nothing
    rel_tol::Union{Nothing,Float64} = nothing
    initial_loglik::Union{Nothing,Float64} = nothing
    used_best_candidate::Union{Nothing,Bool} = nothing
end

Base.summary(r::ShiftFitResult) = "$(r.model) time-shift result (logLik=$(r.loglik), AIC=$(r.aic), converged=$(r.converged))"
Base.show(io::IO,r::ShiftFitResult) = print(io,summary(r))
function Base.show(io::IO,::MIME"text/plain",r::ShiftFitResult)
    println(io,summary(r))
    println(io,"  traits: ",join(r.trait_names,", "),"; age: ",r.age,"; height: ",r.height)
    r.scale_height && println(io,"  normalized time; source height: ",r.source_height,"; time scale: ",r.time_scale)
    println(io,"  success: ",r.success,"; termination: ",r.termination,"; f_calls: ",r.f_calls)
    print(io,"  theta: ",r.theta)
end

function _shift_public_result(r; fitted=false, optimization=:L_BFGS,
    max_iterations=1000, rel_tol=1e-6, source_height=r.height, time_scale=1.0,
    scale_height=false)
    sr = r.model === :SR
    f = sr ? r.fit : r
    p = sr ? f.ntraits : r.ntraits
    return ShiftFitResult(
        model=r.model,input_model=sr ? :SR : r.input_model,
        success=r.success,converged=fitted && r.converged,
        loglik=r.loglik,aic=r.aic,aicc=r.aicc,nparams=r.nparams,nobs=r.nobs,
        ntraits=p,trait_names=copy(f.trait_names),
        A=sr ? nothing : r.A,A_decomp=sr ? nothing : r.A_decomp,
        sigma=sr ? copy(f.sigma[:,:,1]) : copy(r.sigma),
        sig=sr ? copy(f.sigma[:,:,2]) : copy(r.sig),Sigma_decomp=r.Sigma_decomp,
        theta=vec(copy(f.theta)),beta=sr ? nothing : r.beta,
        input_beta=sr ? nothing : r.input_beta,
        beta_bounds=sr || r.beta_bounds===nothing ? nothing : Float64.(r.beta_bounds),
        age=r.age,boundary=r.boundary,height=r.height,mapped_edge=copy(r.mapped_edge),
        source_height=source_height,time_scale=time_scale,scale_height=scale_height,
        optimization=fitted ? (sr ? optimization : r.optimization) : nothing,
        termination=!fitted ? :FIXED : sr ? (r.converged ? :CONVERGED : :NOT_CONVERGED) : Symbol(r.termination),
        iterations=fitted ? f.iterations : 0,f_calls=fitted ? f.f_calls : 0,
        max_iterations=fitted ? Int(max_iterations) : nothing,
        rel_tol=fitted ? Float64(rel_tol) : nothing,
        initial_loglik=fitted && !sr ? r.initial_loglik : nothing,
        used_best_candidate=fitted && !sr ? r.used_best_candidate : nothing)
end

function _shift_scaled_input(tree, p; scale_height, age, mapped_edge, precalc)
    if !scale_height
        return (tree=tree, input=(age=age,mapped_edge=mapped_edge,precalc=precalc),
            source_height=maximum(@view tree.dist_from_root[tree.tip_ids]),time_scale=1.0)
    end
    mapping=_shift_mapping(tree,p;age=age,mapped_edge=mapped_edge,precalc=precalc)
    height=mapping.height
    # Only time arrays change. Do not mutate the caller's tree or mapping cache.
    scaled=CompactTree((name===:edge_length ? tree.edge_length/height :
        name===:dist_from_root ? tree.dist_from_root/height : getfield(tree,name)
        for name in fieldnames(CompactTree))...)
    return (tree=scaled,input=(mapped_edge=mapping.mapped_edge/height,),
        source_height=height,time_scale=1/height)
end

"""
    shift_precalc(tree; age=nothing, mapped_edge=nothing, ntraits=1)

Prepare parameter-independent mapping for an ultrametric binary tree.
Specify age before present or two mapped_edge columns (early, late).
Pass only precalc to subsequent evaluations; cached tree changes are rejected.
"""
shift_precalc(tree::CompactTree;kwargs...) = _shift_precalc(tree;kwargs...)

"""
    mvshift_loglikelihood(tree, Y, sigma; model=:SR, A=nothing, kwargs...)

Evaluate a time-shift model, profiling constant theta unless supplied.
Y rows follow tree.tip_ids; NaN denotes missing observations. Supply exactly
one of age (before present), mapped_edge (early/late) or precalc.
SR takes two matrices in sigma; other models take one matrix and independent
models require sig. OU models require A, EB models require beta.
Supports SR, OUBM/OUBMi/BMOU/BMOUi (ER/RR/EC/RC aliases) and
EBOU/EBOUi/OUEB/OUEBi/EBBM/EBBMi/BMEB/BMEBi. No ASR or observation error.
With scale_height=true, input age/mapping/cache uses original tree units;
the tree and mapping are divided by source height. Parameters are interpreted
in normalized time units, not automatically converted. Results record
source_height and time_scale; returned age/height/mapping use normalized units.
"""
function mvshift_loglikelihood(tree::CompactTree,Y::AbstractMatrix{<:Real},sigma;
    model::Symbol=:SR,A=nothing,scale_height::Bool=false,
    age=nothing,mapped_edge=nothing,precalc=nothing,kwargs...)
    prepared=_shift_scaled_input(tree,size(Y,2);scale_height=scale_height,
        age=age,mapped_edge=mapped_edge,precalc=precalc)
    r = if model===:SR
        A===nothing || throw(ArgumentError("SR has no OU component; omit A"))
        _mvshift_sr_loglikelihood(prepared.tree,Y,sigma;prepared.input...,kwargs...)
    else
        _mvshift_process_loglikelihood(prepared.tree,Y,A,sigma;model=model,prepared.input...,kwargs...)
    end
    return _shift_public_result(r;source_height=prepared.source_height,
        time_scale=prepared.time_scale,scale_height=scale_height)
end

"""
    fit_mvshift(tree, Y; model=:SR, kwargs...)

Fit a time-shift model with profiled constant theta. See mvshift_loglikelihood
for model and time inputs. Initial matrices: guess_sigma, guess_sig and guess_A;
SR guess_sigma is a pair. EB accepts guess_beta and beta_bounds=(-2,2), with
out-of-range beta mapped to zero for reference compatibility.
Default optimization is L_BFGS for SR, LN_SBPLX otherwise. A finite result need
not be converged; NLopt max_iterations limits function evaluations. Curvature
diagnostics require an explicit diagnose_shift call. No global-optimum guarantee
or ASR is provided. scale_height follows the fixed
entry's normalized-time parameter convention, including guesses and beta_bounds.
"""
function fit_mvshift(tree::CompactTree,Y::AbstractMatrix{<:Real};model::Symbol=:SR,
    optimization::Symbol=model===:SR ? :L_BFGS : :LN_SBPLX,
    max_iterations::Integer=1000,rel_tol::Float64=1e-6,scale_height::Bool=false,
    age=nothing,mapped_edge=nothing,precalc=nothing,kwargs...)
    prepared=_shift_scaled_input(tree,size(Y,2);scale_height=scale_height,
        age=age,mapped_edge=mapped_edge,precalc=precalc)
    r = model===:SR ? _fit_mvshift_sr(prepared.tree,Y;optimization=optimization,
        max_iterations=max_iterations,rel_tol=rel_tol,prepared.input...,kwargs...) :
        _fit_mvshift_process(prepared.tree,Y;model=model,optimization=optimization,
            max_iterations=max_iterations,rel_tol=rel_tol,prepared.input...,kwargs...)
    return _shift_public_result(r;fitted=true,optimization=optimization,
        max_iterations=max_iterations,rel_tol=rel_tol,source_height=prepared.source_height,
        time_scale=prepared.time_scale,scale_height=scale_height)
end

"""
    shift_loglikelihood(tree, y, sigma2; model=:SR, alpha=nothing, sig=nothing, theta=nothing, kwargs...)

Scalar counterpart of mvshift_loglikelihood. SR sigma2 is [early, late]; other
models use a scalar variance. Returns ShiftFitResult with 1x1 parameter matrices.
"""
function shift_loglikelihood(tree::CompactTree,y::AbstractVector{<:Real},sigma2;
    model::Symbol=:SR,alpha=nothing,sig=nothing,theta=nothing,trait_name=nothing,kwargs...)
    data=_validate_univariate_trait_allow_missing(tree,y)
    matrix(x)=reshape([Float64(x)],1,1)
    S=model===:SR ? [matrix(x) for x in sigma2] : matrix(sigma2)
    extra=sig===nothing ? NamedTuple() : (sig=matrix(sig),)
    return mvshift_loglikelihood(tree,reshape(data,:,1),S;model=model,
        A=alpha===nothing ? nothing : matrix(alpha),
        theta=theta===nothing ? nothing : [Float64(theta)],
        trait_names=[_continuous_checked_trait_name(trait_name)],extra...,kwargs...)
end

"""
    fit_shift(tree, y; model=:SR, guess_alpha=nothing, guess_sigma2=nothing, guess_sig=nothing, kwargs...)

Scalar counterpart of fit_mvshift. SR guess_sigma2 is a pair; other models use
a scalar. Results use the same matrix/vector contract as multivariate fits.
"""
function fit_shift(tree::CompactTree,y::AbstractVector{<:Real};model::Symbol=:SR,
    guess_alpha=nothing,guess_sigma2=nothing,guess_sig=nothing,trait_name=nothing,kwargs...)
    data=_validate_univariate_trait_allow_missing(tree,y)
    matrix(x)=reshape([Float64(x)],1,1)
    extra=NamedTuple()
    if guess_alpha!==nothing
        model===:SR && throw(ArgumentError("SR has no OU component; omit guess_alpha"))
        extra=merge(extra,(guess_A=matrix(guess_alpha),))
    end
    guess_sig===nothing || (extra=merge(extra,(guess_sig=matrix(guess_sig),)))
    S=guess_sigma2===nothing ? nothing : model===:SR ? [matrix(x) for x in guess_sigma2] : matrix(guess_sigma2)
    return fit_mvshift(tree,reshape(data,:,1);model=model,guess_sigma=S,
        trait_names=[_continuous_checked_trait_name(trait_name)],extra...,kwargs...)
end
