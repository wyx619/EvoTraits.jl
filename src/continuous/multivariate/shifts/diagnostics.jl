# Central differences in locally scaled optimization coordinates, not matrix entries.
function _shift_curvature(objective, x, step)
    n=length(x)
    scales=max.(abs.(x),1.0)
    evaluate(u)=objective(x+scales.*u)
    origin=zeros(n)
    f0=evaluate(origin)
    H=zeros(n,n)
    gradient=zeros(n)
    for i in 1:n
        u=zeros(n);u[i]=step
        fp,fm=evaluate(u),evaluate(-u)
        gradient[i]=(fp-fm)/(2step)
        H[i,i]=(fp-2*f0+fm)/step^2
        for j in 1:i-1
            v=zeros(n);v[j]=step
            H[i,j]=H[j,i]=(evaluate(u+v)-evaluate(u-v)-evaluate(-u+v)+evaluate(-u-v))/(4step^2)
        end
    end
    return (hessian=H,gradient=gradient,scales=scales)
end

"""
    diagnose_shift(tree, traits, result; step=eps(Float64)^(1/4), tolerance=0.05)

Evaluate local curvature of the profiled negative log likelihood without refitting.
Accepts a vector or matrix, in tree tip order. Uses two central-difference step
sizes in scaled optimization coordinates. Returns gradient, Hessian, eigenvalues,
step agreement and status; does not change result.success or result.converged.
Positive curvature is not proof of global optimality, causality or parameter
identifiability. No standard errors or confidence intervals are inferred.
Cost is quadratic in the number of covariance/process coordinates. Theta is
profiled, and is not included in the returned Hessian. The caller must provide
the same tree and data used in fitting; the result does not store a data snapshot.
"""
function diagnose_shift(tree::CompactTree,traits,result::ShiftFitResult;
    step::Real=eps(Float64)^(1/4),tolerance::Real=0.05)
    isfinite(step) && step>0 || throw(ArgumentError("step must be finite and positive"))
    isfinite(tolerance) && 0<tolerance<1 || throw(ArgumentError("tolerance must lie between zero and one"))
    Y=traits isa AbstractVector ? reshape(_validate_univariate_trait_allow_missing(tree,traits),:,1) :
        _validate_multivariate_trait(tree,traits)
    size(Y,2)==result.ntraits || throw(ArgumentError("Trait dimension does not match result"))
    source_age=result.age/result.time_scale
    prepared=_shift_scaled_input(tree,result.ntraits;scale_height=result.scale_height,
        age=source_age,mapped_edge=nothing,precalc=nothing)
    mapping=_shift_mapping(prepared.tree,result.ntraits;prepared.input...)
    isapprox(mapping.height,result.height;rtol=1e-10) &&
        isapprox(mapping.mapped_edge,result.mapped_edge;rtol=1e-10,atol=0.0) ||
        throw(ArgumentError("Tree time mapping does not match result"))
    result.success || return (status=:invalid_result,)
    sr=result.model===:SR
    spec=sr ? nothing : _shift_process_spec(result.model)
    block=_shift_sigma_nparams(result.ntraits,result.Sigma_decomp)
    x=vcat(result.A===nothing ? Float64[] : _shift_pack_A(result.A,result.A_decomp),
        _shift_pack_sigma(result.sigma,result.Sigma_decomp),
        sr || spec.independent ? _shift_pack_sigma(result.sig,result.Sigma_decomp) : Float64[],
        result.input_beta===nothing ? Float64[] : [result.input_beta])
    calls=Ref(0)
    objective=function(pars)
        calls[]+=1
        try
            r=if sr
                S=[_shift_unpack_sigma(@view(pars[(k-1)*block+1:k*block]),result.ntraits,result.Sigma_decomp) for k in 1:2]
                _mvshift_sr_loglikelihood(prepared.tree,Y,S;mapped_edge=mapping.mapped_edge,
                    Sigma_decomp=result.Sigma_decomp)
            else
                A,S,S2,beta=_shift_process_unpack(pars,result.ntraits,result.A_decomp,result.Sigma_decomp,spec)
                extra=merge(spec.independent ? (sig=S2,) : NamedTuple(),
                    spec.eb_period>0 ? (beta=beta,beta_bounds=result.beta_bounds) : NamedTuple())
                _mvshift_process_loglikelihood(prepared.tree,Y,A,S;model=result.model,
                    mapped_edge=mapping.mapped_edge,A_decomp=result.A_decomp===nothing ? :cholesky : result.A_decomp,
                    Sigma_decomp=result.Sigma_decomp,extra...)
            end
            return r.success ? -r.loglik : Inf
        catch err
            err isa ArgumentError || err isa PosDefException || err isa OverflowError || rethrow()
            return Inf
        end
    end
    base=objective(x)
    isapprox(base,-result.loglik;rtol=1e-8,atol=1e-8) ||
        throw(ArgumentError("Result likelihood cannot be reproduced with supplied tree and data"))
    if result.input_beta!==nothing && result.beta_bounds!==nothing
        beta=result.input_beta
        low,high=result.beta_bounds
        if beta<low || beta>high
            return (status=:beta_plateau,f_calls=calls[],)
        elseif min(beta-low,high-beta)<=2step*max(abs(beta),1.0)
            return (status=:beta_boundary,f_calls=calls[],)
        end
    end
    coarse=_shift_curvature(objective,x,Float64(step))
    fine=_shift_curvature(objective,x,Float64(step)/2)
    H=fine.hessian
    finite=all(isfinite,H) && all(isfinite,fine.gradient) && all(isfinite,coarse.hessian)
    agreement=finite ? norm(H-coarse.hessian)/max(norm(H),norm(coarse.hessian),eps(Float64)) : Inf
    values=finite ? eigvals(Symmetric(H)) : Float64[]
    cutoff=finite ? max(maximum(abs,values)*1e-6,1e-8) : Inf
    status=!finite ? :invalid_neighborhood : agreement>tolerance ? :unstable_differences :
        minimum(values)<-cutoff ? :negative_curvature :
        minimum(values)<=cutoff ? :weak_curvature :
        maximum(abs,fine.gradient)>1e-4*max(abs(base),1.0) ? :nonstationary : :positive_curvature
    return (status=status,hessian=H,eigenvalues=values,gradient=fine.gradient,
        parameter_scales=fine.scales,parameters=x,step=Float64(step),
        relative_step_gap=agreement,profile_theta=true,f_calls=calls[])
end
