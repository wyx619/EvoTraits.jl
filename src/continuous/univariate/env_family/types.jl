"""
    EnvCurve(time, values; scale=false)

Piecewise-linear environmental history. Time is age before present in the
same units as tree branch lengths. Times must be finite, nonnegative and
distinct; inputs are sorted together. Extrapolation is not permitted.
With `scale=true`, values are rescaled to [0,1]; constant curves cannot be scaled.
No smoothing is performed.
"""
struct EnvCurve
    time::Vector{Float64}
    values::Vector{Float64}
    function EnvCurve(time::AbstractVector{<:Real}, values::AbstractVector{<:Real}; scale::Bool=false)
        length(time) == length(values) >= 2 || throw(ArgumentError("Environmental history requires at least two paired observations"))
        t, v = Float64.(time), Float64.(values)
        all(isfinite,t) && all(isfinite,v) && all(>=(0),t) ||
            throw(ArgumentError("Environmental times and values must be finite; times must be nonnegative"))
        order = sortperm(t)
        t, v = t[order], v[order]
        all(>(0),diff(t)) || throw(ArgumentError("Duplicate environmental times must be resolved before interpolation"))
        if scale
            lo, hi = extrema(v)
            hi > lo || throw(ArgumentError("Cannot scale a constant environmental curve"))
            v = (v .- lo) ./ (hi-lo)
        end
        new(t,v)
    end
end

function (env::EnvCurve)(age::Real)
    first(env.time) <= age <= last(env.time) || throw(ArgumentError("Environmental age is outside the observed time range"))
    i = min(searchsortedlast(env.time,age), length(env.time)-1)
    fraction = (age-env.time[i])/(env.time[i+1]-env.time[i])
    return (1-fraction)*env.values[i] + fraction*env.values[i+1]
end

"""
    EnvFitResult

Environmental BM result. `sigma2` is the rate at environment=0. For EnvExp,
`beta` is the exponential coefficient; for EnvLin, it is the rate at
environment=1 (not an additive slope). The GLS root mean is profiled and
counted in `nparams=3`. No measurement error or ASR interface is provided.
`success` denotes a valid likelihood, not optimizer convergence.
"""
Base.@kwdef struct EnvFitResult <: AbstractContinuousFitResult
    model::Symbol
    success::Bool = false
    loglik::Float64 = -Inf
    aic::Float64 = Inf
    aicc::Float64 = Inf
    nparams::Int = 3
    nobs::Int
    trait_name::String = "trait"
    sigma2::Float64
    beta::Float64
    theta::Float64 = NaN
    root_state::Float64 = NaN
    root_var::Float64 = NaN
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
    optimization::Symbol = :none
    quadrature_order::Int
    quadrature_breakpoints::Vector{Float64} = Float64[]
    height::Float64
    environment::Any
    edge_variances::Vector{Float64}
end

function Base.show(io::IO, result::EnvFitResult)
    print(io,result.model," environmental BM (logLik=",result.loglik,
        ", AIC=",result.aic,", converged=",result.converged,
        ", sigma2=",result.sigma2,", beta=",result.beta,
        ", root=",result.root_state,")")
end
