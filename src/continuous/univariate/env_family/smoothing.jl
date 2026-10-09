"""
    smooth_env(time, values; df, weights=ones(length(time)), scale=false)

Construct a fixed-effective-df cubic smoothing history followed by FMM cubic
interpolation of smoothed knot values, as in RPANDA's fixed-df preprocessing.
Duplicate times use arithmetic mean values and summed weights (pspline's
sm.spline convention). At least six unique ages and 2 < df < n are required.
Weights follow pspline's solver convention (residual correction multiplied
by w), not the conventional inverse-variance weighting of stats::smooth.spline.
No automatic df selection, extrapolation, or smoothing during fitting.
For more than 1000 unique ages, one-time band solving uses BigFloat to
reduce cancellation; the returned curve and subsequent fitting use Float64.
Dense histories agree with the original pspline solver evaluated at higher
precision and the same penalty. Its Float64 results can differ on close knots;
its loose df-search tolerance can also produce a different achieved df.
"""
struct SmoothEnvCurve
    time::Vector{Float64}
    values::Vector{Float64}
    weights::Vector{Float64}
    smoothed::Vector{Float64}
    requested_df::Float64
    effective_df::Float64
    penalty::Float64
    scaled::Bool
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
end

function (env::SmoothEnvCurve)(age::Real)
    first(env.time) <= age <= last(env.time) || throw(ArgumentError("Environmental age is outside the observed time range"))
    i = min(searchsortedlast(env.time,age),length(env.time)-1)
    t = age-env.time[i]
    return env.smoothed[i]+t*(env.b[i]+t*(env.c[i]+t*env.d[i]))
end

_env_knots(env) = Float64[]
_env_knots(env::Union{EnvCurve,SmoothEnvCurve}) = env.time

function _env_smooth_range(env::SmoothEnvCurve, height)
    candidates = Float64[0.0,height]
    append!(candidates,filter(t -> 0<t<height,env.time))
    for i in 1:length(env.time)-1
        env.time[i] >= height && break
        a,b,c = 3env.d[i],2env.c[i],env.b[i]
        roots = if a == 0
            b == 0 ? Float64[] : [-c/b]
        else
            disc = b*b-4a*c
            disc < 0 ? Float64[] : [(-b+sqrt(disc))/(2a),(-b-sqrt(disc))/(2a)]
        end
        for t in roots
            age = env.time[i]+t
            env.time[i] < age < min(height,env.time[i+1]) && push!(candidates,age)
        end
    end
    return extrema(env.(candidates))
end

# LDL and selected inverse of a symmetric positive definite pentadiagonal matrix.
# Only the two inverse bands needed for trace(B^-1 G) are computed.
function _env_band_solve(diag, off1, off2, rhs, g0, g1, g2, lambda)
    n = length(diag)
    T = eltype(diag)
    D, l1, l2 = copy(diag),zeros(T,n),zeros(T,n)
    z = T.(rhs)
    for i in 1:n
        if i > 2
            l2[i] = off2[i-2]/D[i-2]
        end
        if i > 1
            l1[i] = (off1[i-1]-(i>2 ? l2[i]*D[i-2]*l1[i-1] : 0.0))/D[i-1]
        end
        D[i] -= (i>1 ? l1[i]^2*D[i-1] : 0.0)+(i>2 ? l2[i]^2*D[i-2] : 0.0)
        isfinite(D[i]) && D[i]>0 || throw(PosDefException(i))
        z[i] -= (i>1 ? l1[i]*z[i-1] : 0.0)+(i>2 ? l2[i]*z[i-2] : 0.0)
    end
    solution = z ./ D
    s0,s1,s2 = zeros(T,n),zeros(T,max(n-1,0)),zeros(T,max(n-2,0))
    for i in n:-1:1
        if i+2 <= n
            s2[i] = -l1[i+1]*s1[i+1]-l2[i+2]*s0[i+2]
        end
        if i+1 <= n
            s1[i] = -l1[i+1]*s0[i+1]-(i+2<=n ? l2[i+2]*s1[i+1] : 0.0)
        end
        s0[i] = 1/D[i]-(i+1<=n ? l1[i+1]*s1[i] : 0.0)-(i+2<=n ? l2[i+2]*s2[i] : 0.0)
        solution[i] -= (i+1<=n ? l1[i+1]*solution[i+1] : 0.0)+(i+2<=n ? l2[i+2]*solution[i+2] : 0.0)
    end
    removed_df = lambda*(dot(s0,g0)+2dot(s1,g1)+2dot(s2,g2))
    return solution,removed_df
end

function _env_fmm(x,y)
    n = length(x)
    b,c,d = zeros(n),zeros(n),zeros(n)
    d[1] = x[2]-x[1]
    c[2] = (y[2]-y[1])/d[1]
    for i in 2:n-1
        d[i] = x[i+1]-x[i]
        b[i] = 2*(d[i-1]+d[i])
        c[i+1] = (y[i+1]-y[i])/d[i]
        c[i] = c[i+1]-c[i]
    end
    b[1],b[n] = -d[1],-d[n-1]
    c[1] = (c[3]/(x[4]-x[2])-c[2]/(x[3]-x[1]))*d[1]^2/(x[4]-x[1])
    c[n] = -(c[n-1]/(x[n]-x[n-2])-c[n-2]/(x[n-1]-x[n-3]))*d[n-1]^2/(x[n]-x[n-3])
    for i in 2:n
        t = d[i-1]/b[i-1]
        b[i] -= t*d[i-1]
        c[i] -= t*c[i-1]
    end
    c[n] /= b[n]
    for i in n-1:-1:1
        c[i] = (c[i]-d[i]*c[i+1])/b[i]
    end
    b[n] = (y[n]-y[n-1])/d[n-1]+d[n-1]*(c[n-1]+2c[n])
    for i in 1:n-1
        b[i] = (y[i+1]-y[i])/d[i]-d[i]*(c[i+1]+2c[i])
        d[i] = (c[i+1]-c[i])/d[i]
        c[i] *= 3
    end
    c[n] *= 3
    d[n] = d[n-1]
    return b,c,d
end

function smooth_env(time::AbstractVector{<:Real}, values::AbstractVector{<:Real};
    df::Real,weights::AbstractVector{<:Real}=ones(length(time)),scale::Bool=false)
    return _env_smooth(time,values;df,weights,scale)
end

function _env_smooth(time::AbstractVector{<:Real}, values::AbstractVector{<:Real};
    df::Real,weights::AbstractVector{<:Real}=ones(length(time)),scale::Bool=false,
    penalty=nothing)
    isnothing(penalty) || (isfinite(penalty) && penalty > 0) ||
        throw(ArgumentError("Smoothing penalty must be finite and positive"))
    length(time)==length(values)==length(weights) || throw(ArgumentError("Environmental arrays must have matching lengths"))
    all(isfinite,time) && all(isfinite,values) && all(w -> isfinite(w)&&w>0,weights) &&
        all(>=(0),time) || throw(ArgumentError("Finite observations, nonnegative ages and positive weights are required"))
    order = sortperm(time)
    x,y,w = Float64[],Float64[],Float64[]
    counts = Int[]
    for i in order
        if isempty(x) || time[i] != x[end]
            push!(x,time[i]);push!(y,values[i]);push!(w,weights[i]);push!(counts,1)
        else
            y[end] += values[i];w[end] += weights[i];counts[end] += 1
        end
    end
    y ./= counts
    n = length(x)
    n >= 6 && isfinite(df) && 2 < df < n || throw(ArgumentError("Require at least six unique ages and 2 < df < unique age count"))
    # Normalize time for conditioning; report penalty in original time units.
    span = x[end]-x[1]
    # Dense histories with very close knots can lose the small roughness term
    # in Float64. Higher precision is restricted to one-time preprocessing.
    h = n > 1000 ? diff(BigFloat.(x)) ./ BigFloat(span) : diff(x) ./ span
    m = n-2
    a = 1 ./ h[1:end-1]
    c = 1 ./ h[2:end]
    b = .-a.-c
    r0 = (h[1:end-1]+h[2:end])./3
    r1 = h[2:end-1]./6
    g0 = a.^2 .* w[1:end-2]+b.^2 .* w[2:end-1]+c.^2 .* w[3:end]
    g1 = b[1:end-1].*a[2:end].*w[2:end-2]+c[1:end-1].*b[2:end].*w[3:end-1]
    g2 = c[1:end-2].*a[3:end].*w[3:end-2]
    rhs = a.*y[1:end-2]+b.*y[2:end-1]+c.*y[3:end]
    evaluate(lambda) = _env_band_solve(r0+lambda*g0,r1+lambda*g1,lambda*g2,rhs,g0,g1,g2,lambda)
    low,high = 0.0,median(r0./g0)
    for _ in 1:(isnothing(penalty) ? 100 : 0)
        _,removed = evaluate(high)
        n-removed <= df && break
        high *= 2
    end
    solution,lambda,effective = zeros(m),high,NaN
    for _ in 1:(isnothing(penalty) ? 100 : 0)
        lambda = (low+high)/2
        solution,removed = evaluate(lambda)
        effective = n-removed
        abs(effective-df) <= 1e-9 && break
        effective > df ? (low=lambda) : (high=lambda)
    end
    if isnothing(penalty)
        abs(effective-df) <= 1e-7 || throw(ArgumentError("Could not resolve requested smoothing df numerically"))
    else
        lambda = eltype(h)(penalty)/eltype(h)(span)^3
        solution,removed = evaluate(lambda)
        effective = n-removed
    end
    fitted = eltype(h).(y)
    for i in 1:m
        fitted[i] -= lambda*a[i]*solution[i]*w[i]
        fitted[i+1] -= lambda*b[i]*solution[i]*w[i+1]
        fitted[i+2] -= lambda*c[i]*solution[i]*w[i+2]
    end
    if scale
        lo,hi = extrema(fitted)
        hi>lo || throw(ArgumentError("Cannot scale a constant smoothed curve"))
        fitted = (fitted .- lo)./(hi-lo)
    end
    fitted = Float64.(fitted)
    coef = _env_fmm(x,fitted)
    return SmoothEnvCurve(x,y,w,fitted,Float64(df),Float64(effective),Float64(lambda*span^3),scale,coef...)
end

@doc (@doc SmoothEnvCurve) smooth_env
