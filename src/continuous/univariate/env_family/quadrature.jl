struct EnvPrecalc{F}
    environment::F
    height::Float64
    order::Int
    breakpoints::Vector{Float64}
    offsets::Vector{Int}
    interval_ids::Vector{Int}
    values::Vector{Float64}
    weights::Vector{Float64}
    interval_variances::Vector{Float64}
    edge_durations::Vector{Float64}
    edge_environment_integrals::Vector{Float64}
    analytic_linear::Bool
    minimum_environment::Float64
    maximum_environment::Float64
end

_env_interval_integral(env::EnvCurve,lo,hi) = (hi-lo)*(env(lo)+env(hi))/2

function _env_interval_integral(env::SmoothEnvCurve,lo,hi)
    i = min(searchsortedlast(env.time,lo),length(env.time)-1)
    t,dt = lo-env.time[i],hi-lo
    # Shift the cubic to lo; do not subtract nearly equal primitive values.
    value = env.smoothed[i]+t*(env.b[i]+t*(env.c[i]+t*env.d[i]))
    slope = env.b[i]+t*(2env.c[i]+3t*env.d[i])
    curvature = env.c[i]+3t*env.d[i]
    return dt*(value+dt*(slope/2+dt*(curvature/3+dt*env.d[i]/4)))
end

function _env_precalc(tree::CompactTree,environment;quadrature_order::Integer=16,
    quadrature_breakpoints::AbstractVector{<:Real}=Float64[],model::Symbol=:EnvExp)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    all(x -> isfinite(x) && x>=0,tree.edge_length) || throw(ArgumentError("Branch lengths must be finite and nonnegative"))
    height = maximum(tree.dist_from_root[tree.tip_ids])
    height>0 || throw(ArgumentError("Tree height must be positive"))
    quadrature_order>=2 || throw(ArgumentError("quadrature_order must be at least 2"))
    order = Int(quadrature_order)
    breaks = sort!(unique(Float64.(quadrature_breakpoints)))
    all(t -> isfinite(t) && 0<=t<=height,breaks) || throw(ArgumentError("Quadrature breakpoints must be finite ages between 0 and tree height"))
    known_curve = environment isa Union{EnvCurve,SmoothEnvCurve}
    if known_curve
        first(environment.time)<=0 && last(environment.time)>=height || throw(ArgumentError("Environmental history must cover ages from 0 to tree height"))
    end
    analytic_linear = model===:EnvLin && known_curve
    # Count references before allocation. Shared intervals retain their own
    # quadrature, so branch-end truncations never change the original partition.
    cuts = sort!(unique!(vcat(_env_knots(environment),breaks)))
    young,old = zeros(tree.nedges),zeros(tree.nedges)
    offsets = Vector{Int}(undef,tree.nedges+1)
    offsets[1] = 1
    for edge in 1:tree.nedges
        age = height-tree.dist_from_root[tree.child_of_edge[edge]]
        young[edge] = max(age,0.0)
        old[edge] = max(age+tree.edge_length[edge],young[edge])
        count = old[edge]>young[edge] ? searchsortedfirst(cuts,old[edge])-searchsortedlast(cuts,young[edge]) : 0
        offsets[edge+1] = offsets[edge]+count
    end
    offsets[end]>1 || throw(ArgumentError("Tree has no positive-length branches"))
    ids = Vector{Int}(undef,offsets[end]-1)
    interval_index = Dict{Tuple{Float64,Float64},Int}()
    bounds = Tuple{Float64,Float64}[]
    for edge in 1:tree.nedges
        lo = young[edge]
        cursor = offsets[edge]
        lo==old[edge] && continue
        firstcut = searchsortedlast(cuts,lo)+1
        lastcut = searchsortedfirst(cuts,old[edge])-1
        for j in firstcut:lastcut+1
            hi = j<=lastcut ? cuts[j] : old[edge]
            key = (lo,hi)
            id = get(interval_index,key,0)
            if id==0
                push!(bounds,key)
                id = length(bounds)
                interval_index[key] = id
            end
            ids[cursor] = id
            cursor += 1
            lo = hi
        end
    end
    nintervals = length(bounds)
    values = Vector{Float64}(undef,analytic_linear ? 0 : nintervals*order)
    weights = similar(values)
    integrals = zeros(nintervals)
    if analytic_linear
        for (i,(lo,hi)) in enumerate(bounds)
            integrals[i] = _env_interval_integral(environment,lo,hi)
        end
        minimum_value,maximum_value = 0.0,0.0
    else
        coefficients = [i/sqrt(4.0*i*i-1) for i in 1:order-1]
        decomposition = eigen(SymTridiagonal(zeros(order),coefficients))
        nodes = decomposition.values
        base_weights = 2 .* decomposition.vectors[1,:].^2
        for (i,(lo,hi)) in enumerate(bounds)
            half = (hi-lo)/2
            center = lo+half
            for k in eachindex(nodes)
                value = Float64(environment(center+half*nodes[k]))
                isfinite(value) || throw(ArgumentError("Environmental function returned a non-finite value"))
                index = (i-1)*order+k
                values[index] = value
                weights[index] = half*base_weights[k]
            end
        end
        minimum_value,maximum_value = extrema(values)
    end
    if environment isa EnvCurve
        covered = [environment(0.0),environment(height)]
        append!(covered,environment.values[findall(t -> 0<=t<=height,environment.time)])
        lo,hi = extrema(covered)
        minimum_value,maximum_value = analytic_linear ? (lo,hi) : (min(minimum_value,lo),max(maximum_value,hi))
    elseif environment isa SmoothEnvCurve
        lo,hi = _env_smooth_range(environment,height)
        minimum_value,maximum_value = analytic_linear ? (lo,hi) : (min(minimum_value,lo),max(maximum_value,hi))
    end
    edge_integrals = zeros(tree.nedges)
    if analytic_linear
        for edge in 1:tree.nedges
            for j in offsets[edge]:offsets[edge+1]-1
                edge_integrals[edge] += integrals[ids[j]]
            end
        end
        # Analytical likelihood only needs per-edge sufficient statistics.
        offsets = ones(Int,tree.nedges+1)
        ids = Int[]
        integrals = Float64[]
    end
    return EnvPrecalc(environment,height,order,breaks,offsets,ids,values,weights,
        integrals,old-young,edge_integrals,analytic_linear,minimum_value,maximum_value)
end

function _env_edge_variances!(variances,cache::EnvPrecalc,sigma2,beta,model)
    isfinite(sigma2) && sigma2>0 && isfinite(beta) || return false
    if model===:EnvLin
        beta>0 || return false
        all(x -> sigma2*(1-x)+beta*x>0,(cache.minimum_environment,cache.maximum_environment)) || return false
    end
    if cache.analytic_linear
        model===:EnvLin || throw(ArgumentError("Analytic EnvLin cache cannot evaluate EnvExp"))
        for edge in eachindex(variances)
            isfinite(cache.edge_environment_integrals[edge]) || return false
            duration = cache.edge_durations[edge]
            # The mean lies within the validated range. Clamp integration roundoff
            # and use endpoint blending to retain tiny positive rates at env=1.
            mean_environment = duration==0 ? 0.0 : clamp(cache.edge_environment_integrals[edge]/duration,
                cache.minimum_environment,cache.maximum_environment)
            variance = duration*(sigma2*(1-mean_environment)+beta*mean_environment)
            isfinite(variance) && variance>=0 || return false
            variances[edge] = variance
        end
        return true
    end
    for interval in eachindex(cache.interval_variances)
        variance = 0.0
        for k in (interval-1)*cache.order+1:interval*cache.order
            rate = model===:EnvExp ? sigma2*exp(beta*cache.values[k]) : sigma2*(1-cache.values[k])+beta*cache.values[k]
            isfinite(rate) && rate>0 || return false
            variance += cache.weights[k]*rate
        end
        isfinite(variance) || return false
        cache.interval_variances[interval] = variance
    end
    for edge in eachindex(variances)
        variance = 0.0
        for j in cache.offsets[edge]:cache.offsets[edge+1]-1
            variance += cache.interval_variances[cache.interval_ids[j]]
        end
        isfinite(variance) || return false
        variances[edge] = variance
    end
    return true
end
