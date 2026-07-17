@inline function compute_aic(loglik::Real, nparams::Integer)
    return 2.0 * nparams - 2.0 * Float64(loglik)
end

@inline function compute_aicc(loglik::Real, nparams::Integer, n::Integer)
    aic = compute_aic(loglik, nparams)
    denominator = n - nparams - 1
    denominator > 0 || return Inf
    return aic + (2.0 * nparams * (nparams + 1)) / denominator
end

"""
    aic(result)

Return the AIC recorded in a fit result.
"""
@inline aic(result) = result.aic

"""
    aicc(result; nobs=nothing)

Return AICc using the observation count recorded in the fit result. For
multivariate fits this count is the number of observed trait entries.
"""
@inline function aicc(result; nobs = nothing)
    n =
        nobs === nothing ?
        (hasproperty(result, :nobs) ? getproperty(result, :nobs) : 0) :
        Int(nobs)
    n > 0 || throw(ArgumentError("AICc requires a positive observation count; pass `nobs=` for result types that do not store it"))
    return compute_aicc(loglik(result), nparams(result), n)
end

"""
    loglik(result)

Return the log-likelihood recorded in a fit result.
"""
@inline loglik(result) = result.loglik

"""
    nparams(result)

Return the number of free parameters recorded in a fit result.
"""
@inline nparams(result) = result.nparams

"""
    model(result)

Return the symbolic model name stored in a fit result.
"""
@inline model(result) = result.model

@inline _aic_order_value(x::Real) = isfinite(Float64(x)) ? Float64(x) : Inf

"""
    delta_aic(aic_values)

Convert a vector of AIC values into delta-AIC values relative to the best
finite entry.
"""
function delta_aic(aic_values::AbstractVector{<:Real})
    vals = Float64.(aic_values)
    isempty(vals) && return Float64[]
    best = minimum(vals)
    return vals .- best
end

function _aic_summary(name, result)
    return (
        name = Symbol(name),
        model = getproperty(result, :model),
        aic = aic(result),
        loglik = loglik(result),
        nparams = nparams(result),
        success = getproperty(result, :success),
    )
end

"""
    aic_table(models...)

Build a sorted AIC summary table from `name => fit_result` pairs. Each returned
row contains the model name, fitted model symbol, AIC, delta-AIC, log-likelihood,
parameter count, and success flag.
"""
function aic_table(models::Pair...)
    isempty(models) && return NamedTuple[]
    rows = [_aic_summary(name, result) for (name, result) in models]
    order = sortperm(_aic_order_value.(getfield.(rows, :aic)))
    sorted_rows = rows[order]
    deltas = delta_aic(_aic_order_value.(getfield.(sorted_rows, :aic)))

    return [
        (
            name = row.name,
            model = row.model,
            aic = row.aic,
            delta_aic = deltas[i],
            loglik = row.loglik,
            nparams = row.nparams,
            success = row.success,
        ) for (i, row) in enumerate(sorted_rows)
    ]
end

"""
    best_model(aic_rows)

Return the first successful finite-AIC row from an `aic_table` result.
"""
@inline function best_model(aic_rows::AbstractVector{<:NamedTuple})
    isempty(aic_rows) && throw(ArgumentError("aic_rows is empty"))
    for row in aic_rows
        if row.success && isfinite(row.aic)
            return row
        end
    end
    throw(ArgumentError("No successful finite-AIC models found"))
end
