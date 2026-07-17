function _continuous_print_named_values(io::IO, title::AbstractString, values::AbstractVector{<:Real}, names::Vector{String})
    isempty(values) && return nothing
    println(io, title)
    for i in eachindex(values)
        label = i <= length(names) ? names[i] : "regime$(i)"
        println(io, "    ", rpad(label, 14), round(Float64(values[i]); digits = 6))
    end
    return nothing
end

function _continuous_print_scalar_value(io::IO, label::AbstractString, value::Real)
    isfinite(Float64(value)) || return false
    println(io, "    ", rpad(label, 14), round(Float64(value); digits = 6))
    return true
end

function _continuous_print_scalar_block(io::IO, title::AbstractString, label::AbstractString, value::Real)
    isfinite(Float64(value)) || return false
    println(io, title)
    _continuous_print_scalar_value(io, label, value)
    return true
end

@inline function _continuous_print_blank_before(io::IO, already_printed::Bool)
    already_printed && println(io)
    return nothing
end

@inline function _continuous_same_estimate(a::Real, b::Real)
    af = Float64(a)
    bf = Float64(b)
    return isfinite(af) && isfinite(bf) && isapprox(af, bf; rtol = 1e-8, atol = 1e-10)
end

function Base.summary(res::ContinuousFitResult)
    return string(res.model, " univariate fit (logLik=", round(res.loglik; digits = 6), ", AIC=", round(res.aic; digits = 6), ", nparams=", res.nparams, ")")
end

Base.show(io::IO, res::ContinuousFitResult) = print(io, summary(res))

function Base.show(io::IO, ::MIME"text/plain", res::ContinuousFitResult)
    println(io, res.model, " univariate fit")
    println(io, "  success:   ", res.success)
    println(io, "  converged: ", res.converged)
    println(io, "  logLik:    ", res.loglik)
    println(io, "  AIC:       ", res.aic)
    println(io, "  nparams:   ", res.nparams)
    println(io, "  trait:     ", res.trait_name)
    res.iterations > 0 && println(io, "  iterations:", res.iterations)
    res.f_calls > 0 && println(io, "  f_calls:   ", res.f_calls)
    println(io)
    printed = false
    if res.model === :BM1 || res.model === :EB
        printed |= _continuous_print_scalar_block(io, "root", "state", res.root_state)
        _continuous_print_blank_before(io, printed)
        printed |= _continuous_print_scalar_block(io, "sigma2", "global", res.sigma2)
        if isfinite(res.beta)
            _continuous_print_blank_before(io, printed)
            printed |= _continuous_print_scalar_block(io, "beta", "global", res.beta)
        end
    else
        printed |= _continuous_print_scalar_block(io, "theta", "global", res.theta)
        if isfinite(res.alpha)
            _continuous_print_blank_before(io, printed)
            printed |= _continuous_print_scalar_block(io, "alpha", "global", res.alpha)
        end
        if isfinite(res.sigma2)
            _continuous_print_blank_before(io, printed)
            printed |= _continuous_print_scalar_block(io, "sigma2", "global", res.sigma2)
        end
        if isfinite(res.beta)
            _continuous_print_blank_before(io, printed)
            printed |= _continuous_print_scalar_block(io, "beta", "global", res.beta)
        end
        if isfinite(res.root_state) && !_continuous_same_estimate(res.root_state, res.theta)
            _continuous_print_blank_before(io, printed)
            printed |= _continuous_print_scalar_block(io, "root", "state", res.root_state)
        end
    end
end

function Base.summary(res::ContinuousMultiRegimeResult)
    return string(res.model, " univariate multi-regime fit (logLik=", round(res.loglik; digits = 6), ", AIC=", round(res.aic; digits = 6), ", nparams=", res.nparams, ")")
end

Base.show(io::IO, res::ContinuousMultiRegimeResult) = print(io, summary(res))

function Base.show(io::IO, ::MIME"text/plain", res::ContinuousMultiRegimeResult)
    regime_names = isempty(res.regime_names) ? _continuous_default_regime_names(res.nregimes) : res.regime_names
    println(io, res.model, " univariate multi-regime fit")
    println(io, "  success:   ", res.success)
    println(io, "  converged: ", res.converged)
    println(io, "  logLik:    ", res.loglik)
    println(io, "  AIC:       ", res.aic)
    println(io, "  nparams:   ", res.nparams)
    println(io, "  trait:     ", res.trait_name)
    println(io, "  regimes:   ", join(regime_names, ", "))
    res.iterations > 0 && println(io, "  iterations:", res.iterations)
    res.f_calls > 0 && println(io, "  f_calls:   ", res.f_calls)
    println(io)
    printed = false
    if !isempty(res.theta_regimes)
        _continuous_print_blank_before(io, printed)
        _continuous_print_named_values(io, "theta by regime", res.theta_regimes, regime_names)
        printed = true
    elseif isfinite(res.root_state)
        _continuous_print_blank_before(io, printed)
        _continuous_print_scalar_block(io, "root", "state", res.root_state)
        printed = true
    elseif isfinite(res.theta)
        _continuous_print_blank_before(io, printed)
        _continuous_print_scalar_block(io, "root", "state", res.theta)
        printed = true
    end
    if !isempty(res.alpha_regimes)
        _continuous_print_blank_before(io, printed)
        _continuous_print_named_values(io, "alpha by regime", res.alpha_regimes, regime_names)
        printed = true
    elseif isfinite(res.alpha)
        _continuous_print_blank_before(io, printed)
        _continuous_print_scalar_block(io, "alpha", "global", res.alpha)
        printed = true
    end
    if !isempty(res.sigma2)
        sigma_names = length(res.sigma2) == 1 ? ["global"] : regime_names
        _continuous_print_blank_before(io, printed)
        _continuous_print_named_values(io, length(res.sigma2) == 1 ? "sigma2" : "sigma2 by regime", res.sigma2, sigma_names)
        printed = true
    end
    if !isempty(res.beta_regimes)
        _continuous_print_blank_before(io, printed)
        _continuous_print_named_values(io, "beta by regime", res.beta_regimes, regime_names)
        printed = true
    elseif isfinite(res.beta)
        _continuous_print_blank_before(io, printed)
        _continuous_print_scalar_block(io, "beta", "global", res.beta)
        printed = true
    end
    if !isempty(res.theta_regimes) && isfinite(res.root_state)
        _continuous_print_blank_before(io, printed)
        _continuous_print_scalar_block(io, "root", "state", res.root_state)
    end
end

function _mv_print_matrix(io::IO, M::AbstractMatrix{<:Real}, row_names::Vector{String}, col_names::Vector{String}; digits::Int = 6, indent::AbstractString = "    ")
    print(io, indent, rpad("", 14))
    for name in col_names
        print(io, lpad(name, 14))
    end
    println(io)
    for i in axes(M, 1)
        print(io, indent, rpad(row_names[i], 14))
        for j in axes(M, 2)
            print(io, lpad(string(round(Float64(M[i, j]); digits = digits)), 14))
        end
        println(io)
    end
end

function _mv_print_vector_row(io::IO, values::AbstractVector{<:Real}, label::AbstractString, trait_names::Vector{String}; digits::Int = 6, indent::AbstractString = "    ")
    print(io, indent, rpad(label, 14))
    for j in eachindex(trait_names)
        print(io, trait_names[j], "=", round(Float64(values[j]); digits = digits))
        j < length(trait_names) && print(io, ", ")
    end
    println(io)
end

function Base.summary(res::MVContinuousBMResult)
    return string(res.model, " multivariate fit (logLik=", round(res.loglik; digits = 6), ", AIC=", round(res.aic; digits = 6), ", nparams=", res.nparams, ")")
end

Base.show(io::IO, res::MVContinuousBMResult) = print(io, summary(res))

function Base.show(io::IO, ::MIME"text/plain", res::MVContinuousBMResult)
    trait_names = isempty(res.trait_names) ? _mv_default_trait_names(res.ntraits) : res.trait_names
    println(io, res.model, " multivariate fit")
    println(io, "  success:   ", res.success)
    println(io, "  converged: ", res.converged)
    println(io, "  logLik:    ", res.loglik)
    println(io, "  AIC:       ", res.aic)
    println(io, "  nparams:   ", res.nparams)
    println(io, "  traits:    ", join(trait_names, ", "))
    if isfinite(res.beta)
        println(io, "  beta:      ", res.beta)
    end
    println(io)
    println(io, "theta")
    _mv_print_vector_row(io, res.theta, "root", trait_names)
    println(io)
    println(io, "sigma")
    _mv_print_matrix(io, res.sigma, trait_names, trait_names)
end

function Base.summary(res::MVContinuousMultiBMResult)
    return string(res.model, " multivariate multi-regime fit (logLik=", round(res.loglik; digits = 6), ", AIC=", round(res.aic; digits = 6), ", nparams=", res.nparams, ")")
end

Base.show(io::IO, res::MVContinuousMultiBMResult) = print(io, summary(res))

function Base.show(io::IO, ::MIME"text/plain", res::MVContinuousMultiBMResult)
    trait_names = isempty(res.trait_names) ? _mv_default_trait_names(res.ntraits) : res.trait_names
    regime_names = isempty(res.regime_names) ? _mv_default_regime_names(res.nregimes) : res.regime_names
    println(io, res.model, " multivariate multi-regime fit")
    println(io, "  success:   ", res.success)
    println(io, "  converged: ", res.converged)
    println(io, "  logLik:    ", res.loglik)
    println(io, "  AIC:       ", res.aic)
    println(io, "  nparams:   ", res.nparams)
    println(io, "  traits:    ", join(trait_names, ", "))
    println(io, "  regimes:   ", join(regime_names, ", "))
    println(io)
    println(io, "theta")
    for r in 1:res.nregimes
        _mv_print_vector_row(io, @view(res.theta[r, :]), regime_names[r], trait_names)
    end
    println(io)
    println(io, "sigma by regime")
    for r in 1:res.nregimes
        println(io, "  ", regime_names[r])
        _mv_print_matrix(io, @view(res.sigma[:, :, r]), trait_names, trait_names; indent = "    ")
    end
end
