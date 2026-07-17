"""
    er_nrates(nstates)

Return the number of free rate parameters in the equal-rates (`ER`) Mk model.
"""
function er_nrates(nstates::Integer)
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    return 1
end

"""
    sym_nrates(nstates)

Return the number of free rate parameters in the symmetric (`SYM`) Mk model.
"""
function sym_nrates(nstates::Integer)
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    return (nstates * (nstates - 1)) ÷ 2
end

"""
    suede_nrates(nstates)

Return the number of free rate parameters in the ordered two-direction
(`SUEDE`) Mk model.
"""
function suede_nrates(nstates::Integer)
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    return 2
end

"""
    srd_nrates(nstates)

Return the number of free rate parameters in the ordered stepwise-rate
difference (`SRD`) Mk model.
"""
function srd_nrates(nstates::Integer)
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    return 2 * (nstates - 1)
end

"""
    ard_nrates(nstates)

Return the number of free rate parameters in the all-rates-different (`ARD`)
Mk model.
"""
function ard_nrates(nstates::Integer)
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    return nstates * (nstates - 1)
end

"""
    mk_nrates(rate_model, nstates)

Return the number of free parameters for an exported Mk rate-model
parameterization.
"""
function mk_nrates(rate_model::Symbol, nstates::Integer)
    if rate_model === :ER
        return er_nrates(nstates)
    elseif rate_model === :SYM
        return sym_nrates(nstates)
    elseif rate_model === :SUEDE
        return suede_nrates(nstates)
    elseif rate_model === :SRD
        return srd_nrates(nstates)
    elseif rate_model === :ARD
        return ard_nrates(nstates)
    end
    throw(ArgumentError("Unsupported rate_model=$rate_model"))
end

"""
    er_rates_to_Q(rates, nstates)

Construct an `ER` Mk rate matrix from its free-rate vector.
"""
function er_rates_to_Q(rates::AbstractVector{<:Real}, nstates::Integer)
    length(rates) == 1 || throw(ArgumentError("Expected 1 ER rate, got $(length(rates))"))
    rate = Float64(rates[1])
    rate >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
    Q = fill(rate, nstates, nstates)
    for i in 1:nstates
        Q[i, i] = -rate * (nstates - 1)
    end
    return Q
end

"""
    sym_rates_to_Q(rates, nstates)

Construct a `SYM` Mk rate matrix from its free-rate vector.
"""
function sym_rates_to_Q(rates::AbstractVector{<:Real}, nstates::Integer)
    nrates = sym_nrates(nstates)
    length(rates) == nrates || throw(ArgumentError("Expected $nrates SYM rates, got $(length(rates))"))
    Q = zeros(Float64, nstates, nstates)
    idx = 1
    for i in 1:(nstates - 1)
        for j in (i + 1):nstates
            rate = Float64(rates[idx])
            rate >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
            Q[i, j] = rate
            Q[j, i] = rate
            idx += 1
        end
    end
    for i in 1:nstates
        Q[i, i] = -sum(@view Q[i, :])
    end
    return Q
end

"""
    suede_rates_to_Q(rates, nstates)

Construct a `SUEDE` Mk rate matrix from its two ordered-direction rates.
"""
function suede_rates_to_Q(rates::AbstractVector{<:Real}, nstates::Integer)
    length(rates) == 2 || throw(ArgumentError("Expected 2 SUEDE rates, got $(length(rates))"))
    up = Float64(rates[1])
    down = Float64(rates[2])
    up >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
    down >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
    Q = zeros(Float64, nstates, nstates)
    for i in 1:nstates
        if i < nstates
            Q[i, i + 1] = up
        end
        if i > 1
            Q[i, i - 1] = down
        end
        Q[i, i] = -sum(@view Q[i, :])
    end
    return Q
end

"""
    srd_rates_to_Q(rates, nstates)

Construct an `SRD` Mk rate matrix from its ordered stepwise-rate vector.
"""
function srd_rates_to_Q(rates::AbstractVector{<:Real}, nstates::Integer)
    nrates = srd_nrates(nstates)
    length(rates) == nrates || throw(ArgumentError("Expected $nrates SRD rates, got $(length(rates))"))
    Q = zeros(Float64, nstates, nstates)
    idx = 1
    for i in 1:(nstates - 1)
        rate = Float64(rates[idx])
        rate >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
        Q[i, i + 1] = rate
        idx += 1
    end
    for i in 2:nstates
        rate = Float64(rates[idx])
        rate >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
        Q[i, i - 1] = rate
        idx += 1
    end
    for i in 1:nstates
        Q[i, i] = -sum(@view Q[i, :])
    end
    return Q
end

"""
    ard_rates_to_Q(rates, nstates)

Construct an `ARD` Mk rate matrix from its free-rate vector.
"""
function ard_rates_to_Q(rates::AbstractVector{<:Real}, nstates::Integer)
    nrates = ard_nrates(nstates)
    length(rates) == nrates || throw(ArgumentError("Expected $nrates ARD rates, got $(length(rates))"))
    Q = zeros(Float64, nstates, nstates)
    idx = 1
    for i in 1:nstates
        rowsum = 0.0
        for j in 1:nstates
            if i == j
                continue
            end
            rate = Float64(rates[idx])
            rate >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
            Q[i, j] = rate
            rowsum += rate
            idx += 1
        end
        Q[i, i] = -rowsum
    end
    return Q
end

"""
    ard_Q_to_rates(Q)

Recover the free-rate vector corresponding to an `ARD` Mk rate matrix.
"""
function ard_Q_to_rates(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    nstates = size(Q, 1)
    rates = Vector{Float64}(undef, ard_nrates(nstates))
    idx = 1
    for i in 1:nstates, j in 1:nstates
        if i != j
            rates[idx] = Float64(Q[i, j])
            idx += 1
        end
    end
    return rates
end

"""
    er_Q_to_rates(Q)

Recover the free-rate vector corresponding to an `ER` Mk rate matrix.
"""
function er_Q_to_rates(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    nstates = size(Q, 1)
    ref = Float64(Q[1, 2])
    for i in 1:nstates, j in 1:nstates
        if i != j && !isapprox(Float64(Q[i, j]), ref; atol = 1e-8, rtol = 1e-8)
            throw(ArgumentError("Q is not consistent with an ER parameterization"))
        end
    end
    return [ref]
end

"""
    sym_Q_to_rates(Q)

Recover the free-rate vector corresponding to a `SYM` Mk rate matrix.
"""
function sym_Q_to_rates(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    nstates = size(Q, 1)
    rates = Vector{Float64}(undef, sym_nrates(nstates))
    idx = 1
    for i in 1:(nstates - 1)
        for j in (i + 1):nstates
            qij = Float64(Q[i, j])
            qji = Float64(Q[j, i])
            isapprox(qij, qji; atol = 1e-8, rtol = 1e-8) || throw(ArgumentError("Q is not consistent with a SYM parameterization"))
            rates[idx] = qij
            idx += 1
        end
    end
    return rates
end

"""
    suede_Q_to_rates(Q)

Recover the free-rate vector corresponding to a `SUEDE` Mk rate matrix.
"""
function suede_Q_to_rates(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    nstates = size(Q, 1)
    for i in 1:nstates, j in 1:nstates
        if abs(i - j) > 1 && abs(Float64(Q[i, j])) > 1e-8
            throw(ArgumentError("Q is not consistent with a SUEDE parameterization"))
        end
    end
    up = nstates >= 2 ? Float64(Q[1, 2]) : 0.0
    down = nstates >= 2 ? Float64(Q[2, 1]) : 0.0
    for i in 1:(nstates - 1)
        isapprox(Float64(Q[i, i + 1]), up; atol = 1e-8, rtol = 1e-8) || throw(ArgumentError("Q is not consistent with a SUEDE up-rate parameterization"))
    end
    for i in 2:nstates
        isapprox(Float64(Q[i, i - 1]), down; atol = 1e-8, rtol = 1e-8) || throw(ArgumentError("Q is not consistent with a SUEDE down-rate parameterization"))
    end
    return [up, down]
end

"""
    srd_Q_to_rates(Q)

Recover the free-rate vector corresponding to an `SRD` Mk rate matrix.
"""
function srd_Q_to_rates(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    nstates = size(Q, 1)
    for i in 1:nstates, j in 1:nstates
        if abs(i - j) > 1 && abs(Float64(Q[i, j])) > 1e-8
            throw(ArgumentError("Q is not consistent with an SRD parameterization"))
        end
    end
    rates = Vector{Float64}(undef, srd_nrates(nstates))
    idx = 1
    for i in 1:(nstates - 1)
        rates[idx] = Float64(Q[i, i + 1])
        idx += 1
    end
    for i in 2:nstates
        rates[idx] = Float64(Q[i, i - 1])
        idx += 1
    end
    return rates
end

"""
    mk_rates_to_Q(rates, nstates; rate_model=:ARD)

Construct an Mk rate matrix from a free-rate vector under the chosen exported
rate-model parameterization.
"""
function mk_rates_to_Q(rates::AbstractVector{<:Real}, nstates::Integer; rate_model::Symbol = :ARD)
    if rate_model === :ER
        return er_rates_to_Q(rates, nstates)
    elseif rate_model === :SYM
        return sym_rates_to_Q(rates, nstates)
    elseif rate_model === :SUEDE
        return suede_rates_to_Q(rates, nstates)
    elseif rate_model === :SRD
        return srd_rates_to_Q(rates, nstates)
    elseif rate_model === :ARD
        return ard_rates_to_Q(rates, nstates)
    end
    throw(ArgumentError("Unsupported rate_model=$rate_model"))
end

"""
    mk_Q_to_rates(Q; rate_model=:ARD)

Recover a free-rate vector from an Mk rate matrix under the chosen exported
rate-model parameterization.
"""
function mk_Q_to_rates(Q::AbstractMatrix{<:Real}; rate_model::Symbol = :ARD)
    if rate_model === :ER
        return er_Q_to_rates(Q)
    elseif rate_model === :SYM
        return sym_Q_to_rates(Q)
    elseif rate_model === :SUEDE
        return suede_Q_to_rates(Q)
    elseif rate_model === :SRD
        return srd_Q_to_rates(Q)
    elseif rate_model === :ARD
        return ard_Q_to_rates(Q)
    end
    throw(ArgumentError("Unsupported rate_model=$rate_model"))
end
