function _corhmm_base_index_matrix(nstates::Integer, model::Symbol)
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    mk_nrates(model, nstates)
    idx = zeros(Int, nstates, nstates)
    if model === :ER
        for i in 1:nstates, j in 1:nstates
            i != j && (idx[i, j] = 1)
        end
    elseif model === :SYM
        p = 1
        for i in 1:(nstates - 1), j in (i + 1):nstates
            idx[i, j] = p
            idx[j, i] = p
            p += 1
        end
    elseif model === :ARD
        p = 1
        for j in 1:nstates, i in 1:nstates
            if i != j
                idx[i, j] = p
                p += 1
            end
        end
    elseif model === :SUEDE
        for i in 1:nstates
            i < nstates && (idx[i, i + 1] = 1)
            i > 1 && (idx[i, i - 1] = 2)
        end
    elseif model === :SRD
        p = 1
        for i in 1:(nstates - 1)
            idx[i, i + 1] = p
            p += 1
        end
        for i in 2:nstates
            idx[i, i - 1] = p
            p += 1
        end
    else
        throw(ArgumentError("Unsupported corHMM model=$model"))
    end
    return idx
end

function _offset_positive_indices(mat::AbstractMatrix{<:Integer}, offset::Integer)
    out = zeros(Int, size(mat))
    for i in axes(mat, 1), j in axes(mat, 2)
        mat[i, j] > 0 && (out[i, j] = Int(mat[i, j]) + Int(offset))
    end
    return out
end

function _corhmm_rate_class_matrix(rate_cat::Integer)
    rc = _validate_corhmm_rate_cat(rate_cat)
    mat = zeros(Int, rc, rc)
    p = 1
    for j in 1:rc, i in 1:rc
        if i != j
            mat[i, j] = p
            p += 1
        end
    end
    return mat
end

function rateindex(observed_nstates::Integer; rate_cat::Integer = 1, model::Symbol = :ARD)
    rc = _validate_corhmm_rate_cat(rate_cat)
    observed_nstates >= 2 || throw(ArgumentError("observed_nstates must be at least 2"))
    base = _corhmm_base_index_matrix(observed_nstates, model)
    if rc == 1
        return base
    end

    nstates = observed_nstates * rc
    out = zeros(Int, nstates, nstates)
    base_nparams = maximum(base)
    for r in 1:rc
        offset = model === :ARD ? base_nparams * ((r - 1) * (r + 2) ÷ 2) : base_nparams * (r - 1)
        block = _offset_positive_indices(base, offset)
        lo = (r - 1) * observed_nstates + 1
        hi = r * observed_nstates
        out[lo:hi, lo:hi] .= block
    end

    hidden_offset = maximum(out)
    rate_class = _corhmm_rate_class_matrix(rc)
    for from_r in 1:rc, to_r in 1:rc
        from_r == to_r && continue
        param = hidden_offset + rate_class[from_r, to_r]
        for obs in 1:observed_nstates
            from_state = (from_r - 1) * observed_nstates + obs
            to_state = (to_r - 1) * observed_nstates + obs
            out[from_state, to_state] = param
        end
    end
    return out
end

function rates_to_q(rates::AbstractVector{<:Real}, observed_nstates::Integer; model::Symbol = :ARD, rate_cat::Integer = 1)
    index_matrix = rateindex(observed_nstates; model = model, rate_cat = rate_cat)
    return qfromindex(rates, index_matrix)
end

function normalize_ratematrix(rate_mat::AbstractMatrix)
    size(rate_mat, 1) == size(rate_mat, 2) || throw(ArgumentError("rate_mat must be square"))
    out = zeros(Int, size(rate_mat))
    for i in axes(rate_mat, 1), j in axes(rate_mat, 2)
        if i == j
            continue
        end
        value = rate_mat[i, j]
        if value === missing
            continue
        end
        if value isa Real
            v = Float64(value)
            if !isfinite(v)
                continue
            elseif v <= 0.0
                continue
            elseif !isinteger(v)
                throw(ArgumentError("rate_mat positive entries must be integer parameter ids"))
            else
                out[i, j] = Int(v)
            end
        else
            s = strip(string(value))
            isempty(s) && continue
            lowercase(s) in ("na", "nan", "missing") && continue
            v = parse(Float64, s)
            if v <= 0.0
                continue
            elseif !isinteger(v)
                throw(ArgumentError("rate_mat positive entries must be integer parameter ids"))
            else
                out[i, j] = Int(v)
            end
        end
    end
    maximum(out) >= 1 || throw(ArgumentError("rate_mat must contain at least one positive parameter id"))
    return out
end

function qfromindex(rates::AbstractVector{<:Real}, index_matrix::AbstractMatrix{<:Integer})
    size(index_matrix, 1) == size(index_matrix, 2) || throw(ArgumentError("index_matrix must be square"))
    nparams = maximum(index_matrix)
    length(rates) == nparams || throw(ArgumentError("Expected $nparams rates, got $(length(rates))"))
    Q = zeros(Float64, size(index_matrix))
    for i in axes(index_matrix, 1)
        rowsum = 0.0
        for j in axes(index_matrix, 2)
            if i == j
                continue
            end
            idx = Int(index_matrix[i, j])
            if idx > 0
                rate = Float64(rates[idx])
                rate >= 0.0 || throw(ArgumentError("Rates must be non-negative"))
                Q[i, j] = rate
                rowsum += rate
            end
        end
        Q[i, i] = -rowsum
    end
    return Q
end

function solutionmatrix(rates::AbstractVector{<:Real}, index_matrix::AbstractMatrix{<:Integer})
    nparams = maximum(index_matrix)
    length(rates) == nparams || throw(ArgumentError("Expected $nparams rates, got $(length(rates))"))
    out = fill(NaN, size(index_matrix))
    for i in axes(index_matrix, 1), j in axes(index_matrix, 2)
        idx = Int(index_matrix[i, j])
        idx > 0 && (out[i, j] = Float64(rates[idx]))
    end
    return out
end

corhmm_rate_index_matrix(args...; kwargs...) = rateindex(args...; kwargs...)
corhmm_rates_to_Q(args...; kwargs...) = rates_to_q(args...; kwargs...)
corhmm_normalize_rate_matrix(args...; kwargs...) = normalize_ratematrix(args...; kwargs...)
corhmm_index_to_Q(args...; kwargs...) = qfromindex(args...; kwargs...)
corhmm_solution_matrix(args...; kwargs...) = solutionmatrix(args...; kwargs...)
