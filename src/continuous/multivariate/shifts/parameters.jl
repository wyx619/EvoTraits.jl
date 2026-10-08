function _shift_sigma_nparams(p, decomp)
    decomp === :cholesky && return p * (p + 1) ÷ 2
    decomp === :diagonal && return p
    decomp === :equaldiagonal && return 1
    decomp === :equal && return 1 + p * (p - 1) ÷ 2
    throw(ArgumentError("Unsupported shift Sigma_decomp=$decomp"))
end

function _shift_sigma(S, p, decomp)
    _shift_sigma_nparams(p, decomp)
    matrix = _shift_diffusion(S, p)
    if decomp in (:diagonal, :equaldiagonal)
        isdiag(matrix) || throw(ArgumentError("Sigma_decomp=$decomp requires zero off-diagonal entries"))
    end
    if decomp in (:equal, :equaldiagonal)
        all(==(matrix[1, 1]), diag(matrix)) ||
            throw(ArgumentError("Sigma_decomp=$decomp requires equal diagonal variances"))
    end
    return matrix
end

function _shift_pack_sigma(S, decomp)
    p = size(S, 1)
    matrix = _shift_sigma(S, p, decomp)
    decomp === :cholesky && return _mvbm1_pack_sigma(matrix)
    decomp === :diagonal && return log.(diag(matrix))
    decomp === :equaldiagonal && return [log(matrix[1, 1])]
    correlation = matrix / matrix[1, 1]
    L = Matrix(cholesky(Symmetric(correlation)).L)
    pars = [log(matrix[1, 1])]
    for j in 1:p-1, i in j+1:p
        push!(pars, L[i, j] / L[i, i])
    end
    return pars
end

function _shift_unpack_sigma(pars, p, decomp)
    length(pars) == _shift_sigma_nparams(p, decomp) || throw(ArgumentError("Sigma parameter count mismatch"))
    decomp === :cholesky && return _mvbm1_unpack_sigma(pars, p)
    decomp === :diagonal && return Matrix(Diagonal(exp.(pars)))
    variance = exp(pars[1])
    decomp === :equaldiagonal && return variance * Matrix{Float64}(I, p, p)
    L = Matrix{Float64}(I, p, p)
    index = 2
    for j in 1:p-1, i in j+1:p
        L[i, j] = pars[index]
        index += 1
    end
    for i in 1:p
        L[i, :] ./= norm(@view L[i, :])
    end
    S = variance * (L * L')
    for i in 1:p
        S[i, i] = variance
    end
    return S
end

function _shift_initial_sigma(empirical, decomp)
    decomp === :cholesky && return empirical
    decomp === :diagonal && return Matrix(Diagonal(diag(empirical)))
    variance = mean(diag(empirical))
    decomp === :equaldiagonal && return variance * Matrix{Float64}(I, size(empirical, 1), size(empirical, 1))
    scales = sqrt.(diag(empirical))
    S = variance * (empirical ./ (scales * scales'))
    for i in axes(S, 1)
        S[i, i] = variance
    end
    return S
end
function _shift_beta_bounds(bounds)
    bounds === nothing && return nothing
    bounds isa Tuple && length(bounds) == 2 &&
        all(x -> x isa Real && isfinite(x), bounds) && bounds[1] < bounds[2] ||
        throw(ArgumentError("beta_bounds must be finite increasing bounds or nothing"))
    return bounds
end

function _shift_effective_beta(beta, bounds)
    return bounds === nothing || bounds[1] <= beta <= bounds[2] ? Float64(beta) : 0.0
end
