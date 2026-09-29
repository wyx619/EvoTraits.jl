# Parameterization for the mvOUBM joint process.

function _mvoubm_A_spec(A_decomp::Symbol)
    return MVOUSpec(model = :mvOUBM, A_decomp = A_decomp)
end

function _mvoubm_A_block_nparams(A_decomp::Symbol, kY::Int)
    if A_decomp === :free
        return kY * kY
    elseif A_decomp === :schur
        return _mvou_A_block_nparams(_mvoubm_A_spec(:schur), kY)
    elseif A_decomp === :decomposable_real
        return kY * kY + kY
    elseif A_decomp === :upper_triangular || A_decomp === :lower_triangular
        return div(kY * (kY + 1), 2)
    end
    return _mvou_A_block_nparams(_mvoubm_A_spec(A_decomp), kY)
end

function _mvoubm_unpack_A_block(A_decomp::Symbol, pars::AbstractVector{<:Real}, kY::Int)
    if A_decomp === :free
        length(pars) == kY * kY || throw(ArgumentError("free A block must have $(kY*kY) entries"))
        return reshape(Float64.(pars), kY, kY)
    elseif A_decomp === :decomposable_real
        length(pars) == kY * kY + kY ||
            throw(ArgumentError("decomposable_real A block must have $(kY*kY + kY) entries"))
        lam = exp.(Float64.(pars[1:kY]))
        V = reshape(Float64.(pars[(kY + 1):end]), kY, kY)
        condV = cond(V)
        isfinite(condV) && condV < 1e8 || throw(ArgumentError("eigenvector matrix is ill-conditioned"))
        return (V * Diagonal(lam)) / V
    elseif A_decomp === :upper_triangular || A_decomp === :lower_triangular
        expected = div(kY * (kY + 1), 2)
        length(pars) == expected ||
            throw(ArgumentError("triangular A block must have $expected entries"))
        A = zeros(Float64, kY, kY)
        idx = 1
        if A_decomp === :upper_triangular
            for j in 1:kY
                A[j, j] = exp(Float64(pars[idx]))
                idx += 1
                for i in 1:(j - 1)
                    A[i, j] = Float64(pars[idx])
                    idx += 1
                end
            end
        else
            for j in 1:kY
                A[j, j] = exp(Float64(pars[idx]))
                idx += 1
                for i in (j + 1):kY
                    A[i, j] = Float64(pars[idx])
                    idx += 1
                end
            end
        end
        return A
    end
    return _mvou_unpack_A(_mvoubm_A_spec(A_decomp), pars, kY)
end

function _mvoubm_pack_A_block(A_decomp::Symbol, A::AbstractMatrix{<:Real})
    if A_decomp === :free
        return vec(Matrix{Float64}(A))
    elseif A_decomp === :decomposable_real
        kY = size(A, 1)
        Eig = eigen(A)
        all(x -> abs(imag(x)) < 1e-8, Eig.values) ||
            throw(ArgumentError("decomposable_real requires real eigenvalues"))
        V = Matrix{Float64}(Eig.vectors)
        det(V) != 0.0 || throw(ArgumentError("eigenvector matrix is singular"))
        return vcat(log.(max.(real.(Eig.values), 1e-10)), vec(V))
    elseif A_decomp === :upper_triangular || A_decomp === :lower_triangular
        kY = size(A, 1)
        pars = Float64[]
        if A_decomp === :upper_triangular
            for j in 1:kY
                push!(pars, log(max(Float64(A[j, j]), 1e-10)))
                for i in 1:(j - 1)
                    push!(pars, Float64(A[i, j]))
                end
            end
        else
            for j in 1:kY
                push!(pars, log(max(Float64(A[j, j]), 1e-10)))
                for i in (j + 1):kY
                    push!(pars, Float64(A[i, j]))
                end
            end
        end
        return pars
    end
    return _mvou_pack_A(_mvoubm_A_spec(A_decomp), A)
end

function _mvoubm_unpack_params(pars::AbstractVector{<:Real}, kY::Int, kX::Int, nregimes::Int, A_decomp::Symbol)
    Ablock = _mvoubm_A_block_nparams(A_decomp, kY)
    SblockY = div(kY * (kY + 1), 2)
    SblockX = div(kX * (kX + 1), 2)
    expected = Ablock + kY * kX + SblockY + SblockX + kY * nregimes + kX
    length(pars) == expected ||
        throw(ArgumentError("mvOUBM parameter vector must have $expected entries"))
    idx = 1
    A = _mvoubm_unpack_A_block(A_decomp, pars[idx:(idx + Ablock - 1)], kY)
    idx += Ablock
    B = reshape(Float64.(pars[idx:(idx + kY * kX - 1)]), kY, kX)
    idx += kY * kX
    SigmaY = _mvou_unpack_spd(pars[idx:(idx + SblockY - 1)], kY)
    idx += SblockY
    SigmaX = _mvou_unpack_spd(pars[idx:(idx + SblockX - 1)], kX)
    idx += SblockX
    mPsi = reshape(Float64.(pars[idx:(idx + kY * nregimes - 1)]), kY, nregimes)
    idx += kY * nregimes
    vX0 = Float64.(pars[idx:end])
    return (A = A, B = B, SigmaY = SigmaY, SigmaX = SigmaX, mPsi = mPsi, vX0 = vX0)
end

function _mvoubm_nparams(kY::Int, kX::Int, nregimes::Int, A_decomp::Symbol)
    return _mvoubm_A_block_nparams(A_decomp, kY) + kY * kX +
           div(kY * (kY + 1), 2) + div(kX * (kX + 1), 2) + kY * nregimes + kX
end
