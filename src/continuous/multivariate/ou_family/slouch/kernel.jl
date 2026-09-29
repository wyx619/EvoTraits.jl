# mvOUBM joint-process edge kernel.
#
# Joint state Z = (Y, X) with block-triangular drift
#     H = [ A  B ; 0  0 ]      Theta_r = [ mPsi_r ; 0 ]
# and joint diffusion Sigma_ee.  Phi(t) = exp(-H t) and the incremental
# covariance W(t) = int_0^t e^{-Hs} Sigma_ee e^{-H's} ds are computed from the
# complex eigendecomposition of H.  Unlike the plain OU branch cache, the X
# block is non-stationary BM, so Q must be W(t), not Sstat - Phi*Sstat*Phi'.

function _mvoubm_pair_decay(t::Float64, li::ComplexF64, lj::ComplexF64)
    x = li + lj
    return abs(real(x)) < 1e-12 ? complex(t) : (1 - exp(-x * t)) / x
end

function _mvoubm_Phi!(
    Phi::AbstractMatrix{Float64},
    t::Float64,
    eigH,
    VinvH::AbstractMatrix{ComplexF64},
    expbuf::AbstractVector{ComplexF64},
    cbuf::AbstractMatrix{ComplexF64},
    cbuf2::AbstractMatrix{ComplexF64},
)
    expbuf .= exp.(-eigH.values .* t)
    mul!(cbuf, eigH.vectors, Diagonal(expbuf))
    mul!(cbuf2, cbuf, VinvH)
    Phi .= real.(cbuf2)
    return Phi
end

function _mvoubm_Phi_W!(
    Phi::AbstractMatrix{Float64},
    W::AbstractMatrix{Float64},
    t::Float64,
    eigH,
    VinvH::AbstractMatrix{ComplexF64},
    Mrot::AbstractMatrix{ComplexF64},
    work::AbstractMatrix{ComplexF64},
    expbuf::AbstractVector{ComplexF64},
    cbuf::AbstractMatrix{ComplexF64},
    cbuf2::AbstractMatrix{ComplexF64},
)
    lam = eigH.values
    p = length(lam)
    _mvoubm_Phi!(Phi, t, eigH, VinvH, expbuf, cbuf, cbuf2)
    @inbounds for j in 1:p, i in 1:p
        work[i, j] = Mrot[i, j] * _mvoubm_pair_decay(t, lam[i], lam[j])
    end
    mul!(cbuf, eigH.vectors, work)
    mul!(cbuf2, cbuf, adjoint(eigH.vectors))
    W .= real.(cbuf2)
    @inbounds for j in 1:p
        for i in (j + 1):p
            v = 0.5 * (W[i, j] + W[j, i])
            W[i, j] = v
            W[j, i] = v
        end
    end
    return Phi, W
end
