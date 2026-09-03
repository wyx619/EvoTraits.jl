@inline function _mvou_transition(A::AbstractMatrix{<:Real}, t::Real)
    return exp(-Matrix{Float64}(A) * Float64(t))
end

@inline function _mvou_transition_eigen(A::AbstractMatrix{<:Real})
    return eigen(Symmetric(Matrix{Float64}(A)))
end

@inline function _mvou_transition_factor(A::AbstractMatrix{<:Real}, A_decomp::Symbol)
    if A_decomp === :cholesky
        return _mvou_transition_eigen(A)
    elseif A_decomp === :schur
        return schur(Matrix{Float64}(A))
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$A_decomp"))
end

function _mvou_transition_from_eigen(eigA, t::Real)
    vals = exp.(-eigA.values .* Float64(t))
    return eigA.vectors * Diagonal(vals) * eigA.vectors'
end

function _mvou_transition_from_eigen!(out::AbstractMatrix{Float64}, eigA, t::Real)
    p = size(out, 1)
    size(out, 2) == p || throw(ArgumentError("transition output must be square"))
    fill!(out, 0.0)
    @inbounds for k in 1:p
        scale = exp(-eigA.values[k] * Float64(t))
        for j in 1:p
            right = eigA.vectors[j, k]
            for i in 1:p
                out[i, j] += eigA.vectors[i, k] * scale * right
            end
        end
    end
    return out
end

@inline function _mvou_logdet_chol(chol)
    s = 0.0
    @inbounds for i in axes(chol.L, 1)
        s += log(chol.L[i, i])
    end
    return 2.0 * s
end

@inline _mvou_cholesky_fast!(M::AbstractMatrix{Float64}) = cholesky!(Symmetric(M))


function _mvou_qinv_logdet(Q::AbstractMatrix{<:Real})
    p = size(Q, 1)
    cholQ = _mvou_cholesky_psd(Q)
    Qinv = Matrix{Float64}(I, p, p)
    ldiv!(Qinv, cholQ, Qinv)
    return Qinv, _mvou_logdet_chol(cholQ)
end

function _mvou_qinv_logdet!(Qinv::AbstractMatrix{Float64}, Q::AbstractMatrix{<:Real})
    p = size(Q, 1)
    size(Qinv) == (p, p) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    cholQ = _mvou_cholesky_psd(Q)
    fill!(Qinv, 0.0)
    @inbounds for i in 1:p
        Qinv[i, i] = 1.0
    end
    ldiv!(Qinv, cholQ, Qinv)
    return _mvou_logdet_chol(cholQ)
end

function _mvou_qinv_logdet!(
    Qinv::AbstractMatrix{Float64},
    Q::AbstractMatrix{<:Real},
    work::AbstractMatrix{Float64},
)
    p = size(Q, 1)
    size(Qinv) == (p, p) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    size(work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    copyto!(work, Q)
    try
        cholQ = _mvou_cholesky_fast!(work)
        fill!(Qinv, 0.0)
        @inbounds for i in 1:p
            Qinv[i, i] = 1.0
        end
        ldiv!(Qinv, cholQ, Qinv)
        return _mvou_logdet_chol(cholQ)
    catch err
        err isa PosDefException || rethrow()
        return _mvou_qinv_logdet!(Qinv, Q)
    end
end

function _mvou_qinv_logdet_small!(
    Qinv::AbstractMatrix{Float64},
    Q::AbstractMatrix{<:Real},
    work::AbstractMatrix{Float64},
)
    p = size(Q, 1)
    size(Qinv) == (p, p) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    size(work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    fill!(work, 0.0)
    logdet = 0.0
    @inbounds for j in 1:p
        s = Float64(Q[j, j])
        for k in 1:(j - 1)
            s -= work[j, k] * work[j, k]
        end
        if !(s > 0.0) || !isfinite(s)
            return _mvou_qinv_logdet!(Qinv, Q, work)
        end
        ljj = sqrt(s)
        work[j, j] = ljj
        logdet += 2.0 * log(ljj)
        for i in (j + 1):p
            v = Float64(Q[i, j])
            for k in 1:(j - 1)
                v -= work[i, k] * work[j, k]
            end
            work[i, j] = v / ljj
        end
    end

    fill!(Qinv, 0.0)
    @inbounds for col in 1:p
        for i in 1:p
            v = (i == col ? 1.0 : 0.0)
            for k in 1:(i - 1)
                v -= work[i, k] * Qinv[k, col]
            end
            Qinv[i, col] = v / work[i, i]
        end
        for i in p:-1:1
            v = Qinv[i, col]
            for k in (i + 1):p
                v -= work[k, i] * Qinv[k, col]
            end
            Qinv[i, col] = v / work[i, i]
        end
    end
    @inbounds for j in 1:p
        for i in (j + 1):p
            v = 0.5 * (Qinv[i, j] + Qinv[j, i])
            Qinv[i, j] = v
            Qinv[j, i] = v
        end
    end
    return logdet
end

function _mvou_cholesky_lower_logdet_small!(A::AbstractMatrix{Float64})
    p = size(A, 1)
    size(A, 2) == p || throw(ArgumentError("small Cholesky matrix must be square"))
    logdet = 0.0
    @inbounds for j in 1:p
        s = A[j, j]
        for k in 1:(j - 1)
            s -= A[j, k] * A[j, k]
        end
        s > 0.0 && isfinite(s) || throw(PosDefException(j))
        ljj = sqrt(s)
        A[j, j] = ljj
        logdet += 2.0 * log(ljj)
        for i in (j + 1):p
            v = A[i, j]
            for k in 1:(j - 1)
                v -= A[i, k] * A[j, k]
            end
            A[i, j] = v / ljj
        end
    end
    return logdet
end

function _mvou_cholesky_solve_matrix_small!(
    X::AbstractMatrix{Float64},
    L::AbstractMatrix{Float64},
    nrows::Integer,
    ncols::Integer,
)
    nrows = Int(nrows)
    ncols = Int(ncols)
    @inbounds for col in 1:ncols
        for i in 1:nrows
            v = X[i, col]
            for k in 1:(i - 1)
                v -= L[i, k] * X[k, col]
            end
            X[i, col] = v / L[i, i]
        end
        for i in nrows:-1:1
            v = X[i, col]
            for k in (i + 1):nrows
                v -= L[k, i] * X[k, col]
            end
            X[i, col] = v / L[i, i]
        end
    end
    return X
end

function _mvou_cholesky_solve_vector_small!(
    x::AbstractVector{Float64},
    L::AbstractMatrix{Float64},
    nrows::Integer,
)
    nrows = Int(nrows)
    @inbounds begin
        for i in 1:nrows
            v = x[i]
            for k in 1:(i - 1)
                v -= L[i, k] * x[k]
            end
            x[i] = v / L[i, i]
        end
        for i in nrows:-1:1
            v = x[i]
            for k in (i + 1):nrows
                v -= L[k, i] * x[k]
            end
            x[i] = v / L[i, i]
        end
    end
    return x
end

function _mvou_stationary_covariance(A::AbstractMatrix{<:Real}, Sigma::AbstractMatrix{<:Real})
    p = size(A, 1)
    size(A, 2) == p || throw(ArgumentError("A must be square"))
    size(Sigma, 1) == size(Sigma, 2) == p || throw(ArgumentError("Sigma dimensions must match A"))
    return _mvou_stationary_covariance(_mvou_transition_eigen(A), Sigma)
end

function _mvou_stationary_covariance(
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    A_decomp::Symbol,
)
    p = size(A, 1)
    size(A, 2) == p || throw(ArgumentError("A must be square"))
    size(Sigma, 1) == size(Sigma, 2) == p || throw(ArgumentError("Sigma dimensions must match A"))
    if A_decomp === :cholesky
        return _mvou_stationary_covariance(_mvou_transition_eigen(A), Sigma)
    elseif A_decomp === :schur
        K = kron(I(p), Matrix{Float64}(A)) + kron(Matrix{Float64}(A), I(p))
        vecS = K \ vec(Matrix{Float64}(Sigma))
        S = reshape(vecS, p, p)
        return (S + S') / 2
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$A_decomp"))
end

function _mvou_transition_from_schur(F::LinearAlgebra.Schur, t::Real)
    Texp = exp(-Matrix{Float64}(F.T) * Float64(t))
    Z = Matrix{Float64}(F.Z)
    return Z * Texp * transpose(Z)
end

function _mvou_transition_from_schur!(out::AbstractMatrix{Float64}, F::LinearAlgebra.Schur, t::Real)
    p = size(out, 1)
    size(out, 2) == p || throw(ArgumentError("transition output must be square"))
    Z = Matrix{Float64}(F.Z)
    Texp = exp(-Matrix{Float64}(F.T) * Float64(t))
    tmp = Z * Texp
    out .= tmp * transpose(Z)
    return out
end

@inline function _mvou_transition_from_factor(factor, A_decomp::Symbol, t::Real)
    if A_decomp === :cholesky
        return _mvou_transition_from_eigen(factor, t)
    elseif A_decomp === :schur
        return _mvou_transition_from_schur(factor, t)
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$A_decomp"))
end

@inline function _mvou_transition_from_factor!(out::AbstractMatrix{Float64}, factor, A_decomp::Symbol, t::Real)
    if A_decomp === :cholesky
        return _mvou_transition_from_eigen!(out, factor, t)
    elseif A_decomp === :schur
        return _mvou_transition_from_schur!(out, factor, t)
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$A_decomp"))
end

function _mvou_stationary_covariance(eigA, Sigma::AbstractMatrix{<:Real})
    p = length(eigA.values)
    size(Sigma, 1) == size(Sigma, 2) == p || throw(ArgumentError("Sigma dimensions must match A"))
    rotated = transpose(eigA.vectors) * Matrix{Float64}(Sigma) * eigA.vectors
    @inbounds for j in 1:p, i in 1:p
        rotated[i, j] /= eigA.values[i] + eigA.values[j]
    end
    S = eigA.vectors * rotated * transpose(eigA.vectors)
    return (S + S') / 2
end

function _mvou_root_covariance(precalc::MVOUPrecalc, bundle::MVOUParameterBundle)
    p = isempty(bundle.A) ? size(bundle.A_regimes, 1) : size(bundle.A, 1)
    precalc.root_cov_mode === :fixed && return zeros(Float64, p, p)
    precalc.root_cov_mode === :stationary ||
        throw(ArgumentError("Unsupported multivariate OU root_cov_mode=$(precalc.root_cov_mode)"))

    r = precalc.root_regime
    A = isempty(bundle.A_regimes) ? bundle.A : @view(bundle.A_regimes[:, :, r])
    Sigma = isempty(bundle.Sigma_regimes) ? bundle.Sigma : @view(bundle.Sigma_regimes[:, :, r])
    return _mvou_stationary_covariance(A, Sigma, precalc.A_decomp)
end



function _mvou_branch_cache(
    precalc::MVOUPrecalc,
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
)
    p = size(A, 1)
    Phi = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Q = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Qinv = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    logdet_Q = Vector{Float64}(undef, length(precalc.branch_lengths))
    work = zeros(Float64, p, p)
    return _mvou_branch_cache!(Phi, Q, Qinv, logdet_Q, precalc, A, Sigma; chol_work = work)
end

function _mvou_branch_cache!(
    Phi::Array{Float64, 3},
    Q::Array{Float64, 3},
    Qinv::Array{Float64, 3},
    logdet_Q::AbstractVector{Float64},
    precalc::MVOUPrecalc,
    A::AbstractMatrix{<:Real},
    Sigma::AbstractMatrix{<:Real},
    ;
    chol_work::Union{Nothing, AbstractMatrix{Float64}} = nothing,
    reuse_phi::Bool = false,
)
    p = size(A, 1)
    size(Phi) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Phi workspace has incompatible dimensions"))
    size(Q) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Q workspace has incompatible dimensions"))
    size(Qinv) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    length(logdet_Q) == length(precalc.branch_lengths) || throw(ArgumentError("logdet_Q workspace has incompatible dimensions"))
    factor = reuse_phi ? nothing : _mvou_transition_factor(A, precalc.A_decomp)
    Sstat = _mvou_stationary_covariance(A, Sigma, precalc.A_decomp)
    work = chol_work === nothing ? zeros(Float64, p, p) : chol_work
    size(work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    for edge in eachindex(precalc.branch_lengths)
        t = precalc.branch_lengths[edge]
        Phi_edge = @view Phi[:, :, edge]
        Q_edge = @view Q[:, :, edge]
        tmp = @view Qinv[:, :, edge]
        if !reuse_phi
            _mvou_transition_from_factor!(Phi_edge, factor, precalc.A_decomp, t)
        end
        mul!(tmp, Phi_edge, Sstat)
        mul!(Q_edge, tmp, transpose(Phi_edge), -1.0, 0.0)
        @inbounds for j in 1:p
            Q_edge[j, j] += Sstat[j, j]
            for i in (j + 1):p
                v = 0.5 * (Sstat[i, j] + Q_edge[i, j] + Sstat[j, i] + Q_edge[j, i])
                Q_edge[i, j] = v
                Q_edge[j, i] = v
            end
        end
        logdet_Q[edge] = _mvou_qinv_logdet_small!(@view(Qinv[:, :, edge]), Q_edge, work)
    end
    return (Phi = Phi, Q = Q, Qinv = Qinv, logdet_Q = logdet_Q, Sstat = Sstat)
end

function _mvoumv_branch_cache(
    precalc::MVOUPrecalc,
    A::AbstractMatrix{<:Real},
    Sigma_regimes::Array{Float64, 3},
)
    p = size(A, 1)
    Phi = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Q = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Qinv = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    logdet_Q = Vector{Float64}(undef, length(precalc.branch_lengths))
    work = zeros(Float64, p, p)
    return _mvoumv_branch_cache!(Phi, Q, Qinv, logdet_Q, precalc, A, Sigma_regimes; chol_work = work)
end

function _mvoumv_branch_cache!(
    Phi::Array{Float64, 3},
    Q::Array{Float64, 3},
    Qinv::Array{Float64, 3},
    logdet_Q::AbstractVector{Float64},
    precalc::MVOUPrecalc,
    A::AbstractMatrix{<:Real},
    Sigma_regimes::Array{Float64, 3},
    ;
    chol_work::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)
    p = size(A, 1)
    size(Phi) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Phi workspace has incompatible dimensions"))
    size(Q) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Q workspace has incompatible dimensions"))
    size(Qinv) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    length(logdet_Q) == length(precalc.branch_lengths) || throw(ArgumentError("logdet_Q workspace has incompatible dimensions"))
    size(Sigma_regimes, 1) == p == size(Sigma_regimes, 2) || throw(ArgumentError("Sigma_regimes dimensions must match A"))
    size(Sigma_regimes, 3) == precalc.nregimes || throw(ArgumentError("Sigma_regimes regime count must match precalc"))
    factor = _mvou_transition_factor(A, precalc.A_decomp)
    stationary = [_mvou_stationary_covariance(A, Sigma_regimes[:, :, r], precalc.A_decomp) for r in 1:precalc.nregimes]
    work = chol_work === nothing ? zeros(Float64, p, p) : chol_work
    size(work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    Phi_seg = zeros(Float64, p, p)
    Phi_total = Matrix{Float64}(I, p, p)
    Q_total = zeros(Float64, p, p)
    Q_seg = zeros(Float64, p, p)
    tmp1 = zeros(Float64, p, p)
    tmp2 = zeros(Float64, p, p)

    for edge in eachindex(precalc.branch_lengths)
        segs = precalc.edge_segments[edge]
        isempty(segs) && throw(ArgumentError("edge_segments[$edge] is empty"))
        fill!(Phi_total, 0.0)
        fill!(Q_total, 0.0)
        @inbounds for i in 1:p
            Phi_total[i, i] = 1.0
        end
        for seg in segs
            _mvou_transition_from_factor!(Phi_seg, factor, precalc.A_decomp, seg.length)
            Sstat = stationary[Int(seg.state)]
            mul!(tmp1, Phi_seg, Sstat)
            mul!(Q_seg, tmp1, transpose(Phi_seg))
            @inbounds for j in 1:p
                Q_seg[j, j] = Sstat[j, j] - Q_seg[j, j]
                for i in (j + 1):p
                    v = 0.5 * (Sstat[i, j] - Q_seg[i, j] + Sstat[j, i] - Q_seg[j, i])
                    Q_seg[i, j] = v
                    Q_seg[j, i] = v
                end
            end
            mul!(tmp1, Phi_seg, Q_total)
            mul!(tmp2, tmp1, transpose(Phi_seg))
            @. Q_total = tmp2 + Q_seg
            mul!(tmp1, Phi_seg, Phi_total)
            copyto!(Phi_total, tmp1)
        end
        Phi[:, :, edge] .= Phi_total
        Q_edge = @view Q[:, :, edge]
        copyto!(Q_edge, Q_total)
        @inbounds for j in 1:p
            for i in (j + 1):p
                v = 0.5 * (Q_edge[i, j] + Q_edge[j, i])
                Q_edge[i, j] = v
                Q_edge[j, i] = v
            end
        end
        logdet_Q[edge] = _mvou_qinv_logdet_small!(@view(Qinv[:, :, edge]), @view(Q[:, :, edge]), work)
    end
    return (Phi = Phi, Q = Q, Qinv = Qinv, logdet_Q = logdet_Q)
end

function _mvouma_branch_cache(
    precalc::MVOUPrecalc,
    A_regimes::Array{Float64, 3},
    Sigma::AbstractMatrix{<:Real},
)
    p = size(Sigma, 1)
    Phi = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Q = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Qinv = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    logdet_Q = Vector{Float64}(undef, length(precalc.branch_lengths))
    work = zeros(Float64, p, p)
    return _mvouma_branch_cache!(Phi, Q, Qinv, logdet_Q, precalc, A_regimes, Sigma; chol_work = work)
end

function _mvouma_branch_cache!(
    Phi::Array{Float64, 3},
    Q::Array{Float64, 3},
    Qinv::Array{Float64, 3},
    logdet_Q::AbstractVector{Float64},
    precalc::MVOUPrecalc,
    A_regimes::Array{Float64, 3},
    Sigma::AbstractMatrix{<:Real},
    ;
    chol_work::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)
    p = size(Sigma, 1)
    size(Phi) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Phi workspace has incompatible dimensions"))
    size(Q) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Q workspace has incompatible dimensions"))
    size(Qinv) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    length(logdet_Q) == length(precalc.branch_lengths) || throw(ArgumentError("logdet_Q workspace has incompatible dimensions"))
    size(Sigma, 2) == p || throw(ArgumentError("Sigma must be square"))
    size(A_regimes, 1) == size(A_regimes, 2) == p || throw(ArgumentError("A_regimes dimensions must match Sigma"))
    size(A_regimes, 3) == precalc.nregimes || throw(ArgumentError("A_regimes regime count must match precalc"))
    factors = [_mvou_transition_factor(A_regimes[:, :, r], precalc.A_decomp) for r in 1:precalc.nregimes]
    stationary = [_mvou_stationary_covariance(A_regimes[:, :, r], Sigma, precalc.A_decomp) for r in 1:precalc.nregimes]
    work = chol_work === nothing ? zeros(Float64, p, p) : chol_work
    size(work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    Phi_seg = zeros(Float64, p, p)
    Phi_total = Matrix{Float64}(I, p, p)
    Q_total = zeros(Float64, p, p)
    Q_seg = zeros(Float64, p, p)
    tmp1 = zeros(Float64, p, p)
    tmp2 = zeros(Float64, p, p)

    for edge in eachindex(precalc.branch_lengths)
        segs = precalc.edge_segments[edge]
        isempty(segs) && throw(ArgumentError("edge_segments[$edge] is empty"))
        fill!(Phi_total, 0.0)
        fill!(Q_total, 0.0)
        @inbounds for i in 1:p
            Phi_total[i, i] = 1.0
        end
        for seg in segs
            r = Int(seg.state)
            _mvou_transition_from_factor!(Phi_seg, factors[r], precalc.A_decomp, seg.length)
            Sstat = stationary[r]
            mul!(tmp1, Phi_seg, Sstat)
            mul!(Q_seg, tmp1, transpose(Phi_seg))
            @inbounds for j in 1:p
                Q_seg[j, j] = Sstat[j, j] - Q_seg[j, j]
                for i in (j + 1):p
                    v = 0.5 * (Sstat[i, j] - Q_seg[i, j] + Sstat[j, i] - Q_seg[j, i])
                    Q_seg[i, j] = v
                    Q_seg[j, i] = v
                end
            end
            mul!(tmp1, Phi_seg, Q_total)
            mul!(tmp2, tmp1, transpose(Phi_seg))
            @. Q_total = tmp2 + Q_seg
            mul!(tmp1, Phi_seg, Phi_total)
            copyto!(Phi_total, tmp1)
        end
        Phi[:, :, edge] .= Phi_total
        Q_edge = @view Q[:, :, edge]
        copyto!(Q_edge, Q_total)
        @inbounds for j in 1:p
            for i in (j + 1):p
                v = 0.5 * (Q_edge[i, j] + Q_edge[j, i])
                Q_edge[i, j] = v
                Q_edge[j, i] = v
            end
        end
        logdet_Q[edge] = _mvou_qinv_logdet_small!(@view(Qinv[:, :, edge]), Q_edge, work)
    end
    return (Phi = Phi, Q = Q, Qinv = Qinv, logdet_Q = logdet_Q)
end

function _mvoumva_branch_cache(
    precalc::MVOUPrecalc,
    A_regimes::Array{Float64, 3},
    Sigma_regimes::Array{Float64, 3},
)
    p = size(A_regimes, 1)
    Phi = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Q = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    Qinv = Array{Float64, 3}(undef, p, p, length(precalc.branch_lengths))
    logdet_Q = Vector{Float64}(undef, length(precalc.branch_lengths))
    work = zeros(Float64, p, p)
    return _mvoumva_branch_cache!(Phi, Q, Qinv, logdet_Q, precalc, A_regimes, Sigma_regimes; chol_work = work)
end

function _mvoumva_branch_cache!(
    Phi::Array{Float64, 3},
    Q::Array{Float64, 3},
    Qinv::Array{Float64, 3},
    logdet_Q::AbstractVector{Float64},
    precalc::MVOUPrecalc,
    A_regimes::Array{Float64, 3},
    Sigma_regimes::Array{Float64, 3},
    ;
    chol_work::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)
    p = size(A_regimes, 1)
    size(Phi) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Phi workspace has incompatible dimensions"))
    size(Q) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Q workspace has incompatible dimensions"))
    size(Qinv) == (p, p, length(precalc.branch_lengths)) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    length(logdet_Q) == length(precalc.branch_lengths) || throw(ArgumentError("logdet_Q workspace has incompatible dimensions"))
    size(A_regimes, 2) == p || throw(ArgumentError("A_regimes must be square per regime"))
    size(Sigma_regimes, 1) == size(Sigma_regimes, 2) == p || throw(ArgumentError("Sigma_regimes dimensions must match A_regimes"))
    size(A_regimes, 3) == size(Sigma_regimes, 3) == precalc.nregimes || throw(ArgumentError("Regime counts must match precalc"))
    factors = [_mvou_transition_factor(A_regimes[:, :, r], precalc.A_decomp) for r in 1:precalc.nregimes]
    stationary = [_mvou_stationary_covariance(A_regimes[:, :, r], Sigma_regimes[:, :, r], precalc.A_decomp) for r in 1:precalc.nregimes]
    work = chol_work === nothing ? zeros(Float64, p, p) : chol_work
    size(work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    Phi_seg = zeros(Float64, p, p)
    Phi_total = Matrix{Float64}(I, p, p)
    Q_total = zeros(Float64, p, p)
    Q_seg = zeros(Float64, p, p)
    tmp1 = zeros(Float64, p, p)
    tmp2 = zeros(Float64, p, p)

    for edge in eachindex(precalc.branch_lengths)
        segs = precalc.edge_segments[edge]
        isempty(segs) && throw(ArgumentError("edge_segments[$edge] is empty"))
        fill!(Phi_total, 0.0)
        fill!(Q_total, 0.0)
        @inbounds for i in 1:p
            Phi_total[i, i] = 1.0
        end
        for seg in segs
            r = Int(seg.state)
            _mvou_transition_from_factor!(Phi_seg, factors[r], precalc.A_decomp, seg.length)
            Sstat = stationary[r]
            mul!(tmp1, Phi_seg, Sstat)
            mul!(Q_seg, tmp1, transpose(Phi_seg))
            @inbounds for j in 1:p
                Q_seg[j, j] = Sstat[j, j] - Q_seg[j, j]
                for i in (j + 1):p
                    v = 0.5 * (Sstat[i, j] - Q_seg[i, j] + Sstat[j, i] - Q_seg[j, i])
                    Q_seg[i, j] = v
                    Q_seg[j, i] = v
                end
            end
            mul!(tmp1, Phi_seg, Q_total)
            mul!(tmp2, tmp1, transpose(Phi_seg))
            @. Q_total = tmp2 + Q_seg
            mul!(tmp1, Phi_seg, Phi_total)
            copyto!(Phi_total, tmp1)
        end
        Phi[:, :, edge] .= Phi_total
        Q_edge = @view Q[:, :, edge]
        copyto!(Q_edge, Q_total)
        @inbounds for j in 1:p
            for i in (j + 1):p
                v = 0.5 * (Q_edge[i, j] + Q_edge[j, i])
                Q_edge[i, j] = v
                Q_edge[j, i] = v
            end
        end
        logdet_Q[edge] = _mvou_qinv_logdet_small!(@view(Qinv[:, :, edge]), Q_edge, work)
    end
    return (Phi = Phi, Q = Q, Qinv = Qinv, logdet_Q = logdet_Q)
end



function _mvou_tip_path_means(
    tree::CompactTree,
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    row_standardize::Bool = true,
)
    p = size(bundle.A, 1)
    theta_matrix = reshape(bundle.theta, p, :)
    size(theta_matrix, 2) == precalc.nregimes || throw(ArgumentError("theta regime count must match precalc regime count"))
    if row_standardize
        designs = _mvou_node_design_matrices(tree, bundle.A, precalc.edge_segments, precalc.nregimes, precalc.A_decomp)
        return _mvou_tip_means_from_design(tree, designs, theta_matrix)
    end
    means = _mvou_tip_path_means_from_matrix(tree, bundle.A, theta_matrix, precalc.edge_segments)
    return means
end

function _mvou_node_design_matrices(
    tree::CompactTree,
    A::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    nregimes::Integer,
    A_decomp::Symbol = :cholesky,
    ;
    root_regime::Integer = 0,
)
    p = size(A, 1)
    designs = [zeros(Float64, p, p * nregimes) for _ in 1:tree.nnodes]
    factor = _mvou_transition_factor(A, A_decomp)
    transition_cache = Dict{Float64, Matrix{Float64}}(0.0 => Matrix{Float64}(I, p, p))
    identity = Matrix{Float64}(I, p, p)
    if root_regime > 0
        cols = ((root_regime - 1) * p + 1):(root_regime * p)
        designs[Int(tree.root)][:, cols] .= identity
    end
    getPhi(t) = get!(transition_cache, t) do
        _mvou_transition_from_factor(factor, A_decomp, t)
    end
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            W = copy(designs[node])
            for seg in edge_segments[Int(edge)]
                r = Int(seg.state)
                Phi = getPhi(seg.length)
                W = Phi * W
                cols = ((r - 1) * p + 1):(r * p)
                W[:, cols] .+= identity - Phi
            end
            designs[child] .= W
        end
    end
    return designs
end

function _mvou_node_design_matrices!(
    designs::Vector{Matrix{Float64}},
    work::Matrix{Float64},
    transition_work::Matrix{Float64},
    tree::CompactTree,
    A::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    nregimes::Integer,
    A_decomp::Symbol = :cholesky,
    ;
    root_regime::Integer = 0,
)
    p = size(A, 1)
    length(designs) == tree.nnodes || throw(ArgumentError("design workspace node count does not match tree"))
    size(work) == (p, p * nregimes) || throw(ArgumentError("design work matrix has incompatible dimensions"))
    size(transition_work) == (p, p) || throw(ArgumentError("transition work matrix has incompatible dimensions"))
    factor = _mvou_transition_factor(A, A_decomp)
    for W in designs
        fill!(W, 0.0)
    end
    ncols = p * Int(nregimes)
    if root_regime > 0
        cols = ((root_regime - 1) * p + 1):(root_regime * p)
        designs[Int(tree.root)][:, cols] .= Matrix{Float64}(I, p, p)
    end
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            work .= designs[node]
            for seg in edge_segments[Int(edge)]
                r = Int(seg.state)
                _mvou_transition_from_factor!(transition_work, factor, A_decomp, seg.length)
                dest = designs[child]
                @inbounds for col in 1:ncols
                    for i in 1:p
                        acc = 0.0
                        for k in 1:p
                            acc += transition_work[i, k] * work[k, col]
                        end
                        dest[i, col] = acc
                    end
                end
                offset = (r - 1) * p
                @inbounds for jj in 1:p, ii in 1:p
                    dest[ii, offset + jj] += (ii == jj ? 1.0 : 0.0) - transition_work[ii, jj]
                end
                work .= dest
            end
        end
    end
    return designs
end

function _mvou_tip_means_from_design(
    tree::CompactTree,
    designs::Vector{Matrix{Float64}},
    theta_matrix::AbstractMatrix{<:Real},
)
    p, nregimes = size(theta_matrix)
    theta_vec = vec(Matrix{Float64}(theta_matrix))
    means = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        W = designs[tip]
        for j in 1:p
            denom = sum(@view W[j, :])
            means[i, j] = abs(denom) <= 1e-12 ? 0.0 : dot(@view(W[j, :]), theta_vec) / denom
        end
    end
    return means
end

function _mvou_node_means_from_design(
    tree::CompactTree,
    designs::Vector{Matrix{Float64}},
    theta_matrix::AbstractMatrix{<:Real},
)
    p, nregimes = size(theta_matrix)
    theta_vec = vec(Matrix{Float64}(theta_matrix))
    means = [zeros(Float64, p) for _ in 1:tree.nnodes]
    for node in 1:tree.nnodes
        W = designs[node]
        for j in 1:p
            denom = sum(@view W[j, :])
            means[node][j] = abs(denom) <= 1e-12 ? 0.0 : dot(@view(W[j, :]), theta_vec) / denom
        end
    end
    return means
end

function _mvou_row_standardize_designs!(designs::Vector{Matrix{Float64}})
    for W in designs
        for i in axes(W, 1)
            denom = sum(@view W[i, :])
            abs(denom) <= 1e-12 && continue
            @views W[i, :] ./= denom
        end
    end
    return designs
end

function _mvou_tip_path_means_from_matrix(
    tree::CompactTree,
    A::AbstractMatrix{<:Real},
    theta_matrix::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
)
    p = size(theta_matrix, 1)
    means = Matrix{Float64}(undef, tree.ntips, p)
    node_means = _mvou_node_path_means_zero(tree, A, theta_matrix, edge_segments)
    for (i, tip) in enumerate(tree.tip_ids)
        means[i, :] .= node_means[tip]
    end
    return means
end

function _mvou_edge_regime_mean(
    start_mean::AbstractVector{<:Real},
    A::AbstractMatrix{<:Real},
    theta_matrix::AbstractMatrix{<:Real},
    segs::Vector{SimmapSegment},
)
    mean_vec = Vector{Float64}(start_mean)
    for seg in segs
        theta_seg = theta_matrix[:, Int(seg.state)]
        Phi_seg = _mvou_transition(A, seg.length)
        mean_vec = theta_seg + Phi_seg * (mean_vec - theta_seg)
    end
    return mean_vec
end

function _mvou_node_path_means(
    tree::CompactTree,
    bundle::MVOUParameterBundle,
    edge_segments::Vector{Vector{SimmapSegment}},
    root_regime::Integer,
)
    p = size(bundle.A, 1)
    theta_matrix = reshape(bundle.theta, p, :)
    means = [zeros(Float64, p) for _ in 1:tree.nnodes]
    means[Int(tree.root)] .= theta_matrix[:, root_regime]
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            means[child] .= _mvou_edge_regime_mean(means[node], bundle.A, theta_matrix, edge_segments[Int(edge)])
        end
    end
    return means
end

function _mvou_node_path_means_zero(
    tree::CompactTree,
    A::AbstractMatrix{<:Real},
    theta_matrix::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
)
    p = size(theta_matrix, 1)
    means = [zeros(Float64, p) for _ in 1:tree.nnodes]
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            means[child] .= _mvou_edge_regime_mean(means[node], A, theta_matrix, edge_segments[Int(edge)])
        end
    end
    return means
end

function _mvouma_tip_path_means(
    tree::CompactTree,
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    row_standardize::Bool = true,
)
    p = size(bundle.A, 1)
    theta_matrix = reshape(bundle.theta, p, :)
    size(theta_matrix, 2) == precalc.nregimes || throw(ArgumentError("theta regime count must match precalc regime count"))
    size(bundle.A_regimes, 3) == precalc.nregimes || throw(ArgumentError("A_regimes regime count must match precalc"))
    if row_standardize
        designs = _mvouma_node_design_matrices(tree, bundle.A_regimes, precalc.edge_segments, precalc.nregimes, precalc.A_decomp)
        return _mvou_tip_means_from_design(tree, designs, theta_matrix)
    end
    means = _mvouma_tip_path_means_from_matrix(tree, bundle.A_regimes, theta_matrix, precalc.edge_segments)
    return means
end

function _mvouma_node_design_matrices(
    tree::CompactTree,
    A_regimes::Array{Float64, 3},
    edge_segments::Vector{Vector{SimmapSegment}},
    nregimes::Integer,
    A_decomp::Symbol = :cholesky,
    ;
    root_regime::Integer = 0,
)
    p = size(A_regimes, 1)
    designs = [zeros(Float64, p, p * nregimes) for _ in 1:tree.nnodes]
    factors = [_mvou_transition_factor(A_regimes[:, :, r], A_decomp) for r in 1:nregimes]
    transition_cache = Dict{Tuple{Int, Float64}, Matrix{Float64}}()
    identity = Matrix{Float64}(I, p, p)
    if root_regime > 0
        cols = ((root_regime - 1) * p + 1):(root_regime * p)
        designs[Int(tree.root)][:, cols] .= identity
    end
    getPhi(r, t) = get!(transition_cache, (r, t)) do
        _mvou_transition_from_factor(factors[r], A_decomp, t)
    end
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            W = copy(designs[node])
            for seg in edge_segments[Int(edge)]
                r = Int(seg.state)
                Phi = getPhi(r, seg.length)
                W = Phi * W
                cols = ((r - 1) * p + 1):(r * p)
                W[:, cols] .+= identity - Phi
            end
            designs[child] .= W
        end
    end
    return designs
end

function _mvouma_node_design_matrices!(
    designs::Vector{Matrix{Float64}},
    work::AbstractMatrix{Float64},
    transition_work::AbstractMatrix{Float64},
    tree::CompactTree,
    A_regimes::Array{Float64, 3},
    edge_segments::Vector{Vector{SimmapSegment}},
    nregimes::Integer,
    A_decomp::Symbol = :cholesky,
    ;
    root_regime::Integer = 0,
)
    p = size(A_regimes, 1)
    q = p * Int(nregimes)
    length(designs) == tree.nnodes || throw(ArgumentError("design workspace node count does not match tree"))
    size(work) == (p, q) || throw(ArgumentError("design work matrix has incompatible dimensions"))
    size(transition_work) == (p, p) || throw(ArgumentError("transition work matrix has incompatible dimensions"))
    factors = [_mvou_transition_factor(A_regimes[:, :, r], A_decomp) for r in 1:nregimes]
    for W in designs
        size(W) == (p, q) || throw(ArgumentError("design workspace has incompatible dimensions"))
        fill!(W, 0.0)
    end
    if root_regime > 0
        cols = ((root_regime - 1) * p + 1):(root_regime * p)
        designs[Int(tree.root)][:, cols] .= Matrix{Float64}(I, p, p)
    end
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            copyto!(designs[child], designs[node])
            for seg in edge_segments[Int(edge)]
                r = Int(seg.state)
                _mvou_transition_from_factor!(transition_work, factors[r], A_decomp, seg.length)
                mul!(work, transition_work, designs[child])
                copyto!(designs[child], work)
                offset = (r - 1) * p
                @inbounds for j in 1:p, i in 1:p
                    designs[child][i, offset + j] += (i == j ? 1.0 : 0.0) - transition_work[i, j]
                end
            end
        end
    end
    return designs
end

function _mvouma_tip_path_means_from_matrix(
    tree::CompactTree,
    A_regimes::Array{Float64, 3},
    theta_matrix::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
)
    p = size(theta_matrix, 1)
    means = Matrix{Float64}(undef, tree.ntips, p)
    node_means = _mvouma_node_path_means_zero(tree, A_regimes, theta_matrix, edge_segments)
    for (i, tip) in enumerate(tree.tip_ids)
        means[i, :] .= node_means[tip]
    end
    return means
end

function _mvouma_edge_regime_mean(
    start_mean::AbstractVector{<:Real},
    A_regimes::Array{Float64, 3},
    theta_matrix::AbstractMatrix{<:Real},
    segs::Vector{SimmapSegment},
)
    mean_vec = Vector{Float64}(start_mean)
    for seg in segs
        r = Int(seg.state)
        theta_seg = theta_matrix[:, r]
        Phi_seg = _mvou_transition(A_regimes[:, :, r], seg.length)
        mean_vec = theta_seg + Phi_seg * (mean_vec - theta_seg)
    end
    return mean_vec
end

function _mvouma_node_path_means(
    tree::CompactTree,
    bundle::MVOUParameterBundle,
    edge_segments::Vector{Vector{SimmapSegment}},
    root_regime::Integer,
)
    p = size(bundle.A, 1)
    theta_matrix = reshape(bundle.theta, p, :)
    means = [zeros(Float64, p) for _ in 1:tree.nnodes]
    means[Int(tree.root)] .= theta_matrix[:, root_regime]
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            means[child] .= _mvouma_edge_regime_mean(means[node], bundle.A_regimes, theta_matrix, edge_segments[Int(edge)])
        end
    end
    return means
end

function _mvouma_node_path_means_zero(
    tree::CompactTree,
    A_regimes::Array{Float64, 3},
    theta_matrix::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
)
    p = size(theta_matrix, 1)
    means = [zeros(Float64, p) for _ in 1:tree.nnodes]
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            edge == 0 && continue
            child = Int(tree.child_of_edge[edge])
            means[child] .= _mvouma_edge_regime_mean(means[node], A_regimes, theta_matrix, edge_segments[Int(edge)])
        end
    end
    return means
end
