# Preallocated evaluation workspace for the mvOUBM likelihood.
#
# The tree topology, branch lengths, and simmap segment layout are fixed
# during a fit, so the unique branch times, mean-offset time set, Phi/Q
# caches, and message buffers are built once and reused across objective
# evaluations.  Only eig(H), the cache fill, and the offset matvecs run
# per evaluation.

mutable struct MVOUBMWorkspace
    p::Int
    kY::Int
    kX::Int
    unique_times::Vector{Float64}
    edge_time_idx::Vector{Int}
    offset_times::Vector{Float64}
    edge_offset_idx::Vector{Vector{NTuple{2, Int}}}
    edge_offset_regime::Vector{Vector{Int}}
    Phi::Array{Float64, 3}
    Phi_off::Array{Float64, 3}
    Q::Array{Float64, 3}
    edge_mu::Vector{Vector{Float64}}
    root_mean::Vector{Float64}
    root_cov::Matrix{Float64}
    scratch::Matrix{ComplexF64}
    expbuf::Vector{ComplexF64}
    cbuf::Matrix{ComplexF64}
    cbuf2::Matrix{ComplexF64}
    scratchQ::Matrix{Float64}
    Qinv::Array{Float64, 3}
    logdet_Q::Vector{Float64}
    QPhi::Array{Float64, 3}
    data_t::Matrix{Float64}
    tip_nan::Vector{Bool}
    tp1::Matrix{Float64}
    tp2::Matrix{Float64}
    tp3::Matrix{Float64}
    tp4::Matrix{Float64}
    tS::Matrix{Float64}
    tv1::Vector{Float64}
    tv2::Vector{Float64}
    tip_index::Vector{Int}
    ybuf::Vector{Float64}
    hbuf::Vector{Float64}
    seg_mu::Vector{Float64}
    precision::Vector{Matrix{Float64}}
    linear::Vector{Vector{Float64}}
    logconst::Vector{Float64}
end

function _mvoubm_workspace(tree::CompactTree, precalc::MVOUPrecalc, kY::Int, kX::Int)
    p = kY + kX
    nedges = tree.nedges
    bl = Float64.(precalc.branch_lengths)
    segments = _mvoubm_segments(tree, precalc)
    uts = sort!(unique(bl))
    tmap = Dict{Float64, Int}(t => i for (i, t) in enumerate(uts))
    edge_time_idx = [tmap[bl[e]] for e in 1:nedges]

    raw_off = Float64[]
    for e in 1:nedges
        T = bl[e]
        cum = 0.0
        for seg in segments[e]
            len = Float64(seg.length)
            len <= 0 && continue
            push!(raw_off, T - cum - len, len)
            cum += len
        end
    end
    offset_times = sort!(unique!(raw_off))
    omap = Dict{Float64, Int}(t => i for (i, t) in enumerate(offset_times))

    nregimes = precalc.nregimes
    edge_offset_idx = Vector{Vector{NTuple{2, Int}}}(undef, nedges)
    edge_offset_regime = Vector{Vector{Int}}(undef, nedges)
    for e in 1:nedges
        T = bl[e]
        cum = 0.0
        idxs = NTuple{2, Int}[]
        regs = Int[]
        for seg in segments[e]
            len = Float64(seg.length)
            len <= 0 && continue
            push!(idxs, (omap[T - cum - len], omap[len]))
            push!(regs, min(max(Int(seg.state), 1), nregimes))
            cum += len
        end
        edge_offset_idx[e] = idxs
        edge_offset_regime[e] = regs
    end

    nu = length(uts)
    no = length(offset_times)
    tip_index = zeros(Int, tree.nnodes)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_index[Int(tip)] = i
    end
    MVOUBMWorkspace(
        p, kY, kX, uts, edge_time_idx, offset_times,
        edge_offset_idx, edge_offset_regime,
        zeros(Float64, p, p, nu), zeros(Float64, p, p, no), zeros(Float64, p, p, nu),
        [zeros(Float64, p) for _ in 1:nedges],
        zeros(Float64, p), zeros(Float64, p, p), zeros(ComplexF64, p, p),
        zeros(ComplexF64, p), zeros(ComplexF64, p, p), zeros(ComplexF64, p, p),
        zeros(Float64, p, p),
        zeros(Float64, p, p, nu), zeros(Float64, nu),
        zeros(Float64, p, p, nu), zeros(Float64, p, tree.ntips), zeros(Bool, tree.ntips),
        zeros(Float64, p, p), zeros(Float64, p, p), zeros(Float64, p, p),
        zeros(Float64, p, p),
        zeros(Float64, p, p), zeros(Float64, p), zeros(Float64, p),
        tip_index,
        zeros(Float64, p), zeros(Float64, p), zeros(Float64, p),
        [zeros(Float64, p, p) for _ in 1:tree.nnodes],
        [zeros(Float64, p) for _ in 1:tree.nnodes],
        zeros(Float64, tree.nnodes),
    )
end

function _mvoubm_segments(tree::CompactTree, precalc::MVOUPrecalc)
    secs = precalc.edge_segments
    if isempty(secs) || all(isempty, secs)
        return [[SimmapSegment(state = Int32(precalc.root_regime), length = Float64(precalc.branch_lengths[e]))]
                for e in 1:tree.nedges]
    end
    return secs
end

function _mvoubm_fill_caches!(ws::MVOUBMWorkspace, eigH, VinvH, Mrot)
    for i in eachindex(ws.unique_times)
        _mvoubm_Phi_W!(
            (@view ws.Phi[:, :, i]), (@view ws.Q[:, :, i]),
            ws.unique_times[i], eigH, VinvH, Mrot, ws.scratch,
            ws.expbuf, ws.cbuf, ws.cbuf2,
        )
    end
    for i in eachindex(ws.offset_times)
        _mvoubm_Phi!((@view ws.Phi_off[:, :, i]), ws.offset_times[i], eigH, VinvH, ws.expbuf, ws.cbuf, ws.cbuf2)
    end
    for i in eachindex(ws.unique_times)
        ws.logdet_Q[i] = _mvou_qinv_logdet_small!(
            (@view ws.Qinv[:, :, i]), (@view ws.Q[:, :, i]), ws.scratchQ,
        )
        mul!((@view ws.QPhi[:, :, i]), (@view ws.Qinv[:, :, i]), (@view ws.Phi[:, :, i]))
    end
    return ws
end

function _mvoubm_stage_data!(ws::MVOUBMWorkspace, data::AbstractMatrix{Float64})
    copyto!(ws.data_t, PermutedDimsArray(data, (2, 1)))
    for row in 1:size(ws.data_t, 2)
        ws.tip_nan[row] = any(isnan, @view ws.data_t[:, row])
    end
    return ws
end

function _mvoubm_profile!(tree::CompactTree, data::AbstractMatrix{Float64}, ws::MVOUBMWorkspace)
    precision = ws.precision
    linear = ws.linear
    logconst = ws.logconst
    for node in tree.postorder_internal
        fill!(precision[node], 0.0)
        fill!(linear[node], 0.0)
        logconst[node] = 0.0
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = Int(tree.child_of_edge[edge])
            ti = ws.edge_time_idx[edge]
            Phi = @view ws.Phi[:, :, ti]
            Q = @view ws.Q[:, :, ti]
            mu = ws.edge_mu[edge]
            ti = ws.edge_time_idx[edge]
            qinv = @view ws.Qinv[:, :, ti]
            logdet_q = ws.logdet_Q[ti]
            qphi = @view ws.QPhi[:, :, ti]
            ok = if tree.is_tip[child]
                _mvoubm_tip_message!(tree, data, ws, node, child, Phi, Q, qinv, qphi, logdet_q, mu)
            else
                _mvoubm_internal_message!(ws, node, child, Phi, qinv, qphi, logdet_q, mu)
            end
            ok || return (success = false, loglik = -Inf)
        end
    end
    root = Int(tree.root)
    ll = _mvou_root_info_loglik(
        precision[root], linear[root], logconst[root], ws.root_mean, ws.root_cov,
    )
    return (success = isfinite(ll), loglik = ll)
end

function _mvoubm_tip_message!(
    tree::CompactTree,
    data::AbstractMatrix{Float64},
    ws::MVOUBMWorkspace,
    node::Integer,
    child::Int,
    Phi::AbstractMatrix{Float64},
    Q::AbstractMatrix{Float64},
    qinv::AbstractMatrix{Float64},
    qphi::AbstractMatrix{Float64},
    logdet_q::Float64,
    mu::AbstractVector{Float64},
)
    row = ws.tip_index[child]
    y = ws.ybuf
    copyto!(y, @view ws.data_t[:, row])
    y .-= mu
    if ws.tip_nan[row]
        msg = _mvou_observation_info_to_parent(y, Phi, Q)
        msg.success || return false
        ws.precision[node] .+= msg.precision
        ws.linear[node] .+= msg.linear
        ws.logconst[node] += msg.logconst
        return true
    end
    tp1 = ws.tp1
    tp2 = ws.tp2
    tv1 = ws.tv1
    tv2 = ws.tv2
    mul!(tp2, transpose(Phi), qphi)
    mul!(tv1, qinv, y)
    mul!(tv2, transpose(Phi), tv1)
    ws.precision[node] .+= tp2
    ws.linear[node] .+= tv2
    ws.logconst[node] += -0.5 * (dot(y, tv1) + logdet_q + ws.p * log(2 * pi))
    return true
end

function _mvoubm_internal_message!(
    ws::MVOUBMWorkspace,
    node::Integer,
    child::Integer,
    Phi::AbstractMatrix{Float64},
    qinv::AbstractMatrix{Float64},
    qphi::AbstractMatrix{Float64},
    logdet_q::Float64,
    mu::AbstractVector{Float64},
)
    P = ws.precision[child]
    h = ws.linear[child]
    mul!(ws.hbuf, P, mu)
    @. ws.hbuf = h - ws.hbuf
    c2 = ws.logconst[child] + dot(h, mu) - 0.5 * dot(mu, P * mu)
    tp1 = ws.tp1
    tp2 = ws.tp2
    tp3 = ws.tp3
    tp4 = ws.tp4
    tS = ws.tS
    tv1 = ws.tv1
    tv2 = ws.tv2
    p = ws.p
    tp1 = qphi
    @inbounds for j in 1:p, i in 1:p
        tS[i, j] = P[i, j] + qinv[i, j]
    end
    local C
    try
        C = cholesky!(Symmetric(tS))
    catch
        return false
    end
    logdet_S = 0.0
    @inbounds for j in 1:p
        logdet_S += log(C.L[j, j])
    end
    logdet_S *= 2.0
    copyto!(tv1, ws.hbuf)
    ldiv!(C, tv1)
    copyto!(tp2, tp1)
    ldiv!(C, tp2)
    mul!(tp3, transpose(tp1), Phi)
    mul!(tp4, transpose(tp1), tp2)
    @inbounds for j in 1:p, i in 1:p
        tp3[i, j] -= tp4[i, j]
    end
    mul!(tv2, transpose(tp2), ws.hbuf)
    ws.precision[node] .+= tp3
    ws.linear[node] .+= tv2
    ws.logconst[node] += c2 - 0.5 * (logdet_q + logdet_S) + 0.5 * dot(ws.hbuf, tv1)
    return true
end

function _mvoubm_prepare!(
    tree::CompactTree,
    data::AbstractMatrix{Float64},
    precalc::MVOUPrecalc,
    ws::MVOUBMWorkspace,
    pars::AbstractVector{<:Real},
)
    kY = ws.kY
    kX = ws.kX
    p = ws.p
    up = try
        _mvoubm_unpack_params(pars, kY, kX, precalc.nregimes, precalc.A_decomp)
    catch
        return nothing
    end
    if minimum(real(eigen(complex(up.A)).values)) < 1e-10
        return nothing
    end
    H = [up.A up.B; zeros(Float64, kX, kY) zeros(Float64, kX, kX)]
    Sigma_ee = [
        up.SigmaY zeros(Float64, kY, kX)
        zeros(Float64, kX, kY) up.SigmaX
    ]
    eigH = eigen(complex(H))
    Vinv = inv(eigH.vectors)
    Mrot = Vinv * Sigma_ee * Vinv'
    _mvoubm_fill_caches!(ws, eigH, Vinv, Mrot)

    theta_full = [up.mPsi; zeros(Float64, kX, precalc.nregimes)]
    for e in 1:tree.nedges
        mu = ws.edge_mu[e]
        fill!(mu, 0.0)
        terms = ws.edge_offset_idx[e]
        regs = ws.edge_offset_regime[e]
        for i in eachindex(terms)
            ti, si = terms[i]
            theta = @view theta_full[:, regs[i]]
            mul!(ws.seg_mu, @view(ws.Phi_off[:, :, si]), theta)
            @. ws.seg_mu = theta - ws.seg_mu
            mul!(mu, @view(ws.Phi_off[:, :, ti]), ws.seg_mu, 1.0, 1.0)
        end
    end

    root_r = min(max(Int(precalc.root_regime), 1), precalc.nregimes)
    vY0 = up.mPsi[:, root_r] - up.A \ (up.B * up.vX0)
    copyto!(ws.root_mean, 1, vY0, 1, kY)
    copyto!(ws.root_mean, kY + 1, up.vX0, 1, kX)
    _mvoubm_stage_data!(ws, data)
    return up
end

function _mvoubm_loglik!(
    tree::CompactTree,
    data::AbstractMatrix{Float64},
    precalc::MVOUPrecalc,
    ws::MVOUBMWorkspace,
    pars::AbstractVector{<:Real},
)
    _mvoubm_prepare!(tree, data, precalc, ws, pars) === nothing && return -Inf
    prof = _mvoubm_profile!(tree, data, ws)
    return prof.success ? Float64(prof.loglik) : -Inf
end

function mvoubm_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    A::AbstractMatrix{<:Real},
    B::AbstractMatrix{<:Real},
    SigmaY::AbstractMatrix{<:Real},
    SigmaX::AbstractMatrix{<:Real},
    mPsi::AbstractMatrix{<:Real},
    ;
    vX0::AbstractVector{<:Real} = zeros(Float64, size(B, 2)),
    A_decomp::Symbol = :cholesky,
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    kY = size(A, 1)
    kX = p - kY
    kX == size(B, 2) == size(SigmaX, 1) ||
        throw(ArgumentError("trait/B/SigmaX dimensions are inconsistent"))
    spec = MVOUSpec(
        model = :mvOUBM,
        theta_mode = :by_regime,
        A_mode = :shared,
        Sigma_mode = :shared,
        A_decomp = A_decomp,
        root_mean_mode = :theta,
        root_cov_mode = :fixed,
    )
    precalc = _mvou_build_precalc(tree, spec; edge_segments = edge_segments)
    ws = _mvoubm_workspace(tree, precalc, kY, kX)
    pars = vcat(
        _mvoubm_pack_A_block(A_decomp, A),
        vec(Matrix{Float64}(B)),
        _mvou_pack_spd(SigmaY),
        _mvou_pack_spd(SigmaX),
        vec(Matrix{Float64}(mPsi)),
        Float64.(vX0),
    )
    ll = _mvoubm_loglik!(tree, data, precalc, ws, pars)
    nparams = _mvoubm_nparams(kY, kX, size(mPsi, 2), A_decomp)
    return MVContinuousOUResult(
        model = :mvOUBM,
        success = isfinite(ll),
        loglik = ll,
        aic = isfinite(ll) ? compute_aic(ll, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = size(mPsi, 2),
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, size(mPsi, 2), "regime"),
        theta = Matrix{Float64}(permutedims(mPsi)),
        A = reshape(Matrix{Float64}(A), kY, kY, 1),
        Sigma = reshape([
            Matrix{Float64}(SigmaY) zeros(Float64, kY, kX)
            zeros(Float64, kX, kY) Matrix{Float64}(SigmaX)
        ], p, p, 1),
        A_decomp = A_decomp,
        root_mean_mode = :theta,
        root_cov_mode = :fixed,
        B = Matrix{Float64}(B),
        vX0 = Float64.(vX0),
    )
end
