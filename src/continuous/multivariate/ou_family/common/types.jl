Base.@kwdef struct MVOUSpec
    model::Symbol
    theta_mode::Symbol = :shared
    A_mode::Symbol = :shared
    Sigma_mode::Symbol = :shared
    A_decomp::Symbol = :cholesky
    root_mean_mode::Symbol = :theta
    root_cov_mode::Symbol = :fixed
end

function mvou_spec(model::Symbol; A_decomp::Symbol = :cholesky)
    if model === :mvOU1
        return MVOUSpec(
            model = :mvOU1,
            theta_mode = :shared,
            A_mode = :shared,
            Sigma_mode = :shared,
            A_decomp = A_decomp,
            root_mean_mode = :theta,
            root_cov_mode = :fixed,
        )
    elseif model === :mvOUM
        return MVOUSpec(
            model = :mvOUM,
            theta_mode = :by_regime,
            A_mode = :shared,
            Sigma_mode = :shared,
            A_decomp = A_decomp,
            root_mean_mode = :stationary_design,
            root_cov_mode = :fixed,
        )
    elseif model === :mvOUMV
        return MVOUSpec(
            model = :mvOUMV,
            theta_mode = :by_regime,
            A_mode = :shared,
            Sigma_mode = :by_regime,
            A_decomp = A_decomp,
            root_mean_mode = :stationary_design,
            root_cov_mode = :fixed,
        )
    elseif model === :mvOUMA
        return MVOUSpec(
            model = :mvOUMA,
            theta_mode = :by_regime,
            A_mode = :by_regime,
            Sigma_mode = :shared,
            A_decomp = A_decomp,
            root_mean_mode = :stationary_design,
            root_cov_mode = :fixed,
        )
    elseif model === :mvOUMVA
        return MVOUSpec(
            model = :mvOUMVA,
            theta_mode = :by_regime,
            A_mode = :by_regime,
            Sigma_mode = :by_regime,
            A_decomp = A_decomp,
            root_mean_mode = :stationary_design,
            root_cov_mode = :fixed,
        )
    end
    throw(ArgumentError("Unsupported multivariate OU model $model"))
end

@inline function _mvou_A_block_nparams(spec::MVOUSpec, p::Integer)
    if spec.A_decomp === :cholesky
        return div(p * (p + 1), 2)
    elseif spec.A_decomp === :schur
        p == 2 || throw(ArgumentError("A_decomp=:schur is currently supported only for p=2"))
        return p * p
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$(spec.A_decomp)"))
end

@inline function _mvou_sigma_block_nparams(p::Integer)
    return div(p * (p + 1), 2)
end


@inline function _mvou_pack_spd(M::AbstractMatrix{<:Real})
    p = size(M, 1)
    L = cholesky(Symmetric(Matrix{Float64}(M))).L
    pars = Float64[]
    for j in 1:p
        push!(pars, max(L[j, j], 1e-8))
        for i in (j + 1):p
            push!(pars, L[i, j])
        end
    end
    return pars
end

@inline function _mvou_pack_schur2(A::AbstractMatrix{<:Real})
    size(A) == (2, 2) || throw(ArgumentError("Schur A parameterization currently requires a 2x2 matrix"))
    A64 = Matrix{Float64}(A)
    return Float64[A64[1, 1], A64[2, 2], A64[1, 2], A64[2, 1]]
end

@inline function _mvou_unpack_schur2(pars::AbstractVector{<:Real})
    length(pars) == 4 || throw(ArgumentError("2x2 Schur parameter vector must have 4 entries"))
    B = [
        Float64(pars[1]) Float64(pars[3]);
        Float64(pars[4]) Float64(pars[2]);
    ]
    F = schur(B)
    Z = Matrix{Float64}(F.Z)
    T = Matrix{Float64}(F.T)
    D = Matrix{Float64}(I, 2, 2)
    if det(Z) < 0.0
        D[1, 1] = -1.0
        Z = Z * D
        T = D * T * D
    end
    if abs(T[2, 1]) <= 1e-8
        Tstable = [
            exp(T[1, 1]) T[1, 2];
            0.0 exp(T[2, 2]);
        ]
    else
        a = exp(0.5 * (T[1, 1] + T[2, 2]))
        Tstable = [
            a T[1, 2];
            T[2, 1] a;
        ]
    end
    return Z * Tstable * transpose(Z)
end

@inline function _mvou_pack_A(spec::MVOUSpec, A::AbstractMatrix{<:Real})
    if spec.A_decomp === :cholesky
        return _mvou_pack_spd(A)
    elseif spec.A_decomp === :schur
        return _mvou_pack_schur2(A)
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$(spec.A_decomp)"))
end

@inline function _mvou_unpack_A(spec::MVOUSpec, pars::AbstractVector{<:Real}, p::Integer)
    if spec.A_decomp === :cholesky
        return _mvou_unpack_spd(pars, p)
    elseif spec.A_decomp === :schur
        p == 2 || throw(ArgumentError("A_decomp=:schur is currently supported only for p=2"))
        return _mvou_unpack_schur2(pars)
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$(spec.A_decomp)"))
end

function _mvou_unpack_spd(pars::AbstractVector{<:Real}, p::Integer)
    expected = div(p * (p + 1), 2)
    length(pars) == expected || throw(ArgumentError("SPD parameter vector must have $expected entries"))
    L = zeros(Float64, p, p)
    idx = 1
    for j in 1:p
        L[j, j] = max(Float64(pars[idx]), 1e-8)
        idx += 1
        for i in (j + 1):p
            L[i, j] = Float64(pars[idx])
            idx += 1
        end
    end
    return L * L'
end

function _mvou_unpack_spd!(
    M::AbstractMatrix{Float64},
    pars::AbstractVector{<:Real},
    idx::Integer,
    p::Integer,
    L::AbstractMatrix{Float64},
)
    fill!(L, 0.0)
    k = Int(idx)
    @inbounds for j in 1:p
        L[j, j] = max(Float64(pars[k]), 1e-8)
        k += 1
        for i in (j + 1):p
            L[i, j] = Float64(pars[k])
            k += 1
        end
    end
    fill!(M, 0.0)
    @inbounds for j in 1:p
        for kcol in 1:j
            ljk = L[j, kcol]
            for i in j:p
                M[i, j] += L[i, kcol] * ljk
            end
        end
    end
    @inbounds for j in 1:p
        for i in (j + 1):p
            M[j, i] = M[i, j]
        end
    end
    return Int(idx) + div(Int(p) * (Int(p) + 1), 2)
end

function _mvou_initial_params(
    spec::MVOUSpec,
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    nregimes::Integer = 1,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])
    scale = max(log(2.0) / max(tree_height / 4.0, 1e-8), 1e-6)
    A0 = Matrix{Float64}(I, p, p) .* scale
    empirical = _mv_complete_rows_cov(data)
    empirical = (empirical + empirical') / 2
    scatter = A0 * empirical + empirical * A0'
    Sigma0 = (scatter + scatter') / 2 + 1e-6 * Matrix{Float64}(I, p, p)
    theta0 = repeat(vec(mean(data; dims = 1)), nregimes)
    return _mvou_pack_initial_params(spec, p, nregimes, A0, Sigma0, theta0)
end

function _mvou_pack_initial_params(
    spec::MVOUSpec,
    p::Integer,
    nregimes::Integer,
    A0::AbstractArray{<:Real},
    Sigma0::AbstractArray{<:Real},
    theta0::AbstractVector{<:Real},
)
    A0_matrix = ndims(A0) == 3 ? Matrix{Float64}(@view A0[:, :, 1]) : Matrix{Float64}(A0)
    Sigma0_matrix = ndims(Sigma0) == 3 ? Matrix{Float64}(@view Sigma0[:, :, 1]) : Matrix{Float64}(Sigma0)
    A0_regimes = ndims(A0) == 3 ? Array{Float64, 3}(A0) : nothing
    Sigma0_regimes = ndims(Sigma0) == 3 ? Array{Float64, 3}(Sigma0) : nothing
    theta_vec = Float64.(collect(theta0))

    size(A0_matrix, 1) == p == size(A0_matrix, 2) || throw(ArgumentError("initial A dimensions must match trait dimension"))
    size(Sigma0_matrix, 1) == p == size(Sigma0_matrix, 2) || throw(ArgumentError("initial Sigma dimensions must match trait dimension"))
    length(theta_vec) == p * nregimes || throw(ArgumentError("initial theta length must match trait dimension times regime count"))

    if spec.A_decomp === :cholesky
        A0_matrix = (A0_matrix + A0_matrix') / 2 + 1e-8 * Matrix{Float64}(I, p, p)
    elseif spec.A_decomp === :schur
        p == 2 || throw(ArgumentError("A_decomp=:schur is currently supported only for p=2"))
        vals = eigvals(Matrix{Float64}(A0_matrix))
        any(z -> real(z) <= 0.0 || abs(imag(z)) > 1e-8, vals) && (A0_matrix .= Matrix{Float64}(I, p, p))
    end
    Sigma0_matrix = (Sigma0_matrix + Sigma0_matrix') / 2 + 1e-8 * Matrix{Float64}(I, p, p)

    pars = Float64[]
    if spec.A_mode === :by_regime
        if A0_regimes !== nothing
            size(A0_regimes, 1) == p == size(A0_regimes, 2) || throw(ArgumentError("initial A dimensions must match trait dimension"))
            size(A0_regimes, 3) == nregimes || throw(ArgumentError("initial A regime count must match nregimes"))
            for r in 1:nregimes
                append!(pars, _mvou_pack_A(spec, @view A0_regimes[:, :, r]))
            end
        else
            for _ in 1:nregimes
                append!(pars, _mvou_pack_A(spec, A0_matrix))
            end
        end
    else
        append!(pars, _mvou_pack_A(spec, A0_matrix))
    end
    if spec.Sigma_mode === :by_regime
        if Sigma0_regimes !== nothing
            size(Sigma0_regimes, 1) == p == size(Sigma0_regimes, 2) || throw(ArgumentError("initial Sigma dimensions must match trait dimension"))
            size(Sigma0_regimes, 3) == nregimes || throw(ArgumentError("initial Sigma regime count must match nregimes"))
            for r in 1:nregimes
                append!(pars, _mvou_pack_spd(@view Sigma0_regimes[:, :, r]))
            end
        else
            for _ in 1:nregimes
                append!(pars, _mvou_pack_spd(Sigma0_matrix))
            end
        end
    else
        append!(pars, _mvou_pack_spd(Sigma0_matrix))
    end
    append!(pars, theta_vec)
    return pars
end

function _mvou_initial_cov_params(
    spec::MVOUSpec,
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    nregimes::Integer = 1,
)
    full = _mvou_initial_params(
        spec,
        tree,
        trait;
        nregimes = nregimes,
    )
    p = size(trait, 2)
    return full[1:(end - p * nregimes)]
end

function _mvou_pack_initial_cov_params(
    spec::MVOUSpec,
    p::Integer,
    nregimes::Integer,
    A0::AbstractArray{<:Real},
    Sigma0::AbstractArray{<:Real},
)
    full = _mvou_pack_initial_params(spec, p, nregimes, A0, Sigma0, zeros(Float64, p * nregimes))
    return full[1:(end - p * nregimes)]
end

function _mvou_unpack_params(spec::MVOUSpec, pars::AbstractVector{<:Real}, p::Integer)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    if spec.theta_mode === :by_regime
        shared_blocks = (spec.A_mode === :shared ? Ablock : 0) + (spec.Sigma_mode === :shared ? Sblock : 0)
        regime_blocks = p + (spec.A_mode === :by_regime ? Ablock : 0) + (spec.Sigma_mode === :by_regime ? Sblock : 0)
        numer = length(pars) - shared_blocks
        numer >= 0 || throw(ArgumentError("Parameter vector length does not match $(spec.model)"))
        nregimes = Int(div(numer, regime_blocks))
        numer == nregimes * regime_blocks || throw(ArgumentError("Parameter vector length does not match $(spec.model)"))
    else
        nregimes = 1
    end
    expected =
        (spec.A_mode === :by_regime ? nregimes * Ablock : Ablock) +
        (spec.Sigma_mode === :by_regime ? nregimes * Sblock : Sblock) +
        p * nregimes
    length(pars) == expected || throw(ArgumentError("Parameter vector length does not match $(spec.model)"))

    idx = 1
    A = zeros(Float64, p, p)
    A_regimes = zeros(Float64, 0, 0, 0)
    if spec.A_mode === :by_regime
        A_regimes = Array{Float64, 3}(undef, p, p, nregimes)
        for r in 1:nregimes
            A_regimes[:, :, r] .= _mvou_unpack_A(spec, pars[idx:(idx + Ablock - 1)], p)
            idx += Ablock
        end
        A .= A_regimes[:, :, 1]
    else
        A .= _mvou_unpack_A(spec, pars[idx:(idx + Ablock - 1)], p)
        idx += Ablock
    end

    Sigma = zeros(Float64, p, p)
    Sigma_regimes = zeros(Float64, 0, 0, 0)
    if spec.Sigma_mode === :by_regime
        Sigma_regimes = Array{Float64, 3}(undef, p, p, nregimes)
        for r in 1:nregimes
            Sigma_regimes[:, :, r] .= _mvou_unpack_spd(pars[idx:(idx + Sblock - 1)], p)
            idx += Sblock
        end
        Sigma .= Sigma_regimes[:, :, 1]
    else
        Sigma .= _mvou_unpack_spd(pars[idx:(idx + Sblock - 1)], p)
        idx += Sblock
    end

    theta = Float64.(pars[idx:(idx + p * nregimes - 1)])
    return MVOUParameterBundle(theta = theta, A = A, A_regimes = A_regimes, Sigma = Sigma, Sigma_regimes = Sigma_regimes)
end

function _mvou_unpack_cov_params(spec::MVOUSpec, pars::AbstractVector{<:Real}, p::Integer, nregimes::Integer)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    expected =
        (spec.A_mode === :by_regime ? nregimes * Ablock : Ablock) +
        (spec.Sigma_mode === :by_regime ? nregimes * Sblock : Sblock)
    length(pars) == expected || throw(ArgumentError("Covariance parameter vector length does not match $(spec.model)"))

    idx = 1
    A = zeros(Float64, p, p)
    A_regimes = zeros(Float64, 0, 0, 0)
    if spec.A_mode === :by_regime
        A_regimes = Array{Float64, 3}(undef, p, p, nregimes)
        for r in 1:nregimes
            A_regimes[:, :, r] .= _mvou_unpack_A(spec, pars[idx:(idx + Ablock - 1)], p)
            idx += Ablock
        end
        A .= A_regimes[:, :, 1]
    else
        A .= _mvou_unpack_A(spec, pars[idx:(idx + Ablock - 1)], p)
        idx += Ablock
    end

    Sigma = zeros(Float64, p, p)
    Sigma_regimes = zeros(Float64, 0, 0, 0)
    if spec.Sigma_mode === :by_regime
        Sigma_regimes = Array{Float64, 3}(undef, p, p, nregimes)
        for r in 1:nregimes
            Sigma_regimes[:, :, r] .= _mvou_unpack_spd(pars[idx:(idx + Sblock - 1)], p)
            idx += Sblock
        end
        Sigma .= Sigma_regimes[:, :, 1]
    else
        Sigma .= _mvou_unpack_spd(pars[idx:(idx + Sblock - 1)], p)
    end

    return MVOUParameterBundle(
        theta = zeros(Float64, p * nregimes),
        A = A,
        A_regimes = A_regimes,
        Sigma = Sigma,
        Sigma_regimes = Sigma_regimes,
    )
end

function _mvou_cov_unpack_workspace(spec::MVOUSpec, p::Integer, nregimes::Integer)
    p = Int(p)
    nregimes = Int(nregimes)
    A = zeros(Float64, p, p)
    A_regimes = spec.A_mode === :by_regime ? Array{Float64, 3}(undef, p, p, nregimes) : zeros(Float64, 0, 0, 0)
    Sigma = zeros(Float64, p, p)
    Sigma_regimes = spec.Sigma_mode === :by_regime ? Array{Float64, 3}(undef, p, p, nregimes) : zeros(Float64, 0, 0, 0)
    bundle = MVOUParameterBundle(
        theta = zeros(Float64, p * nregimes),
        A = A,
        A_regimes = A_regimes,
        Sigma = Sigma,
        Sigma_regimes = Sigma_regimes,
    )
    return (bundle = bundle, L = zeros(Float64, p, p))
end

function _mvou_unpack_cov_params!(
    bundle::MVOUParameterBundle,
    L::AbstractMatrix{Float64},
    spec::MVOUSpec,
    pars::AbstractVector{<:Real},
    p::Integer,
    nregimes::Integer,
)
    p = Int(p)
    nregimes = Int(nregimes)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    expected =
        (spec.A_mode === :by_regime ? nregimes * Ablock : Ablock) +
        (spec.Sigma_mode === :by_regime ? nregimes * Sblock : Sblock)
    length(pars) == expected || throw(ArgumentError("Covariance parameter vector length does not match $(spec.model)"))

    idx = 1
    if spec.A_mode === :by_regime
        for r in 1:nregimes
            if spec.A_decomp === :cholesky
                idx = _mvou_unpack_spd!(@view(bundle.A_regimes[:, :, r]), pars, idx, p, L)
            else
                bundle.A_regimes[:, :, r] .= _mvou_unpack_A(spec, pars[idx:(idx + Ablock - 1)], p)
                idx += Ablock
            end
        end
        copyto!(bundle.A, @view(bundle.A_regimes[:, :, 1]))
    else
        if spec.A_decomp === :cholesky
            idx = _mvou_unpack_spd!(bundle.A, pars, idx, p, L)
        else
            bundle.A .= _mvou_unpack_A(spec, pars[idx:(idx + Ablock - 1)], p)
            idx += Ablock
        end
    end

    if spec.Sigma_mode === :by_regime
        for r in 1:nregimes
            idx = _mvou_unpack_spd!(@view(bundle.Sigma_regimes[:, :, r]), pars, idx, p, L)
        end
        copyto!(bundle.Sigma, @view(bundle.Sigma_regimes[:, :, 1]))
    else
        _mvou_unpack_spd!(bundle.Sigma, pars, idx, p, L)
    end
    return bundle
end




@inline function _mvou_allow_zero_internal_edge_mismatch(tree::CompactTree, edge::Integer, total::Float64; atol::Float64 = 1e-8)
    child = Int(tree.child_of_edge[edge])
    tree.is_tip[child] && return false
    tree.edge_length[edge] == 0.0 || return false
    return abs(total) <= 10 * atol
end

function _mvou_validate_edge_segments(
    tree::CompactTree,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}},
)
    if edge_segments === nothing
        return (edge_segments = Vector{Vector{SimmapSegment}}(), tip_terminal_regime = fill(Int32(1), tree.ntips), nregimes = 1, root_regime = 1)
    end
    length(edge_segments) == tree.nedges || throw(ArgumentError("edge_segments must have $(tree.nedges) entries"))
    nregimes = 0
    tip_index = zeros(Int, tree.nnodes)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_index[tip] = i
    end
    tip_terminal_regime = zeros(Int32, tree.ntips)
    for edge in 1:tree.nedges
        segs = edge_segments[edge]
        isempty(segs) && throw(ArgumentError("edge_segments[$edge] is empty"))
        total = 0.0
        for seg in segs
            total += Float64(seg.length)
            nregimes = max(nregimes, Int(seg.state))
        end
        if !(isapprox(total, tree.edge_length[edge]; atol = 1e-8) || _mvou_allow_zero_internal_edge_mismatch(tree, edge, total; atol = 1e-8))
            throw(ArgumentError("edge_segments[$edge] lengths do not sum to branch length"))
        end
        child = Int(tree.child_of_edge[edge])
        if tree.is_tip[child]
            tip_terminal_regime[tip_index[child]] = Int32(segs[end].state)
        end
    end
    root = Int(tree.root)
    first_edge = Int(tree.first_child_edge[root])
    first_edge > 0 || throw(ArgumentError("Root must have outgoing edges"))
    root_regime = Int(edge_segments[first_edge][1].state)
    for edge in tree.first_child_edge[root]:tree.last_child_edge[root]
        Int(edge_segments[Int(edge)][1].state) == root_regime || throw(ArgumentError("Root outgoing edges do not share a consistent initial regime"))
    end
    all(>(0), tip_terminal_regime) || throw(ArgumentError("Could not resolve terminal regime for every tip"))
    return (edge_segments = edge_segments, tip_terminal_regime = tip_terminal_regime, nregimes = nregimes, root_regime = root_regime)
end

function _mvou_regularize_zero_length_internal_edges!(
    tree::CompactTree,
    branch_lengths::Vector{Float64},
    edge_segments::Vector{Vector{SimmapSegment}};
    min_length::Float64 = 1e-8,
)
    length(branch_lengths) == tree.nedges || throw(ArgumentError("branch length count must match tree.nedges"))
    length(edge_segments) == tree.nedges || throw(ArgumentError("edge_segments count must match tree.nedges"))
    min_length > 0 || throw(ArgumentError("min_length must be positive"))
    for edge in 1:tree.nedges
        child = Int(tree.child_of_edge[edge])
        tree.is_tip[child] && continue
        branch_lengths[edge] > 0.0 && continue
        segs = edge_segments[edge]
        isempty(segs) && continue
        reg = Float64(min_length - branch_lengths[edge])
        reg > 0.0 || continue
        branch_lengths[edge] = min_length
        last = segs[end]
        segs[end] = SimmapSegment(state = last.state, length = last.length + reg)
    end
    return edge_segments
end

function _mvou_build_precalc(
    tree::CompactTree,
    spec::MVOUSpec,
    ;
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    checked = _mvou_validate_edge_segments(tree, edge_segments)
    branch_lengths = Float64.(tree.edge_length)
    segments = [copy(v) for v in checked.edge_segments]
    if !isempty(segments)
        _mvou_regularize_zero_length_internal_edges!(tree, branch_lengths, segments)
    end
    return MVOUPrecalc(
        branch_lengths = branch_lengths,
        edge_order = Int32.(collect(1:tree.nedges)),
        tip_terminal_regime = checked.tip_terminal_regime,
        edge_segments = segments,
        nregimes = checked.nregimes,
        root_regime = checked.root_regime,
        A_decomp = spec.A_decomp,
    )
end

function _mvou_precalc(tree::CompactTree, spec::MVOUSpec; edge_segments = nothing)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    spec.root_cov_mode == :fixed || throw(ArgumentError("multivariate OU tree pruning requires root_cov_mode=:fixed"))
    return _mvou_build_precalc(tree, spec; edge_segments = edge_segments)
end

@inline function _mvou_profile_family(bundle::MVOUParameterBundle)
    is_ou1 = length(bundle.theta) == size(bundle.A, 1)
    is_shared_sigma = size(bundle.Sigma_regimes, 3) == 0
    is_shared_A = size(bundle.A_regimes, 3) == 0
    if is_ou1
        return :mvOU1
    elseif is_shared_sigma && is_shared_A
        return :mvOUM
    elseif is_shared_A
        return :mvOUMV
    elseif is_shared_sigma
        return :mvOUMA
    end
    return :mvOUMVA
end

@inline function _mvou_profile_handler(family::Symbol)
    if family === :mvOU1
        return _mvou1_tree_pruning_profile
    elseif family === :mvOUM
        return _mvoum_tree_pruning_profile
    elseif family === :mvOUMV
        return _mvoumv_tree_pruning_profile
    elseif family === :mvOUMA
        return _mvouma_tree_pruning_profile
    elseif family === :mvOUMVA
        return _mvoumva_tree_pruning_profile
    end
    throw(ArgumentError("Unsupported multivariate OU family $family"))
end

@inline function _mvou_asr_handler(model::Symbol)
    if model === :mvOU1
        return _mvou1_tree_pruning_asr
    elseif model in (:mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA)
        return _mvoum_family_tree_pruning_asr
    end
    throw(ArgumentError("Unsupported multivariate OU model $model"))
end

function _mvou_profile_dispatch(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    family = _mvou_profile_family(bundle)
    handler = _mvou_profile_handler(family)
    return handler(tree, trait, bundle, precalc, workspace)
end

function _mvou_asr_dispatch(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    handler = _mvou_asr_handler(fit.model)
    if fit.model === :mvOU1
        return handler(tree, trait, fit)
    end
    edge_segments === nothing && throw(ArgumentError("$(fit.model) ASR requires edge_segments"))
    return handler(tree, trait, fit, edge_segments)
end



mutable struct MVOUThetaProfileWorkspace
    precision::Vector{Matrix{Float64}}
    linear::Vector{Vector{Float64}}
    logconst::Vector{Float64}
    tip_index::Vector{Int}
    msg_precision::Matrix{Float64}
    msg_linear::Vector{Float64}
    F::Matrix{Float64}
    Qobs::Matrix{Float64}
    yobs::Vector{Float64}
    obs_index::Vector{Int}
    Qinv_F::Matrix{Float64}
    Qinv_y::Vector{Float64}
    Qinv::Matrix{Float64}
    Qinv_Phi::Matrix{Float64}
    Avv::Matrix{Float64}
    Ava::Matrix{Float64}
    Aaa::Matrix{Float64}
    ha::Vector{Float64}
    solve_Ava::Matrix{Float64}
    solve_hv::Vector{Float64}
    theta::Vector{Float64}
    edge_Phi::Array{Float64, 3}
    edge_Q::Array{Float64, 3}
    edge_Qinv::Array{Float64, 3}
    edge_logdet_Q::Vector{Float64}
    identity_designs::Vector{Matrix{Float64}}
    regime_designs::Vector{Matrix{Float64}}
    design_work::Matrix{Float64}
    transition_work::Matrix{Float64}
    edge_chol_work::Matrix{Float64}
    cached_A::Matrix{Float64}
    has_cached_A::Bool
    shared_design_valid::Bool
    shared_phi_valid::Bool
end

function _mvou_theta_profile_workspace(tree::CompactTree, p::Integer, q::Integer; nregimes::Integer = max(1, div(Int(q), Int(p))))
    p = Int(p)
    q = Int(q)
    m = p + q
    nregimes = Int(nregimes)
    base = _mv_profile_workspace(tree, m)
    return MVOUThetaProfileWorkspace(
        base.precision,
        base.linear,
        base.logconst,
        base.tip_index,
        zeros(Float64, m, m),
        zeros(Float64, m),
        zeros(Float64, p, m),
        zeros(Float64, p, p),
        zeros(Float64, p),
        zeros(Int, p),
        zeros(Float64, p, m),
        zeros(Float64, p),
        zeros(Float64, p, p),
        zeros(Float64, p, p),
        zeros(Float64, p, p),
        zeros(Float64, p, m),
        zeros(Float64, m, m),
        zeros(Float64, m),
        zeros(Float64, p, m),
        zeros(Float64, p),
        zeros(Float64, q),
        Array{Float64, 3}(undef, p, p, tree.nedges),
        Array{Float64, 3}(undef, p, p, tree.nedges),
        Array{Float64, 3}(undef, p, p, tree.nedges),
        zeros(Float64, tree.nedges),
        [Matrix{Float64}(I, p, p) for _ in 1:tree.nnodes],
        [zeros(Float64, p, p * nregimes) for _ in 1:tree.nnodes],
        zeros(Float64, p, p * nregimes),
        zeros(Float64, p, p),
        zeros(Float64, p, p),
        zeros(Float64, p, p),
        false,
        false,
        false,
    )
end

