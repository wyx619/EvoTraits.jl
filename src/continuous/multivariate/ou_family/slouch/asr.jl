# Joint (Y, X) ancestral state reconstruction for fitted mvOUBM models.

function _mvoubm_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}},
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    p == fit.ntraits || throw(ArgumentError("trait dimension does not match the fitted model"))
    size(fit.A, 3) == 1 || throw(ArgumentError("mvOUBM ASR requires a single A matrix"))
    A = fit.A[:, :, 1]
    kY = size(A, 1)
    kX = p - kY
    size(fit.B) == (kY, kX) || throw(ArgumentError("B matrix dimensions are inconsistent"))
    length(fit.vX0) == kX || throw(ArgumentError("vX0 dimension is inconsistent"))
    Sigma = fit.Sigma[:, :, 1]
    SigmaY = Sigma[1:kY, 1:kY]
    SigmaX = Sigma[(kY + 1):end, (kY + 1):end]
    mPsi = Matrix{Float64}(permutedims(fit.theta))

    # A is fixed during ASR. Repack it through the free representation so
    # the likelihood path reconstructs exactly the matrix stored in fit.A,
    # without an unnecessary decomposition round trip.
    spec = MVOUSpec(
        model = :mvOUBM,
        theta_mode = :by_regime,
        A_mode = :shared,
        Sigma_mode = :shared,
        A_decomp = :free,
        root_mean_mode = :theta,
        root_cov_mode = :fixed,
    )
    precalc = _mvou_build_precalc(tree, spec; edge_segments = edge_segments)
    ws = _mvoubm_workspace(tree, precalc, kY, kX)
    pars = vcat(
        vec(Matrix{Float64}(A)),
        vec(Matrix{Float64}(fit.B)),
        _mvou_pack_spd(SigmaY),
        _mvou_pack_spd(SigmaX),
        vec(mPsi),
        Float64.(fit.vX0),
    )
    _mvoubm_prepare!(tree, data, precalc, ws, pars) === nothing &&
        return MVContinuousASRResult(model = :mvOUBM)

    # Center the entire affine process, including offsets on ancestral edges.
    node_means = [zeros(Float64, p) for _ in 1:tree.nnodes]
    node_means[Int(tree.root)] .= ws.root_mean
    for parent in tree.preorder
        tree.is_tip[parent] && continue
        for e in tree.first_child_edge[parent]:tree.last_child_edge[parent]
            child = Int(tree.child_of_edge[e])
            mul!(node_means[child], @view(ws.Phi[:, :, ws.edge_time_idx[e]]), node_means[parent])
            node_means[child] .+= ws.edge_mu[e]
        end
    end
    data_shifted = copy(data)
    for (i, tip) in enumerate(tree.tip_ids)
        data_shifted[i, :] .-= node_means[Int(tip)]
    end
    edge_Phi = ws.Phi[:, :, ws.edge_time_idx]
    edge_Q = ws.Q[:, :, ws.edge_time_idx]
    result = _mv_recursive_asr(
        tree,
        data_shifted,
        edge_Phi,
        edge_Q;
        root_mean = zeros(Float64, p),
        root_cov = ws.root_cov,
        shift = zeros(Float64, p),
        model = :mvOUBM,
        regression = -(A \ fit.B),
    )
    result.success || return result
    for (i, node) in enumerate(result.node_ids)
        result.estimates[i, :] .+= node_means[Int(node)]
    end
    for node in 1:tree.nnodes
        result.all_node_estimates[node, :] .+= node_means[node]
    end
    for (i, tip) in enumerate(tree.tip_ids)
        result.tip_estimates[i, :] .+= node_means[Int(tip)]
    end
    return result
end
