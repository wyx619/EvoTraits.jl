function _mvoum_family_tree_pruning_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult,
    edge_segments::Vector{Vector{SimmapSegment}},
)
    data = _validate_multivariate_trait(tree, trait)
    fit.success || throw(ArgumentError("$(fit.model) ASR requires a successful fit"))
    fit.model in (:mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA) ||
        throw(ArgumentError("OUM-family ASR supports mvOUM, mvOUMV, mvOUMA, and mvOUMVA"))
    p = fit.ntraits
    bundle = _mvou_asr_bundle(fit)
    spec = _mvou_spec_with_root(
        fit.model;
        root_mean_mode = fit.root_mean_mode,
        root_cov_mode = fit.root_cov_mode,
    )
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    node_means = _mvou_asr_node_means(tree, fit, bundle, edge_segments, precalc.root_regime)

    centered = copy(data)
    for (i, tip) in enumerate(tree.tip_ids)
        centered[i, :] .-= node_means[Int(tip)]
    end

    branch = _mvou_asr_branch_cache(fit, precalc)
    asr = _mv_recursive_asr(
        tree,
        centered,
        branch.Phi,
        branch.Q;
        root_mean = zeros(Float64, p),
        root_cov = _mvou_root_covariance(precalc, bundle),
        shift = zeros(Float64, p),
        model = fit.model,
    )
    asr.success || return asr

    for (i, node) in enumerate(asr.node_ids)
        asr.estimates[i, :] .+= node_means[Int(node)]
    end
    for node in 1:tree.nnodes
        asr.all_node_estimates[node, :] .+= node_means[node]
    end
    for (i, tip) in enumerate(tree.tip_ids)
        asr.tip_estimates[i, :] .+= node_means[Int(tip)]
    end
    return asr
end
