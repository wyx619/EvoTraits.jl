"""
    estim_node_table(tree, asr; map = build_phyloref(tree))

Return a DataFrame view of a continuous ancestral-state reconstruction result
augmented with `phyloref` node translations. This keeps `estim_node` itself as
the pure computational API while exposing stable R/ape-facing node ids and tip
anchors for validation and downstream reporting.
"""
function estim_node_table(
    tree::CompactTree,
    asr::ContinuousASRResult;
    map = build_phyloref(tree),
    simmap::Union{Nothing, SimmapSample} = nothing,
)
    asr.success || throw(ArgumentError("estim_node_table requires a successful ASR result"))
    df = DataFrame(
        node_id = Int.(asr.node_ids),
        node_label = asr.node_labels,
        time_from_root = asr.time_from_root,
        time_before_present = asr.time_before_present,
        estimate = asr.estimates,
        variance = asr.variances,
        se = asr.se,
    )
    simmap === nothing || _attach_simmap_node_regime!(df, tree, asr.node_ids, simmap)
    out = attach_R_node_map(df, tree; node_col = :node_id, map = map)
    rename!(out, :R_tipX => :tipX, :R_tipY => :tipY)
    return out
end

function _mv_node_table_df(
    asr::MVContinuousASRResult,
    trait_names,
)
    asr.success || throw(ArgumentError("estim_node_table requires a successful ASR result"))
    n = length(asr.node_ids)
    p = size(asr.estimates, 2)
    df = DataFrame(
        node_id = Int.(asr.node_ids),
        node_label = asr.node_labels,
        time_from_root = asr.time_from_root,
        time_before_present = asr.time_before_present,
    )
    for j in 1:p
        suffix =
            trait_names === nothing ? string(j) :
            begin
                length(trait_names) == p || throw(ArgumentError("trait_names length must match multivariate trait dimension"))
                _mv_table_safe_name(trait_names[j])
            end
        df[!, Symbol("estimate_", suffix)] = asr.estimates[:, j]
        df[!, Symbol("variance_", suffix)] = [asr.node_covariances[i, j, j] for i in 1:n]
        df[!, Symbol("se_", suffix)] = sqrt.(max.(0.0, df[!, Symbol("variance_", suffix)]))
    end
    return df
end

function estim_node_table(
    tree::CompactTree,
    asr::MVContinuousASRResult;
    map = build_phyloref(tree),
    simmap::Union{Nothing, SimmapSample} = nothing,
)
    df = _mv_node_table_df(asr, nothing)
    simmap === nothing || _attach_simmap_node_regime!(df, tree, asr.node_ids, simmap)
    out = attach_R_node_map(df, tree; node_col = :node_id, map = map)
    rename!(out, :R_tipX => :tipX, :R_tipY => :tipY)
    return out
end

function estim_node_table(
    tree::CompactTree,
    asr::MVContinuousASRResult,
    fit::AbstractMVContinuousFitResult;
    map = build_phyloref(tree),
    simmap::Union{Nothing, SimmapSample} = nothing,
)
    p = size(asr.estimates, 2)
    df = _mv_node_table_df(asr, _result_trait_names(fit, p))
    simmap === nothing || _attach_simmap_node_regime!(df, tree, asr.node_ids, simmap)
    out = attach_R_node_map(df, tree; node_col = :node_id, map = map)
    rename!(out, :R_tipX => :tipX, :R_tipY => :tipY)
    return out
end

