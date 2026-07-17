"""
    estim_branch_table(tree, branch; map = build_phyloref(tree), R_order = :postorder)

Return a DataFrame view of a simmap-aware branch posterior reconstruction
augmented with `phyloref` edge translations. Table output is standardized on
`time_from_root_*` and `time_before_present_*` and does not expose the legacy
`absolute_*` aliases.
"""
function estim_branch_table(
    tree::CompactTree,
    branch::ContinuousBranchPosteriorResult;
    map = build_phyloref(tree),
    R_order::Symbol = :postorder,
)
    branch.success || throw(ArgumentError("estim_branch_table requires a successful branch posterior result"))
    df = DataFrame(
        edge_id = Int.(branch.edge_ids),
        parent_node_id = Int.(branch.parent_nodes),
        child_node_id = Int.(branch.child_nodes),
        segment_index = Int.(branch.segment_indices),
        segment_state = Int.(branch.segment_states),
        branch_start = branch.branch_start,
        branch_end = branch.branch_end,
        branch_midpoint = branch.branch_midpoint,
        time_from_root_start = branch.time_from_root_start,
        time_from_root_end = branch.time_from_root_end,
        time_from_root_midpoint = branch.time_from_root_midpoint,
        time_before_present_start = branch.time_before_present_start,
        time_before_present_end = branch.time_before_present_end,
        time_before_present_midpoint = branch.time_before_present_midpoint,
        start_estimate = branch.start_estimates,
        midpoint_estimate = branch.midpoint_estimates,
        end_estimate = branch.end_estimates,
        start_variance = branch.start_variances,
        midpoint_variance = branch.midpoint_variances,
        end_variance = branch.end_variances,
        start_se = branch.start_se,
        midpoint_se = branch.midpoint_se,
        end_se = branch.end_se,
    )
    out = attach_R_edge_map(df, tree; edge_col = :edge_id, order = R_order, map = map)
    rename!(out, :R_edge_id => :edge_R_id, :R_parent_node_id => :parent_R_node_id, :R_child_node_id => :child_R_node_id)
    return out
end

function _mv_branch_table_df(
    branch::MVContinuousBranchPosteriorResult,
    trait_names,
)
    branch.success || throw(ArgumentError("estim_branch_table requires a successful branch posterior result"))
    n = length(branch.edge_ids)
    p = size(branch.start_estimates, 2)
    df = DataFrame(
        edge_id = Int.(branch.edge_ids),
        parent_node_id = Int.(branch.parent_nodes),
        child_node_id = Int.(branch.child_nodes),
        segment_index = Int.(branch.segment_indices),
        segment_state = Int.(branch.segment_states),
        branch_start = branch.branch_start,
        branch_end = branch.branch_end,
        branch_midpoint = branch.branch_midpoint,
        time_from_root_start = branch.time_from_root_start,
        time_from_root_end = branch.time_from_root_end,
        time_from_root_midpoint = branch.time_from_root_midpoint,
        time_before_present_start = branch.time_before_present_start,
        time_before_present_end = branch.time_before_present_end,
        time_before_present_midpoint = branch.time_before_present_midpoint,
    )
    for j in 1:p
        suffix =
            trait_names === nothing ? string(j) :
            begin
                length(trait_names) == p || throw(ArgumentError("trait_names length must match multivariate trait dimension"))
                _mv_table_safe_name(trait_names[j])
            end
        df[!, Symbol("start_estimate_", suffix)] = branch.start_estimates[:, j]
        df[!, Symbol("midpoint_estimate_", suffix)] = branch.midpoint_estimates[:, j]
        df[!, Symbol("end_estimate_", suffix)] = branch.end_estimates[:, j]
        df[!, Symbol("start_variance_", suffix)] = [branch.start_covariances[i, j, j] for i in 1:n]
        df[!, Symbol("midpoint_variance_", suffix)] = [branch.midpoint_covariances[i, j, j] for i in 1:n]
        df[!, Symbol("end_variance_", suffix)] = [branch.end_covariances[i, j, j] for i in 1:n]
        df[!, Symbol("start_se_", suffix)] = sqrt.(max.(0.0, df[!, Symbol("start_variance_", suffix)]))
        df[!, Symbol("midpoint_se_", suffix)] = sqrt.(max.(0.0, df[!, Symbol("midpoint_variance_", suffix)]))
        df[!, Symbol("end_se_", suffix)] = sqrt.(max.(0.0, df[!, Symbol("end_variance_", suffix)]))
    end
    return df
end

function estim_branch_table(
    tree::CompactTree,
    branch::MVContinuousBranchPosteriorResult;
    map = build_phyloref(tree),
    R_order::Symbol = :postorder,
)
    df = _mv_branch_table_df(branch, nothing)
    out = attach_R_edge_map(df, tree; edge_col = :edge_id, order = R_order, map = map)
    rename!(out, :R_edge_id => :edge_R_id, :R_parent_node_id => :parent_R_node_id, :R_child_node_id => :child_R_node_id)
    return out
end

function estim_branch_table(
    tree::CompactTree,
    branch::MVContinuousBranchPosteriorResult,
    fit::AbstractMVContinuousFitResult;
    map = build_phyloref(tree),
    R_order::Symbol = :postorder,
)
    p = size(branch.start_estimates, 2)
    df = _mv_branch_table_df(branch, _result_trait_names(fit, p))
    out = attach_R_edge_map(df, tree; edge_col = :edge_id, order = R_order, map = map)
    rename!(out, :R_edge_id => :edge_R_id, :R_parent_node_id => :parent_R_node_id, :R_child_node_id => :child_R_node_id)
    return out
end

