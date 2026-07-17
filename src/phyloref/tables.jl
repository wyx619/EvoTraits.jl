"""
    phyloref_node_table(tree; map = build_phyloref(tree))

Return a DataFrame translating EvoTraits internal node ids to the `ape` node
numbering convention where tips are `1:ntips` and internal nodes are numbered
after tips in cladewise / preorder-internal order.
"""
function phyloref_node_table(
    tree::CompactTree;
    map::PhyloRef = build_phyloref(tree),
)
    n = tree.nnodes
    ape_node_id = Vector{Int}(undef, n)
    is_tip = Vector{Bool}(undef, n)
    label = Vector{String}(undef, n)
    tipX = Vector{String}(undef, n)
    tipY = Vector{String}(undef, n)

    @inbounds for node in 1:n
        ape_node_id[node] = map.ape_node_ids[node]
        is_tip[node] = tree.is_tip[node]
        label[node] = tree.node_labels[node]
        tipX[node], tipY[node] = phylo_node_anchor(tree, node; map = map)
    end

    return DataFrame(
        evotraits_node_id = collect(1:n),
        ape_node_id = ape_node_id,
        is_tip = is_tip,
        label = label,
        tipX = tipX,
        tipY = tipY,
    )
end


"""
    phyloref_edge_table(tree; edges = nothing, map = build_phyloref(tree))

Return a DataFrame translating EvoTraits internal edge ids to branch identities
expressed through parent/child `ape` node ids, `ape` edge ranks in cladewise
and postorder orderings, two-tip anchors, and descendant-tip signatures.
"""
function phyloref_edge_table(
    tree::CompactTree;
    edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    map::PhyloRef = build_phyloref(tree),
)
    edge_ids = edges === nothing ? collect(1:tree.nedges) : Int.(edges)
    n = length(edge_ids)
    parent_node_id = Vector{Int}(undef, n)
    child_node_id = Vector{Int}(undef, n)
    ape_parent_node_id = Vector{Int}(undef, n)
    ape_child_node_id = Vector{Int}(undef, n)
    ape_cladewise_edge_rank = Vector{Int}(undef, n)
    ape_postorder_edge_rank = Vector{Int}(undef, n)
    branch_length = Vector{Float64}(undef, n)
    tipX = Vector{String}(undef, n)
    tipY = Vector{String}(undef, n)
    descendant_signature = Vector{String}(undef, n)

    @inbounds for (i, edge) in enumerate(edge_ids)
        1 <= edge <= tree.nedges || throw(ArgumentError("edge $edge is outside 1:$(tree.nedges)"))
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        parent_node_id[i] = parent
        child_node_id[i] = child
        ape_parent_node_id[i] = map.ape_node_ids[parent]
        ape_child_node_id[i] = map.ape_node_ids[child]
        ape_cladewise_edge_rank[i] = map.ape_cladewise_edge_ranks[edge]
        ape_postorder_edge_rank[i] = map.ape_postorder_edge_ranks[edge]
        branch_length[i] = tree.edge_length[edge]
        tipX[i], tipY[i] = phylo_branch_anchor(tree, edge; map = map)
        descendant_signature[i] = phylo_edge_signature(tree, edge; map = map)
    end

    return DataFrame(
        evotraits_edge_id = edge_ids,
        evotraits_parent_node_id = parent_node_id,
        evotraits_child_node_id = child_node_id,
        ape_parent_node_id = ape_parent_node_id,
        ape_child_node_id = ape_child_node_id,
        ape_cladewise_edge_rank = ape_cladewise_edge_rank,
        ape_postorder_edge_rank = ape_postorder_edge_rank,
        branch_length = branch_length,
        tipX = tipX,
        tipY = tipY,
        descendant_signature = descendant_signature,
    )
end


"""
    attach_R_node_map(df, tree; node_col = :node_id, map = build_phyloref(tree), prefix = "R_")

Return a copy of `df` with R-facing node translation columns attached for the
EvoTraits node ids stored in `node_col`.
"""
function attach_R_node_map(
    df::AbstractDataFrame,
    tree::CompactTree;
    node_col::Union{Symbol, AbstractString} = :node_id,
    map::PhyloRef = build_phyloref(tree),
    prefix::AbstractString = "R_",
)
    col = Symbol(node_col)
    col in propertynames(df) || throw(ArgumentError("node_col=$node_col is not present in the DataFrame"))
    out = DataFrame(df)
    nodes = Int.(out[!, col])

    R_node_vals = Vector{Int}(undef, length(nodes))
    is_tip_vals = Vector{Bool}(undef, length(nodes))
    label_vals = Vector{String}(undef, length(nodes))
    tipX_vals = Vector{String}(undef, length(nodes))
    tipY_vals = Vector{String}(undef, length(nodes))
    @inbounds for (i, node) in enumerate(nodes)
        1 <= node <= tree.nnodes || throw(ArgumentError("node id $node from column $node_col is outside 1:$(tree.nnodes)"))
        R_node_vals[i] = map.ape_node_ids[node]
        is_tip_vals[i] = tree.is_tip[node]
        label_vals[i] = tree.node_labels[node]
        tipX_vals[i], tipY_vals[i] = phylo_node_anchor(tree, node; map = map)
    end

    out[!, Symbol(prefix, "node_id")] = R_node_vals
    out[!, Symbol(prefix, "is_tip")] = is_tip_vals
    out[!, Symbol(prefix, "label")] = label_vals
    out[!, Symbol(prefix, "tipX")] = tipX_vals
    out[!, Symbol(prefix, "tipY")] = tipY_vals
    return out
end

"""
    attach_R_edge_map(df, tree; edge_col = :edge_id, order = :postorder, map = build_phyloref(tree), prefix = "R_")

Return a copy of `df` with R-facing edge translation columns attached for the
EvoTraits edge ids stored in `edge_col`.
"""
function attach_R_edge_map(
    df::AbstractDataFrame,
    tree::CompactTree;
    edge_col::Union{Symbol, AbstractString} = :edge_id,
    order::Symbol = :postorder,
    map::PhyloRef = build_phyloref(tree),
    prefix::AbstractString = "R_",
)
    col = Symbol(edge_col)
    col in propertynames(df) || throw(ArgumentError("edge_col=$edge_col is not present in the DataFrame"))
    out = DataFrame(df)
    edges = Int.(out[!, col])

    R_edge_vals = Vector{Int}(undef, length(edges))
    R_parent_vals = Vector{Int}(undef, length(edges))
    R_child_vals = Vector{Int}(undef, length(edges))
    branch_length_vals = Vector{Float64}(undef, length(edges))
    tipX_vals = Vector{String}(undef, length(edges))
    tipY_vals = Vector{String}(undef, length(edges))
    sig_vals = Vector{String}(undef, length(edges))
    @inbounds for (i, edge) in enumerate(edges)
        1 <= edge <= tree.nedges || throw(ArgumentError("edge id $edge from column $edge_col is outside 1:$(tree.nedges)"))
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        R_edge_vals[i] =
            order === :cladewise ? map.ape_cladewise_edge_ranks[edge] :
            order === :postorder ? map.ape_postorder_edge_ranks[edge] :
            throw(ArgumentError("Unsupported order=$order; expected :cladewise or :postorder"))
        R_parent_vals[i] = map.ape_node_ids[parent]
        R_child_vals[i] = map.ape_node_ids[child]
        branch_length_vals[i] = tree.edge_length[edge]
        tipX_vals[i], tipY_vals[i] = phylo_branch_anchor(tree, edge; map = map)
        sig_vals[i] = phylo_edge_signature(tree, edge; map = map)
    end

    out[!, Symbol(prefix, "edge_id")] = R_edge_vals
    out[!, Symbol(prefix, "parent_node_id")] = R_parent_vals
    out[!, Symbol(prefix, "child_node_id")] = R_child_vals
    out[!, :branch_length] = branch_length_vals
    out[!, :tipX] = tipX_vals
    out[!, :tipY] = tipY_vals
    out[!, :descendant_signature] = sig_vals
    return out
end
