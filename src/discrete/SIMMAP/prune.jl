abstract type _AbstractPrunedSimmapBranch end

mutable struct _PrunedSimmapNode
    label::String
    branches::Vector{_AbstractPrunedSimmapBranch}
end

struct _PrunedSimmapBranch <: _AbstractPrunedSimmapBranch
    segments::Vector{SimmapSegment}
    child::_PrunedSimmapNode
end

function _merge_simmap_segments(a::Vector{SimmapSegment}, b::Vector{SimmapSegment})
    isempty(a) && return copy(b)
    isempty(b) && return copy(a)
    out = copy(a)
    for seg in b
        if !isempty(out) && out[end].state == seg.state
            out[end] = SimmapSegment(state = seg.state, length = out[end].length + seg.length)
        else
            push!(out, seg)
        end
    end
    return out
end

function _simmap_edge_for_child(tree::CompactTree, parent::Integer, child::Integer)
    first_edge = Int(tree.first_child_edge[parent])
    last_edge = Int(tree.last_child_edge[parent])
    for edge in first_edge:last_edge
        tree.child_of_edge[edge] == child && return edge
    end
    throw(ArgumentError("No edge found for parent=$parent child=$child"))
end

function _prune_simmap_edge(tree::CompactTree, simmap::SimmapSample, edge::Integer, keep::Set{String})
    child = Int(tree.child_of_edge[edge])
    base = copy(simmap.edge_segments[edge])
    if tree.is_tip[child]
        label = tree.node_labels[child]
        label in keep || return nothing
        return _PrunedSimmapBranch(base, _PrunedSimmapNode(label, _AbstractPrunedSimmapBranch[]))
    end

    branches = _AbstractPrunedSimmapBranch[]
    for grandchild in tree.children[child]
        grand_edge = _simmap_edge_for_child(tree, child, grandchild)
        branch = _prune_simmap_edge(tree, simmap, grand_edge, keep)
        branch !== nothing && push!(branches, branch)
    end
    isempty(branches) && return nothing

    if length(branches) == 1
        only_branch = branches[1]::_PrunedSimmapBranch
        return _PrunedSimmapBranch(_merge_simmap_segments(base, only_branch.segments), only_branch.child)
    end

    node = _PrunedSimmapNode(tree.node_labels[child], branches)
    return _PrunedSimmapBranch(base, node)
end

function _build_pruned_simmap_root(tree::CompactTree, simmap::SimmapSample, keep::Set{String})
    branches = _AbstractPrunedSimmapBranch[]
    for child in tree.children[Int(tree.root)]
        edge = _simmap_edge_for_child(tree, Int(tree.root), child)
        branch = _prune_simmap_edge(tree, simmap, edge, keep)
        branch !== nothing && push!(branches, branch)
    end
    isempty(branches) && throw(ArgumentError("Cannot drop all tips from a simmap tree"))

    root = _PrunedSimmapNode(tree.node_labels[Int(tree.root)], branches)
    while length(root.branches) == 1 && !isempty((root.branches[1]::_PrunedSimmapBranch).child.branches)
        root = (root.branches[1]::_PrunedSimmapBranch).child
    end
    return root
end

function _compact_tree_from_pruned_root(root::_PrunedSimmapNode)
    parent_of_node = Int32[]
    children = Vector{Int32}[]
    node_edge_length = Float64[]
    is_tip = BitVector()
    node_labels = String[]
    segment_by_edge = Dict{Tuple{Int32, Int32}, Vector{SimmapSegment}}()

    function add_node!(node::_PrunedSimmapNode, parent::Int32, incoming::Union{Nothing, Vector{SimmapSegment}})
        idx = Int32(length(parent_of_node) + 1)
        push!(parent_of_node, parent)
        push!(children, Int32[])
        push!(node_edge_length, incoming === nothing || isempty(incoming) ? 0.0 : sum(seg.length for seg in incoming))
        push!(is_tip, isempty(node.branches))
        push!(node_labels, node.label)
        if parent != 0
            push!(children[parent], idx)
            segment_by_edge[(parent, idx)] = incoming === nothing ? SimmapSegment[] : incoming
        end
        for raw_branch in node.branches
            branch = raw_branch::_PrunedSimmapBranch
            add_node!(branch.child, idx, branch.segments)
        end
        return idx
    end

    root_id = add_node!(root, Int32(0), nothing)
    nnodes = length(parent_of_node)
    dist_from_root = zeros(Float64, nnodes)
    preorder = Int32[]
    stack = Int32[root_id]
    while !isempty(stack)
        node = pop!(stack)
        push!(preorder, node)
        for child in Iterators.reverse(children[node])
            dist_from_root[child] = dist_from_root[node] + node_edge_length[child]
            push!(stack, child)
        end
    end

    postorder = Int32[]
    postorder_internal = Int32[]
    function dfs(node::Int32)
        for child in children[node]
            dfs(child)
        end
        push!(postorder, node)
        if !is_tip[node]
            push!(postorder_internal, node)
        end
        return nothing
    end
    dfs(root_id)

    tip_ids = Int32[findall(is_tip)...]
    ntips = length(tip_ids)
    nedges = nnodes - 1
    tip_labels = [node_labels[idx] for idx in tip_ids]
    tipname_to_id = Dict{String, Int32}(tip_labels[i] => tip_ids[i] for i in eachindex(tip_ids))
    parent_of_edge = Vector{Int32}(undef, nedges)
    child_of_edge = Vector{Int32}(undef, nedges)
    edge_length = Vector{Float64}(undef, nedges)
    first_child_edge = fill(Int32(0), nnodes)
    last_child_edge = fill(Int32(0), nnodes)

    edge_idx = Int32(1)
    ordered_segments = Vector{Vector{SimmapSegment}}(undef, nedges)
    for parent in Int32.(1:nnodes)
        for child in children[parent]
            parent_of_edge[edge_idx] = parent
            child_of_edge[edge_idx] = child
            edge_length[edge_idx] = node_edge_length[child]
            ordered_segments[edge_idx] = segment_by_edge[(parent, child)]
            if first_child_edge[parent] == 0
                first_child_edge[parent] = edge_idx
            end
            last_child_edge[parent] = edge_idx
            edge_idx += 1
        end
    end

    compact = CompactTree(
        ntips,
        nnodes,
        nedges,
        parent_of_edge,
        child_of_edge,
        edge_length,
        root_id,
        parent_of_node,
        dist_from_root,
        is_tip,
        tip_ids,
        postorder,
        postorder_internal,
        preorder,
        children,
        first_child_edge,
        last_child_edge,
        tipname_to_id,
        tip_labels,
        node_labels,
    )
    return compact, ordered_segments
end

function _simmap_sample_from_segments(tree::CompactTree, template::SimmapSample, edge_segments::Vector{Vector{SimmapSegment}})
    nstates = max(template.nstates, length(template.state_labels))
    for segments in edge_segments, seg in segments
        nstates = max(nstates, Int(seg.state))
    end
    state_labels = _simmap_state_labels(template, nstates)
    mapped_edge = zeros(Float64, tree.nedges, nstates)
    for edge in 1:tree.nedges
        for seg in edge_segments[edge]
            1 <= seg.state <= nstates && (mapped_edge[edge, Int(seg.state)] += seg.length)
        end
    end

    node_states = fill(Int32(0), tree.nnodes)
    edge_start_states = fill(Int32(0), tree.nedges)
    edge_end_states = fill(Int32(0), tree.nedges)
    for edge in 1:tree.nedges
        segments = edge_segments[edge]
        isempty(segments) && continue
        parent = Int(tree.parent_of_edge[edge])
        child = Int(tree.child_of_edge[edge])
        edge_start_states[edge] = segments[1].state
        edge_end_states[edge] = segments[end].state
        node_states[parent] == 0 && (node_states[parent] = segments[1].state)
        node_states[child] = segments[end].state
    end
    root_state = node_states[Int(tree.root)] == 0 ? template.root_state : node_states[Int(tree.root)]
    node_states[Int(tree.root)] = root_state

    return SimmapSample(
        success = true,
        nstates = nstates,
        root_state = root_state,
        state_labels = state_labels,
        node_states = node_states,
        edge_start_states = edge_start_states,
        edge_end_states = edge_end_states,
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
        loglik = NaN,
    )
end

function _normalize_tip_set(tree::CompactTree, tips)
    tip_vec =
        if tips isa AbstractString
            String[tips]
        else
            String.(collect(tips))
        end
    unknown = setdiff(tip_vec, tree.tip_labels)
    isempty(unknown) || throw(ArgumentError("Unknown tip labels: $(join(unknown, ", "))"))
    return Set(tip_vec)
end

"""
    keep_tip_simmap(tree, simmap, tips)

Return a pruned `(tree, simmap)` pair containing only `tips`. The topology is
compressed after pruning, and SIMMAP segments are merged across collapsed
degree-2 nodes while preserving mapped branch histories.
"""
function keepmap(tree::CompactTree, simmap::SimmapSample, tips)
    length(simmap.edge_segments) == tree.nedges || throw(ArgumentError("simmap.edge_segments must match tree.nedges"))
    keep = _normalize_tip_set(tree, tips)
    isempty(keep) && throw(ArgumentError("Cannot keep zero tips from a simmap tree"))
    length(keep) == tree.ntips && return (tree = tree, simmap = simmap)
    root = _build_pruned_simmap_root(tree, simmap, keep)
    pruned_tree, edge_segments = _compact_tree_from_pruned_root(root)
    pruned_simmap = _simmap_sample_from_segments(pruned_tree, simmap, edge_segments)
    return (tree = pruned_tree, simmap = pruned_simmap)
end

"""
    drop_tip_simmap(tree, simmap, tips)

Return a pruned `(tree, simmap)` pair after removing `tips`. This is the
EvoTraits-native counterpart of phytools' `drop.tip.simmap`.
"""
function dropmap(tree::CompactTree, simmap::SimmapSample, tips)
    drop = _normalize_tip_set(tree, tips)
    keep = setdiff(Set(tree.tip_labels), drop)
    isempty(keep) && throw(ArgumentError("Cannot drop all tips from a simmap tree"))
    return keepmap(tree, simmap, keep)
end

"""
    keep_tip(tree::CompactTree, tips)

Return a pruned `CompactTree` containing only `tips`. The topology is compressed
after pruning (degree-2 nodes are collapsed).

This is the tree-only counterpart of `keep_tip_simmap`.
"""
function keep_tip(tree::CompactTree, tips)
    keep = _normalize_tip_set(tree, tips)
    isempty(keep) && throw(ArgumentError("Cannot keep zero tips from a tree"))
    length(keep) == tree.ntips && return tree

    dummy_segments = [[SimmapSegment(state = Int32(1), length = tree.edge_length[edge])] for edge in 1:tree.nedges]
    dummy_simmap = SimmapSample(
        nstates = 1,
        edge_segments = dummy_segments,
        state_labels = ["1"],
        mapped_edge = zeros(Float64, tree.nedges, 1),
    )
    root = _build_pruned_simmap_root(tree, dummy_simmap, keep)
    pruned_tree, _ = _compact_tree_from_pruned_root(root)
    return pruned_tree
end

"""
    drop_tip(tree::CompactTree, tips)

Return a pruned `CompactTree` after removing `tips`. This is the tree-only
counterpart of `drop_tip_simmap`.
"""
function drop_tip(tree::CompactTree, tips)
    drop = _normalize_tip_set(tree, tips)
    keep = setdiff(Set(tree.tip_labels), drop)
    isempty(keep) && throw(ArgumentError("Cannot drop all tips from a tree"))
    return keep_tip(tree, collect(keep))
end

keep_tip_simmap(args...; kwargs...) = keepmap(args...; kwargs...)
drop_tip_simmap(args...; kwargs...) = dropmap(args...; kwargs...)
