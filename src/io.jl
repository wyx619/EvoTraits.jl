"""
    CompactTree

Internal array-based tree cache used by all likelihood, mapping, and ancestral
state reconstruction kernels. `CompactTree` stores topology, branch lengths,
tree traversals, tip labels, and node-label metadata in a form optimized for
large-tree computation.
"""
struct CompactTree
    ntips::Int
    nnodes::Int
    nedges::Int
    parent_of_edge::Vector{Int32}
    child_of_edge::Vector{Int32}
    edge_length::Vector{Float64}
    root::Int32
    parent_of_node::Vector{Int32}
    dist_from_root::Vector{Float64}
    is_tip::BitVector
    tip_ids::Vector{Int32}
    postorder::Vector{Int32}
    postorder_internal::Vector{Int32}
    preorder::Vector{Int32}
    children::Vector{Vector{Int32}}
    first_child_edge::Vector{Int32}
    last_child_edge::Vector{Int32}
    tipname_to_id::Dict{String, Int32}
    tip_labels::Vector{String}
    node_labels::Vector{String}
end

"""
    read_tree(path::AbstractString)

Load a tree from a Newick file.

This function intentionally reads the file contents first and then calls
`NewickTree.readnw` on the Newick string itself. This avoids path-handling
ambiguities on some platforms, especially with non-ASCII file names.

The return value is a `NewickTree.Node`. Use `serialize_tree` explicitly to
convert the parsed tree into the engine's `CompactTree` representation.

If the raw parse fails, the loader retries after stripping internal node labels
such as bootstrap/support values (`)100.0:` -> `):`). This keeps the engine
robust to large empirical trees whose support annotations are irrelevant to the
core likelihood kernels.
"""
function _read_newick_text(path::AbstractString)
    isfile(path) || throw(ArgumentError("Tree file does not exist: $path"))
    newick = String(strip(read(path, String)))
    endswith(newick, ';') || throw(ArgumentError("Malformed Newick file: missing trailing semicolon"))
    return newick
end

function _strip_internal_support_labels(newick::AbstractString)
    return String(replace(newick, r"\)([^():;,]+):" => "):"))
end

function _parse_newick_text(newick::AbstractString)
    try
        return NewickTree.readnw(newick)
    catch first_error
        cleaned = _strip_internal_support_labels(newick)
        cleaned == newick && rethrow(first_error)
        try
            return NewickTree.readnw(cleaned)
        catch
            rethrow(first_error)
        end
    end
end

function read_tree(path::AbstractString)
    newick = _read_newick_text(path)
    return _parse_newick_text(newick)
end

"""
    load_tree(path::AbstractString) -> CompactTree

Read a single Newick tree directly into the compact array representation used
by EvoTraits. This is the high-throughput tree-loading path for internal
analysis. It does not construct a `NewickTree.Node` intermediate.
"""
function load_tree(path::AbstractString)
    return _parse_load_tree_text(_read_newick_text(path))
end

@inline _load_tree_space(c::UInt8) = c == 0x20 || c == 0x09 || c == 0x0a || c == 0x0d

@inline function _load_tree_delimiter(c::UInt8)
    return c == UInt8('(') || c == UInt8(')') || c == UInt8(',') ||
           c == UInt8(':') || c == UInt8(';')
end

@inline function _load_tree_skip_space(bytes, i::Int, n::Int)
    while i <= n && _load_tree_space(bytes[i])
        i += 1
    end
    return i
end

function _load_tree_token(bytes, i::Int, n::Int)
    i = _load_tree_skip_space(bytes, i, n)
    start = i
    while i <= n && !_load_tree_delimiter(bytes[i])
        i += 1
    end
    stop = i - 1
    while stop >= start && _load_tree_space(bytes[stop])
        stop -= 1
    end
    return start <= stop ? String(bytes[start:stop]) : "", i
end

function _load_tree_branch_length(bytes, i::Int, n::Int)
    i = _load_tree_skip_space(bytes, i, n)
    (i > n || bytes[i] != UInt8(':')) && return NaN, i
    i = _load_tree_skip_space(bytes, i + 1, n)
    start = i
    while i <= n && !_load_tree_delimiter(bytes[i])
        i += 1
    end
    stop = i - 1
    while stop >= start && _load_tree_space(bytes[stop])
        stop -= 1
    end
    start <= stop || throw(ArgumentError("Malformed Newick branch length"))
    text = String(bytes[start:stop])
    value = tryparse(Float64, text)
    value === nothing && throw(ArgumentError("Malformed Newick branch length"))
    return value, i
end

@inline function _load_tree_internal_label(label::String)
    isempty(label) && return label
    tryparse(Float64, label) === nothing ? label : ""
end

function _load_tree_attach_child!(children::Vector{Vector{Int32}}, parent::Int32, child::Int32)
    index = Int(parent)
    isassigned(children, index) || (children[index] = Int32[])
    push!(children[index], child)
    return nothing
end

function _parse_load_tree_text(newick::AbstractString)
    bytes = codeunits(newick)
    n = length(bytes)
    n > 0 || throw(ArgumentError("Malformed Newick tree: empty input"))

    # A valid single Newick tree has one more tip than commas and one internal
    # node per closing parenthesis. These counts let the parser allocate its
    # scalar arrays exactly once.
    ncommas = 0
    nclosed = 0
    for byte in bytes
        byte == UInt8(',') && (ncommas += 1)
        byte == UInt8(')') && (nclosed += 1)
    end
    nnodes = ncommas + nclosed + 1
    ntips_expected = ncommas + 1

    parent_of_node = Vector{Int32}(undef, nnodes)
    node_edge_length = fill(NaN, nnodes)
    node_labels = fill("", nnodes)
    is_tip = BitVector(undef, nnodes)
    children = Vector{Vector{Int32}}(undef, nnodes)
    stack = Vector{Int32}(undef, max(nclosed, 1))
    stack_depth = 0
    node_count = 0
    tip_count = 0
    root = Int32(0)
    i = 1

    while true
        i = _load_tree_skip_space(bytes, i, n)
        i <= n || throw(ArgumentError("Malformed Newick tree: missing semicolon"))
        c = bytes[i]

        if c == UInt8('(')
            node_count += 1
            node_count <= nnodes || throw(ArgumentError("Malformed Newick tree"))
            idx = Int32(node_count)
            parent = stack_depth == 0 ? Int32(0) : stack[stack_depth]
            parent_of_node[node_count] = parent
            is_tip[node_count] = false
            root == 0 && (root = idx)
            parent == 0 || _load_tree_attach_child!(children, parent, idx)
            stack_depth += 1
            stack[stack_depth] = idx
            i += 1
        elseif c == UInt8(',')
            stack_depth > 0 || throw(ArgumentError("Malformed Newick tree: unexpected comma"))
            i += 1
        elseif c == UInt8(')')
            stack_depth > 0 || throw(ArgumentError("Malformed Newick tree: unbalanced parentheses"))
            idx = stack[stack_depth]
            stack_depth -= 1
            label, i = _load_tree_token(bytes, i + 1, n)
            node_labels[Int(idx)] = _load_tree_internal_label(label)
            node_edge_length[Int(idx)], i = _load_tree_branch_length(bytes, i, n)
        elseif c == UInt8(';')
            root != 0 || throw(ArgumentError("Malformed Newick tree: missing root"))
            stack_depth == 0 || throw(ArgumentError("Malformed Newick tree: unbalanced parentheses"))
            i = _load_tree_skip_space(bytes, i + 1, n)
            i > n || throw(ArgumentError("Multiple trees are not supported"))
            break
        elseif c == UInt8(':')
            throw(ArgumentError("Malformed Newick tree: unexpected branch length"))
        else
            label, next_i = _load_tree_token(bytes, i, n)
            isempty(label) && throw(ArgumentError("Malformed Newick tree: empty tip label"))
            parent = stack_depth == 0 ? Int32(0) : stack[stack_depth]
            parent == 0 && node_count != 0 && throw(ArgumentError("Malformed Newick tree: multiple roots"))
            node_count += 1
            node_count <= nnodes || throw(ArgumentError("Malformed Newick tree"))
            idx = Int32(node_count)
            parent_of_node[node_count] = parent
            is_tip[node_count] = true
            node_labels[node_count] = label
            node_edge_length[node_count], next_i = _load_tree_branch_length(bytes, next_i, n)
            root == 0 && (root = idx)
            parent == 0 || _load_tree_attach_child!(children, parent, idx)
            tip_count += 1
            i = next_i
        end
    end

    node_count == nnodes || throw(ArgumentError("Malformed Newick tree: incomplete tree"))
    tip_count == ntips_expected || throw(ArgumentError("Malformed Newick tree: invalid tip count"))

    for node in 1:nnodes
        isassigned(children, node) || (children[node] = Int32[])
        isempty(children[node]) && !is_tip[node] &&
            throw(ArgumentError("Malformed Newick tree: internal node without children"))
    end
    node_edge_length[Int(root)] = 0.0

    dist_from_root = zeros(Float64, nnodes)
    for node in 2:nnodes
        parent = parent_of_node[node]
        parent != 0 || throw(ArgumentError("Malformed Newick tree: multiple roots"))
        dist_from_root[node] = dist_from_root[Int(parent)] + node_edge_length[node]
    end

    preorder = Vector{Int32}(undef, nnodes)
    for node in 1:nnodes
        preorder[node] = Int32(node)
    end

    postorder = Vector{Int32}(undef, nnodes)
    postorder_internal = Vector{Int32}(undef, nnodes - tip_count)
    postorder_pos = 0
    internal_pos = 0
    traversal_nodes = Int32[root]
    traversal_next_child = Int[1]
    while !isempty(traversal_nodes)
        node = Int(last(traversal_nodes))
        next_child = traversal_next_child[end]
        if next_child <= length(children[node])
            traversal_next_child[end] += 1
            push!(traversal_nodes, children[node][next_child])
            push!(traversal_next_child, 1)
        else
            postorder_pos += 1
            postorder[postorder_pos] = Int32(node)
            if !is_tip[node]
                internal_pos += 1
                postorder_internal[internal_pos] = Int32(node)
            end
            pop!(traversal_nodes)
            pop!(traversal_next_child)
        end
    end

    ntips = tip_count
    tip_ids = Vector{Int32}(undef, ntips)
    tip_pos = 0
    for node in 1:nnodes
        if is_tip[node]
            tip_pos += 1
            tip_ids[tip_pos] = Int32(node)
        end
    end
    tip_labels = Vector{String}(undef, ntips)
    for tip in 1:ntips
        tip_labels[tip] = node_labels[Int(tip_ids[tip])]
    end
    tipname_to_id = Dict{String, Int32}(tip_labels[i] => tip_ids[i] for i in eachindex(tip_labels))

    nedges = nnodes - 1
    parent_of_edge = Vector{Int32}(undef, nedges)
    child_of_edge = Vector{Int32}(undef, nedges)
    edge_length = Vector{Float64}(undef, nedges)
    first_child_edge = fill(Int32(0), nnodes)
    last_child_edge = fill(Int32(0), nnodes)
    edge_idx = Int32(1)
    for parent in 1:nnodes
        for child in children[parent]
            parent_of_edge[Int(edge_idx)] = Int32(parent)
            child_of_edge[Int(edge_idx)] = child
            edge_length[Int(edge_idx)] = node_edge_length[Int(child)]
            first_child_edge[parent] == 0 && (first_child_edge[parent] = edge_idx)
            last_child_edge[parent] = edge_idx
            edge_idx += 1
        end
    end

    return CompactTree(
        ntips,
        nnodes,
        nedges,
        parent_of_edge,
        child_of_edge,
        edge_length,
        root,
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
end

"""
    serialize_tree(tree::NewickTree.Node)

Convert a parsed `NewickTree` tree into the engine's internal `CompactTree`
representation. This is the canonical ingest boundary for external tree input.
"""
function serialize_tree(tree::T) where {T <: NewickTree.Node}
    root_node = tree
    while !NewickTree.isroot(root_node)
        root_node = root_node.parent
    end

    # Count first so the conversion pass can fill fixed-size arrays. This
    # avoids repeated growth and temporary child vectors for every node.
    nnodes = 0
    ntips = 0
    child_counts = Int[]
    count_stack = T[root_node]
    while !isempty(count_stack)
        node = pop!(count_stack)
        nnodes += 1
        if !isdefined(node, :children)
            push!(child_counts, 0)
            ntips += 1
            continue
        end
        raw_children = node.children
        degree = length(raw_children)
        push!(child_counts, degree)
        degree == 0 && (ntips += 1)
        for child in Iterators.reverse(raw_children)
            push!(count_stack, child)
        end
    end

    parent_of_node = Vector{Int32}(undef, nnodes)
    children = Vector{Vector{Int32}}(undef, nnodes)
    for node in 1:nnodes
        children[node] = Vector{Int32}(undef, child_counts[node])
    end
    node_edge_length = Vector{Float64}(undef, nnodes)
    is_tip = BitVector(undef, nnodes)
    node_labels = Vector{String}(undef, nnodes)
    preorder = Vector{Int32}(undef, nnodes)
    child_positions = zeros(Int, nnodes)
    stack = Tuple{T, Int32}[(root_node, 0)]
    node_index = 0
    while !isempty(stack)
        node, parent_idx = pop!(stack)

        node_index += 1
        idx = Int32(node_index)
        preorder[node_index] = idx
        parent_of_node[node_index] = parent_idx
        node_edge_length[node_index] = NewickTree.isroot(node) ? 0.0 : Float64(NewickTree.distance(node))
        node_labels[node_index] = String(NewickTree.name(node))
        is_tip[node_index] = child_counts[node_index] == 0

        if parent_idx != 0
            parent = Int(parent_idx)
            child_positions[parent] += 1
            children[parent][child_positions[parent]] = idx
        end

        if isdefined(node, :children)
            for child in Iterators.reverse(node.children)
                push!(stack, (child, idx))
            end
        end
    end

    root = Int32(1)

    dist_from_root = zeros(Float64, nnodes)
    for node in preorder
        for child in children[Int(node)]
            dist_from_root[Int(child)] = dist_from_root[Int(node)] + node_edge_length[Int(child)]
        end
    end

    postorder = Vector{Int32}(undef, nnodes)
    postorder_internal = Vector{Int32}(undef, nnodes - ntips)
    postorder_pos = 0
    internal_pos = 0
    traversal_nodes = Int32[root]
    traversal_next_child = Int[1]
    while !isempty(traversal_nodes)
        node = Int(last(traversal_nodes))
        next_child = traversal_next_child[end]
        if next_child <= length(children[node])
            traversal_next_child[end] += 1
            push!(traversal_nodes, children[node][next_child])
            push!(traversal_next_child, 1)
        else
            postorder_pos += 1
            postorder[postorder_pos] = Int32(node)
            if !is_tip[node]
                internal_pos += 1
                postorder_internal[internal_pos] = Int32(node)
            end
            pop!(traversal_nodes)
            pop!(traversal_next_child)
        end
    end

    tip_ids = Vector{Int32}(undef, ntips)
    tip_pos = 0
    for node in 1:nnodes
        if is_tip[node]
            tip_pos += 1
            tip_ids[tip_pos] = Int32(node)
        end
    end
    nedges = nnodes - 1
    tip_labels = [node_labels[idx] for idx in tip_ids]
    tipname_to_id = Dict{String, Int32}(tip_labels[i] => tip_ids[i] for i in eachindex(tip_ids))

    parent_of_edge = Vector{Int32}(undef, nedges)
    child_of_edge = Vector{Int32}(undef, nedges)
    edge_length = Vector{Float64}(undef, nedges)
    first_child_edge = fill(Int32(0), nnodes)
    last_child_edge = fill(Int32(0), nnodes)

    edge_idx = Int32(1)
    for parent in Int32.(1:nnodes)
        for child in children[parent]
            parent_of_edge[edge_idx] = parent
            child_of_edge[edge_idx] = child
            edge_length[edge_idx] = node_edge_length[child]
            if first_child_edge[parent] == 0
                first_child_edge[parent] = edge_idx
            end
            last_child_edge[parent] = edge_idx
            edge_idx += 1
        end
    end

    return CompactTree(
        ntips,
        nnodes,
        nedges,
        parent_of_edge,
        child_of_edge,
        edge_length,
        root,
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
end
