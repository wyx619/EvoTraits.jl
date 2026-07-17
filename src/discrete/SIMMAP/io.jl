@inline function _state_label(simmap::SimmapSample, state::Int32)
    if 1 <= state <= length(simmap.state_labels)
        return simmap.state_labels[state]
    end
    return string(state)
end

@inline function _map_order_symbol(map_order)
    if map_order isa Symbol
        return map_order
    elseif lowercase(String(map_order)) in ("right-to-left", "right_to_left", "r", "right")
        return :right_to_left
    elseif lowercase(String(map_order)) in ("left-to-right", "left_to_right", "l", "left")
        return :left_to_right
    end
    throw(ArgumentError("Unsupported map_order=$map_order"))
end

@inline function _segment_sequence_for_output(segments::Vector{SimmapSegment}, map_order::Symbol)
    return map_order === :right_to_left ? reverse(segments) : segments
end

function _write_branch_v1(simmap::SimmapSample, segments::Vector{SimmapSegment}, map_order::Symbol)
    seq = _segment_sequence_for_output(segments, map_order)
    parts = [string(_state_label(simmap, seg.state), ",", round(seg.length; digits = 8)) for seg in seq]
    return ":{".*join(parts, ":").*"}"
end

function _write_branch_v2(simmap::SimmapSample, segments::Vector{SimmapSegment}, total_length::Float64, map_order::Symbol)
    seq = _segment_sequence_for_output(segments, map_order)
    if length(seq) == 1
        body = _state_label(simmap, seq[1].state)
    else
        parts = String[]
        for i in 1:(length(seq) - 1)
            push!(parts, string(_state_label(simmap, seq[i].state), ",", round(seq[i].length; digits = 8)))
        end
        push!(parts, _state_label(simmap, seq[end].state))
        body = join(parts, ",")
    end
    return ":[&map={".*body.*"}]".*string(round(total_length; digits = 8))
end

function _children_edge_pairs(tree::CompactTree, node::Int)
    first_edge = Int(tree.first_child_edge[node])
    last_edge = Int(tree.last_child_edge[node])
    return [(Int(tree.child_of_edge[edge]), edge) for edge in first_edge:last_edge]
end

function _write_subtree_simmap(tree::CompactTree, simmap::SimmapSample, node::Int, map_order::Symbol, version::Float64)
    if tree.is_tip[node]
        return tree.tip_labels[findfirst(==(Int32(node)), tree.tip_ids)]
    end
    child_strings = String[]
    for (child, edge) in _children_edge_pairs(tree, node)
        child_text = _write_subtree_simmap(tree, simmap, child, map_order, version)
        branch_text =
            if version <= 1.0
                _write_branch_v1(simmap, simmap.edge_segments[edge], map_order)
            else
                _write_branch_v2(simmap, simmap.edge_segments[edge], tree.edge_length[edge], map_order)
            end
        push!(child_strings, child_text * branch_text)
    end
    return "(" * join(child_strings, ",") * ")"
end

"""
    write_simmap(tree, simmap; kwargs...)

Serialize a `SimmapSample` to `phytools`-style PHYLIP or NEXUS text. When
`file` is provided, the serialized representation is also written to disk.
"""
function write_simmap(
    tree::CompactTree,
    simmap::SimmapSample;
    file::Union{Nothing, AbstractString} = nothing,
    format::Symbol = :phylip,
    version::Float64 = 1.0,
    map_order = :right_to_left,
)
    order = _map_order_symbol(map_order)
    !isempty(simmap.state_labels) || throw(ArgumentError("write_simmap requires simmap.state_labels to be populated"))
    length(simmap.state_labels) >= simmap.nstates || throw(ArgumentError("write_simmap requires at least nstates state_labels"))
    text = _write_subtree_simmap(tree, simmap, Int(tree.root), order, version) * ";"
    if format === :phylip
        if file !== nothing
            write(file, text * "\n")
        end
        return text
    elseif format === :nexus
        labels = tree.tip_labels
        translation = join(["\t\t$(i)\t$(labels[i])" * (i < length(labels) ? "," : "") for i in eachindex(labels)], "\n")
        translated = text
        for i in reverse(eachindex(labels))
            translated = replace(translated, labels[i] => string(i))
        end
        block = join([
            "#NEXUS",
            "BEGIN TAXA;",
            "\tDIMENSIONS NTAX = $(length(labels));",
            "\tTAXLABELS",
            join(["\t\t" * label for label in labels], "\n"),
            "\t;",
            "END;",
            version <= 1.0 ? "BEGIN SMPTREES;" : "BEGIN TREES;",
            "\tTRANSLATE",
            translation,
            "\t;",
            "\tTREE * UNTITLED = [&R] " * translated,
            "END;",
        ], "\n")
        if file !== nothing
            write(file, block * "\n")
        end
        return block
    end
    throw(ArgumentError("Unsupported format=$format"))
end

function _extract_tree_strings_from_nexus(content::String)
    lines = split(content, '\n')
    trans = Dict{String, String}()
    in_translate = false
    tree_strings = String[]
    for raw in lines
        line = strip(raw)
        isempty(line) && continue
        if occursin(r"(?i)^translate", line)
            in_translate = true
            continue
        end
        if in_translate
            if line == ";"
                in_translate = false
                continue
            end
            m = match(r"^([^ \t,]+)\s+(.+?)(,)?$", line)
            if m !== nothing
                key = strip(m.captures[1], ['"', '\''])
                val = strip(m.captures[2], ['"', '\'', ','])
                trans[key] = val
            end
            continue
        end
        if occursin(r"(?i)^tree\b", line)
            tree_text = replace(line, r"^[^=]*=\s*" => "")
            tree_text = replace(tree_text, r"^\s*\[&R\]\s*" => "")
            push!(tree_strings, tree_text)
        end
    end
    return tree_strings, trans
end

@inline function _detect_simmap_version(text::AbstractString, fallback::Float64)
    if occursin("[&map={", text)
        return 1.5
    elseif occursin(":{", text)
        return 1.0
    end
    return fallback
end

function _looks_like_missing_path(source::AbstractString)
    s = String(source)
    occursin(r"^[A-Za-z]:[\\/]", s) && return true
    occursin('\\', s) && return true
    occursin('/', s) && return !occursin('(', s)
    return false
end

function _parse_label(text::String, i::Int)
    start = i
    while i <= lastindex(text) && !(text[i] in (':', ',', ')', ';'))
        i += 1
    end
    return text[start:(i - 1)], i
end

function _read_number(text::String, i::Int)
    start = i
    while i <= lastindex(text) && !(text[i] in (',', ':', '}', ')', ';'))
        i += 1
    end
    return parse(Float64, text[start:(i - 1)]), i
end

function _parse_branch_annotation(text::String, i::Int, version::Float64)
    text[i] == ':' || throw(ArgumentError("Expected branch annotation at position $i"))
    i += 1
    maps = NamedTuple{(:state, :length), Tuple{String, Float64}}[]
    branch_length = 0.0

    if version <= 1.0
        text[i] == '{' || throw(ArgumentError("Expected '{' at position $i"))
        i += 1
        while text[i] != '}'
            state_start = i
            while text[i] != ','
                i += 1
            end
            state = text[state_start:(i - 1)]
            i += 1
            len, i = _read_number(text, i)
            push!(maps, (state = state, length = len))
            branch_length += len
            if text[i] == ':'
                i += 1
            end
        end
        i += 1
    else
        startswith(SubString(text, i), "[&map={") || throw(ArgumentError("Expected [&map={ at position $i"))
        i += 7
        tokens = String[]
        start = i
        while text[i] != '}'
            if text[i] == ','
                push!(tokens, text[start:(i - 1)])
                i += 1
                start = i
            else
                i += 1
            end
        end
        push!(tokens, text[start:(i - 1)])
        i += 2
        branch_length, i = _read_number(text, i)
        if length(tokens) == 1
            push!(maps, (state = tokens[1], length = branch_length))
        else
            used = 0.0
            for j in 1:2:(length(tokens) - 2)
                state = tokens[j]
                len = parse(Float64, tokens[j + 1])
                used += len
                push!(maps, (state = state, length = len))
            end
            push!(maps, (state = tokens[end], length = branch_length - used))
        end
    end

    return maps, branch_length, i
end

function _encode_segments(edge_segments_raw; state_order = nothing)
    observed = Set(seg.state for segs in edge_segments_raw for seg in segs)
    labels =
        if state_order === nothing
            sort!(collect(observed))
        else
            ordered = String.(collect(state_order))
            ordered_set = Set(ordered)
            missing_states = setdiff(observed, ordered_set)
            extra_states = setdiff(ordered_set, observed)
            isempty(missing_states) || throw(ArgumentError("state_order is missing simmap states: $(collect(missing_states))"))
            isempty(extra_states) || throw(ArgumentError("state_order contains states absent from simmap: $(collect(extra_states))"))
            length(ordered) == length(ordered_set) || throw(ArgumentError("state_order contains duplicate states"))
            ordered
        end
    label_to_id = Dict(label => Int32(i) for (i, label) in enumerate(labels))
    encoded = [SimmapSegment[SimmapSegment(state = label_to_id[seg.state], length = seg.length) for seg in segs] for segs in edge_segments_raw]
    return encoded, String.(labels)
end

function _segments_to_state_vectors(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}})
    node_states = fill(Int32(0), tree.nnodes)
    edge_start_states = fill(Int32(0), tree.nedges)
    edge_end_states = fill(Int32(0), tree.nedges)
    root = Int(tree.root)
    root_edge = Int(tree.first_child_edge[root])
    node_states[root] = edge_segments[root_edge][1].state
    for node in tree.preorder
        tree.is_tip[node] && continue
        parent_state = node_states[node]
        for edge in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            segs = edge_segments[edge]
            edge_start_states[edge] = segs[1].state
            edge_end_states[edge] = segs[end].state
            edge_start_states[edge] == parent_state || (node_states[node] = edge_start_states[edge])
            node_states[Int(tree.child_of_edge[edge])] = segs[end].state
        end
    end
    return node_states, edge_start_states, edge_end_states
end

function _make_mapped_edge(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}})
    nstates = maximum(seg.state for segs in edge_segments for seg in segs)
    mapped = zeros(Float64, tree.nedges, nstates)
    for edge in 1:tree.nedges
        for seg in edge_segments[edge]
            mapped[edge, seg.state] += seg.length
        end
    end
    return mapped
end

const _SIMMAP_SIGNATURE_SEP = "\u001f"

@inline _desc_signature(labels::Vector{String}) = join(sort(labels), _SIMMAP_SIGNATURE_SEP)

function _edge_signatures(tree::CompactTree)
    map = build_phyloref(tree)
    sigs = Vector{String}(undef, tree.nedges)
    for edge in 1:tree.nedges
        sigs[edge] = phylo_edge_signature(tree, edge; sep = _SIMMAP_SIGNATURE_SEP, map = map)
    end
    return sigs
end

function _reorder_edge_segments(tree::CompactTree, edge_segments_raw, signatures_raw::Vector{String})
    edge_sigs = _edge_signatures(tree)
    sig_to_edge = Dict(sig => i for (i, sig) in enumerate(edge_sigs))
    reordered = Vector{typeof(edge_segments_raw[1])}(undef, tree.nedges)
    for (segments, sig) in zip(edge_segments_raw, signatures_raw)
        edge = get(sig_to_edge, sig, 0)
        edge == 0 && throw(ArgumentError("Failed to match simmap branch to internal edge ordering"))
        reordered[edge] = segments
    end
    return reordered
end

function _parse_simmap_text_to_plain(text::String, version::Float64, rev_order::Bool)
    pos = Ref(firstindex(text))
    plain = IOBuffer()
    edge_segments = Vector{Vector{NamedTuple{(:state, :length), Tuple{String, Float64}}}}()
    edge_signatures = String[]

    function parse_subtree()
        if text[pos[]] == '('
            print(plain, '(')
            pos[] += 1
            descendants = parse_subtree()
            while text[pos[]] == ','
                print(plain, ',')
                pos[] += 1
                append!(descendants, parse_subtree())
            end
            text[pos[]] == ')' || throw(ArgumentError("Malformed simmap text"))
            print(plain, ')')
            pos[] += 1
            node_descendants = descendants
        else
            label, newpos = _parse_label(text, pos[])
            print(plain, label)
            pos[] = newpos
            node_descendants = String[label]
        end

        if pos[] <= lastindex(text) && text[pos[]] == ':'
            maps, branch_length, newpos = _parse_branch_annotation(text, pos[], version)
            push!(edge_segments, rev_order ? reverse(maps) : maps)
            push!(edge_signatures, _desc_signature(node_descendants))
            print(plain, ':', branch_length)
            pos[] = newpos
        end
        return node_descendants
    end

    parse_subtree()
    text[pos[]] == ';' || throw(ArgumentError("Malformed simmap text: missing ';'"))
    print(plain, ';')
    return String(take!(plain)), edge_segments, edge_signatures
end

function _apply_translation(text::AbstractString, trans::Dict{String, String})
    isempty(trans) && return String(text)
    replaced = String(text)
    for (key, val) in sort(collect(trans); by = x -> -length(first(x)))
        replaced = replace(replaced, Regex("(?<=[(,])" * key * "(?=[:),])") => val)
    end
    return replaced
end

function _read_single_simmap_text(
    text::AbstractString;
    version::Float64 = 1.0,
    rev_order::Bool = true,
    translation::Dict{String, String} = Dict{String, String}(),
    state_order = nothing,
)
    translated = _apply_translation(strip(String(text)), translation)
    effective_version = _detect_simmap_version(translated, version)
    plain_newick, edge_segments_raw, signatures_raw = _parse_simmap_text_to_plain(translated, effective_version, rev_order)
    parsed_tree = NewickTree.readnw(plain_newick)
    tree = to_compact_tree(parsed_tree)
    edge_segments_raw = _reorder_edge_segments(tree, edge_segments_raw, signatures_raw)
    edge_segments, state_labels = _encode_segments(edge_segments_raw; state_order = state_order)
    length(edge_segments) == tree.nedges || throw(ArgumentError("Parsed simmap edges do not match tree edge count"))
    mapped_edge = _make_mapped_edge(tree, edge_segments)
    node_states, edge_start_states, edge_end_states = _segments_to_state_vectors(tree, edge_segments)
    sample = SimmapSample(
        success = true,
        nstates = size(mapped_edge, 2),
        root_state = node_states[Int(tree.root)],
        state_labels = state_labels,
        node_states = node_states,
        edge_start_states = edge_start_states,
        edge_end_states = edge_end_states,
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
        loglik = NaN,
    )
    return (tree = tree, simmap = sample, newick = plain_newick)
end

"""
    read_simmap(source; tree=nothing, version=1.0, state_order=nothing)

Read a `phytools`-style simmap tree from a file path or raw text and return the
corresponding `CompactTree` together with a `SimmapSample`.
"""
function read_simmap(
    source::AbstractString;
    format::Symbol = :auto,
    version::Float64 = 1.0,
    rev_order::Bool = true,
    state_order = nothing,
)
    content =
        if isfile(source)
            read(source, String)
        else
            _looks_like_missing_path(source) && throw(ArgumentError("SIMMAP file does not exist: $source"))
            source
        end
    fmt =
        if format === :auto
            occursin(r"(?i)#nexus", content) ? :nexus : :phylip
        else
            format
        end

    if fmt === :phylip
        texts = filter(!isempty, strip.(split(content, '\n')))
        trees = [_read_single_simmap_text(text; version = version, rev_order = rev_order, state_order = state_order) for text in texts]
    elseif fmt === :nexus
        texts, trans = _extract_tree_strings_from_nexus(content)
        trees = [_read_single_simmap_text(text; version = version, rev_order = rev_order, translation = trans, state_order = state_order) for text in texts]
    else
        throw(ArgumentError("Unsupported format=$fmt"))
    end

    return length(trees) == 1 ? trees[1] : trees
end



