"""
    SimmapSummary

Compact, human-readable summary of a `SimmapSample`. The summary is derived
from `edge_segments` and `mapped_edge`, with optional tree-level metadata when
`describe_simmap(tree, simmap)` is used.
"""
Base.@kwdef struct SimmapSummary
    success::Bool = false
    ntips::Union{Nothing, Int} = nothing
    nnodes::Union{Nothing, Int} = nothing
    nedges::Int = 0
    nstates::Int = 0
    root_state::Int32 = 0
    root_label::String = ""
    state_labels::Vector{String} = String[]
    total_branch_length::Float64 = 0.0
    mapped_lengths::Vector{Float64} = Float64[]
    mapped_proportions::Vector{Float64} = Float64[]
    transition_matrix::Matrix{Int} = zeros(Int, 0, 0)
    transition_count::Int = 0
    edges_with_transitions::Int = 0
    segment_count::Int = 0
    max_segments_per_edge::Int = 0
    loglik::Float64 = NaN
end

function _simmap_infer_nstates(simmap::SimmapSample)
    nstates = max(simmap.nstates, length(simmap.state_labels), size(simmap.mapped_edge, 2))
    for segments in simmap.edge_segments
        for seg in segments
            nstates = max(nstates, Int(seg.state))
        end
    end
    return nstates
end

function _simmap_state_labels(simmap::SimmapSample, nstates::Int)
    labels = Vector{String}(undef, nstates)
    for state in 1:nstates
        labels[state] = state <= length(simmap.state_labels) ? simmap.state_labels[state] : string(state)
    end
    return labels
end

function _simmap_mapped_lengths(simmap::SimmapSample, nstates::Int)
    lengths = zeros(Float64, nstates)
    if !isempty(simmap.mapped_edge)
        ncols = min(nstates, size(simmap.mapped_edge, 2))
        for state in 1:ncols
            lengths[state] = sum(@view simmap.mapped_edge[:, state])
        end
        return lengths
    end
    for segments in simmap.edge_segments
        for seg in segments
            1 <= seg.state <= nstates && (lengths[Int(seg.state)] += seg.length)
        end
    end
    return lengths
end

function _simmap_transition_stats(simmap::SimmapSample, nstates::Int)
    matrix = zeros(Int, nstates, nstates)
    transitions = 0
    edges_with_transitions = 0
    segment_count = 0
    max_segments_per_edge = 0
    for segments in simmap.edge_segments
        nseg = length(segments)
        segment_count += nseg
        max_segments_per_edge = max(max_segments_per_edge, nseg)
        edge_transitions = 0
        for i in 2:nseg
            if segments[i - 1].state != segments[i].state
                from = Int(segments[i - 1].state)
                to = Int(segments[i].state)
                if 1 <= from <= nstates && 1 <= to <= nstates
                    matrix[from, to] += 1
                end
                edge_transitions += 1
            end
        end
        transitions += edge_transitions
        edge_transitions > 0 && (edges_with_transitions += 1)
    end
    return matrix, transitions, edges_with_transitions, segment_count, max_segments_per_edge
end

"""
    describe_simmap(simmap; tree=nothing)
    describe_simmap(tree, simmap)

Summarize a stochastic character map with state-wise mapped branch lengths,
transition counts, segment counts, and optional tree metadata.
"""
function describe_simmap(simmap::SimmapSample; tree::Union{Nothing, CompactTree} = nothing)
    nstates = _simmap_infer_nstates(simmap)
    labels = _simmap_state_labels(simmap, nstates)
    mapped_lengths = _simmap_mapped_lengths(simmap, nstates)
    total = sum(mapped_lengths)
    if total <= 0.0 && tree !== nothing
        total = sum(tree.edge_length)
    end
    proportions = total > 0.0 ? mapped_lengths ./ total : zeros(Float64, nstates)
    transition_matrix, transitions, edges_with_transitions, segment_count, max_segments_per_edge =
        _simmap_transition_stats(simmap, nstates)
    root_label = 1 <= simmap.root_state <= length(labels) ? labels[Int(simmap.root_state)] : string(simmap.root_state)
    nedges = tree === nothing ? length(simmap.edge_segments) : tree.nedges

    return SimmapSummary(
        success = simmap.success,
        ntips = tree === nothing ? nothing : tree.ntips,
        nnodes = tree === nothing ? nothing : tree.nnodes,
        nedges = nedges,
        nstates = nstates,
        root_state = simmap.root_state,
        root_label = root_label,
        state_labels = labels,
        total_branch_length = total,
        mapped_lengths = mapped_lengths,
        mapped_proportions = proportions,
        transition_matrix = transition_matrix,
        transition_count = transitions,
        edges_with_transitions = edges_with_transitions,
        segment_count = segment_count,
        max_segments_per_edge = max_segments_per_edge,
        loglik = simmap.loglik,
    )
end

describe_simmap(tree::CompactTree, simmap::SimmapSample) = describe_simmap(simmap; tree = tree)

"""
    summary_simmap(args...; kwargs...)

Alias for `describe_simmap`, provided for users coming from phytools-style
`summary.simmap` workflows.
"""
summary_simmap(args...; kwargs...) = describe_simmap(args...; kwargs...)
summarize(args...; kwargs...) = describe_simmap(args...; kwargs...)

function _simmap_preview(labels::Vector{String}, n::Int)
    if length(labels) <= n
        return join(labels, ", ")
    end
    return join(labels[1:n], ", ") * ", ..."
end

function _simmap_has_branch_lengths(tree::CompactTree)
    return !isempty(tree.edge_length) && all(isfinite, tree.edge_length)
end

"""
    print_simmap(tree, simmap; printlen=6)
    print_simmap(io, tree, simmap; printlen=6)

Print a phytools-style tree-plus-SIMMAP overview. This is separate from
`Base.show(simmap)` because a `SimmapSample` does not store the tree.
"""
function print_simmap(io::IO, tree::CompactTree, simmap::SimmapSample; printlen::Int = 6)
    labels = _simmap_state_labels(simmap, _simmap_infer_nstates(simmap))
    println(io, "Phylogenetic tree with $(tree.ntips) tips and $(tree.nnodes - tree.ntips) internal nodes.")
    println(io)
    println(io, "Tip labels:")
    println(io, "\t", _simmap_preview(tree.tip_labels, printlen))
    println(io)
    println(io, "The tree includes a mapped, $(length(labels))-state discrete character")
    println(io, "with states:")
    println(io, "\t", join(labels, ", "))
    println(io)
    rooted = tree.root > 0 ? "Rooted" : "Unrooted"
    branch_lengths = _simmap_has_branch_lengths(tree) ? "includes branch lengths." : "does not include branch lengths."
    println(io, rooted, "; ", branch_lengths)
    return nothing
end

print_simmap(tree::CompactTree, simmap::SimmapSample; kwargs...) = print_simmap(stdout, tree, simmap; kwargs...)

function _simmap_float(x::Real)
    return isfinite(Float64(x)) ? string(round(Float64(x); sigdigits = 7)) : string(Float64(x))
end

function _simmap_maybe(value)
    return value === nothing ? "unknown" : string(value)
end

function Base.summary(simmap::SimmapSample)
    s = describe_simmap(simmap)
    return "SimmapSample with $(s.nstates) states, $(s.nedges) edges, $(s.transition_count) transitions"
end

function Base.show(io::IO, simmap::SimmapSample)
    s = describe_simmap(simmap)
    print(io, "SimmapSample(success=", s.success,
        ", states=", s.nstates,
        ", edges=", s.nedges,
        ", transitions=", s.transition_count,
        ", total_length=", _simmap_float(s.total_branch_length),
        ")")
end

Base.show(io::IO, ::MIME"text/plain", simmap::SimmapSample) = show(io, MIME("text/plain"), describe_simmap(simmap))

function Base.show(io::IO, summary::SimmapSummary)
    print(io, "SimmapSummary(states=", summary.nstates,
        ", edges=", summary.nedges,
        ", transitions=", summary.transition_count,
        ", total_length=", _simmap_float(summary.total_branch_length),
        ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SimmapSummary)
    tree_count = summary.ntips === nothing ? "1 tree" : "1 tree"
    println(io, tree_count, " with a mapped discrete character with states:")
    println(io, " ", join(summary.state_labels, ", "), " ")
    println(io)
    println(io, "tree has ", summary.transition_count, " changes between states")
    if summary.nstates > 0
        println(io)
        println(io, "changes are of the following types:")
        _show_simmap_matrix(io, summary.state_labels, summary.transition_matrix)
        println(io)
        println(io, "mean total time spent in each state is:")
        _show_simmap_time_table(io, summary)
    end
end

function _show_simmap_matrix(io::IO, labels::Vector{String}, matrix::Matrix{Int})
    label_width = max(12, maximum(length.(labels); init = 0))
    col_widths = [max(length(label), 8) for label in labels]
    print(io, lpad("", label_width))
    for (label, width) in zip(labels, col_widths)
        print(io, " ", lpad(label, width))
    end
    println(io)
    for i in eachindex(labels)
        print(io, rpad(labels[i], label_width))
        for j in eachindex(labels)
            print(io, " ", lpad(string(matrix[i, j]), col_widths[j]))
        end
        println(io)
    end
end

function _show_simmap_time_table(io::IO, summary::SimmapSummary)
    labels = vcat(summary.state_labels, ["total"])
    values = vcat(summary.mapped_lengths, [summary.total_branch_length])
    props = vcat(summary.mapped_proportions, [sum(summary.mapped_proportions)])
    row_width = 5
    col_widths = [max(length(labels[i]), length(_simmap_float(values[i])), length(_simmap_float(props[i])), 10) for i in eachindex(labels)]
    print(io, lpad("", row_width))
    for (label, width) in zip(labels, col_widths)
        print(io, " ", lpad(label, width))
    end
    println(io)
    print(io, rpad("raw", row_width))
    for (value, width) in zip(values, col_widths)
        print(io, " ", lpad(_simmap_float(value), width))
    end
    println(io)
    print(io, rpad("prop", row_width))
    for (value, width) in zip(props, col_widths)
        print(io, " ", lpad(_simmap_float(value), width))
    end
    println(io)
end
