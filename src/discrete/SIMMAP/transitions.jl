"""
    SimmapTransitionEvent

One state transition event extracted from a stochastic character map.
`nodeheight` is distance from the root, and `time` is distance before the
present for ultrametric-style trees (`tree_height - nodeheight`).
"""
Base.@kwdef struct SimmapTransitionEvent
    sim_id::Int = 0
    edge::Int = 0
    from_state::Int32 = 0
    to_state::Int32 = 0
    from_label::String = ""
    to_label::String = ""
    transition::String = ""
    nodeheight::Float64 = NaN
    time::Float64 = NaN
end

"""
    SimmapTransitionRate

Through-time transition-rate summary for one directed state transition.
`mya` is the upper edge of the time-before-present bin. Rates are computed as
transition counts divided by branch length spent in the source state.
"""
Base.@kwdef struct SimmapTransitionRate
    mya::Float64 = NaN
    state_from::Int32 = 0
    state_to::Int32 = 0
    from_label::String = ""
    to_label::String = ""
    transition::String = ""
    mean_count::Float64 = 0.0
    mean_start_branch_length::Float64 = 0.0
    mean_rate::Float64 = 0.0
end

function _simmap_node_depths(tree::CompactTree)
    depths = zeros(Float64, tree.nnodes)
    for node in tree.preorder
        tree.is_tip[node] && continue
        first_edge = tree.first_child_edge[node]
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            depths[child] = depths[node] + tree.edge_length[edge]
        end
    end
    return depths
end

function _simmap_tree_height(tree::CompactTree, depths::AbstractVector{<:Real})
    if !isempty(tree.tip_ids)
        return maximum(depths[tree.tip_ids])
    end
    return maximum(depths; init = 0.0)
end

function _simmap_labels_for_transitions(simmap::SimmapSample)
    return _simmap_state_labels(simmap, _simmap_infer_nstates(simmap))
end

function _simmap_transition_label(labels::AbstractVector{<:AbstractString}, from::Integer, to::Integer)
    from_label = 1 <= from <= length(labels) ? labels[from] : string(from)
    to_label = 1 <= to <= length(labels) ? labels[to] : string(to)
    return from_label, to_label, from_label * " -> " * to_label
end

"""
    transition_times(tree, simmap)
    transition_times(tree, simmaps)

Extract transition times from one stochastic character map or from a vector of
replicate maps. The returned events include directed state labels, distance from
root (`nodeheight`), and time before present (`time`).
"""
function transition_times(tree::CompactTree, simmap::SimmapSample; sim_id::Integer = 1)
    depths = _simmap_node_depths(tree)
    tree_height = _simmap_tree_height(tree, depths)
    labels = _simmap_labels_for_transitions(simmap)
    events = SimmapTransitionEvent[]

    for edge in 1:min(tree.nedges, length(simmap.edge_segments))
        segments = simmap.edge_segments[edge]
        length(segments) <= 1 && continue
        parent = tree.parent_of_edge[edge]
        distance_on_edge = 0.0
        for i in 1:(length(segments) - 1)
            distance_on_edge += segments[i].length
            from_state = Int(segments[i].state)
            to_state = Int(segments[i + 1].state)
            from_state == to_state && continue
            nodeheight = depths[parent] + distance_on_edge
            time = tree_height - nodeheight
            from_label, to_label, transition = _simmap_transition_label(labels, from_state, to_state)
            push!(events, SimmapTransitionEvent(
                sim_id = Int(sim_id),
                edge = edge,
                from_state = Int32(from_state),
                to_state = Int32(to_state),
                from_label = from_label,
                to_label = to_label,
                transition = transition,
                nodeheight = nodeheight,
                time = time,
            ))
        end
    end

    sort!(events; by = event -> (event.time, event.transition, event.edge))
    return events
end

function transition_times(tree::CompactTree, simmaps::AbstractVector{<:SimmapSample})
    events = SimmapTransitionEvent[]
    for (sim_id, simmap) in enumerate(simmaps)
        append!(events, transition_times(tree, simmap; sim_id = sim_id))
    end
    return events
end

function _transition_events_dataframe(events)
    return DataFrame(
        sim_id = [event.sim_id for event in events],
        edge_id = [event.edge for event in events],
        from = [event.from_label for event in events],
        to = [event.to_label for event in events],
        time_before_present = [event.time for event in events],
    )
end

"""
    transition_events_DataFrame(tree, simmap; R_order = nothing, map = nothing)
    transition_events_DataFrame(tree, simmaps; R_order = nothing, map = nothing)

Return a `DataFrame` with one row per SIMMAP transition event. By default the
columns are `sim_id`, `edge_id`, `from`, `to`, and `time_before_present`.

If `R_order` is `:cladewise` or `:postorder`, the returned table is augmented
with `phyloref` branch-identity columns using the requested R/ape edge order.
"""
function transitionevents(
    tree::CompactTree,
    simmap::SimmapSample;
    R_order::Union{Nothing, Symbol} = nothing,
    map = nothing,
)
    events = transition_times(tree, simmap)
    df = _transition_events_dataframe(events)
    R_order === nothing && return df
    checked_map = map === nothing ? build_phyloref(tree) : map
    return attach_R_edge_map(df, tree; edge_col = :edge_id, order = R_order, map = checked_map)
end

function transitionevents(
    tree::CompactTree,
    simmaps::AbstractVector{<:SimmapSample};
    R_order::Union{Nothing, Symbol} = nothing,
    map = nothing,
)
    events = transition_times(tree, simmaps)
    df = _transition_events_dataframe(events)
    R_order === nothing && return df
    checked_map = map === nothing ? build_phyloref(tree) : map
    return attach_R_edge_map(df, tree; edge_col = :edge_id, order = R_order, map = checked_map)
end

transition_events_DataFrame(args...; kwargs...) = transitionevents(args...; kwargs...)

function _simmap_bin_index(time::Real, bin::Real, nbins::Integer)
    bin > 0 || throw(ArgumentError("bin must be positive"))
    idx = floor(Int, max(Float64(time), 0.0) / Float64(bin)) + 1
    return clamp(idx, 1, nbins)
end

function _accumulate_simmap_rate_stats!(counts, exposure, tree::CompactTree, simmap::SimmapSample, depths, tree_height::Float64, bin::Float64)
    nbins = size(counts, 1)
    nstates = size(counts, 2)

    for edge in 1:min(tree.nedges, length(simmap.edge_segments))
        segments = simmap.edge_segments[edge]
        parent = tree.parent_of_edge[edge]
        nodeheight = depths[parent]

        for (i, segment) in enumerate(segments)
            state = Int(segment.state)
            seg_start = nodeheight
            seg_end = nodeheight + segment.length
            start_time = tree_height - seg_start
            end_time = tree_height - seg_end
            lo = min(start_time, end_time)
            hi = max(start_time, end_time)

            if 1 <= state <= nstates && segment.length > 0.0
                first_bin = _simmap_bin_index(lo, bin, nbins)
                last_bin = _simmap_bin_index(max(hi - eps(Float64), 0.0), bin, nbins)
                for bin_idx in first_bin:last_bin
                    bin_lo = (bin_idx - 1) * bin
                    bin_hi = bin_idx * bin
                    overlap = min(hi, bin_hi) - max(lo, bin_lo)
                    overlap > 0.0 && (exposure[bin_idx, state] += overlap)
                end
            end

            if i < length(segments)
                from_state = state
                to_state = Int(segments[i + 1].state)
                if from_state != to_state && 1 <= from_state <= nstates && 1 <= to_state <= nstates
                    event_time = tree_height - seg_end
                    bin_idx = _simmap_bin_index(event_time, bin, nbins)
                    counts[bin_idx, from_state, to_state] += 1.0
                end
            end

            nodeheight = seg_end
        end
    end
    return nothing
end

"""
    transition_rates_through_time(tree, simmaps; bin=1.0)

Summarize directed transition rates through time across one or more SIMMAP
replicates. For each time-before-present bin and directed state pair, this
returns mean transition count, mean branch length spent in the source state, and
mean transition rate.
"""
function transition_rates_through_time(tree::CompactTree, simmaps::AbstractVector{<:SimmapSample}; bin::Real = 1.0)
    bin > 0 || throw(ArgumentError("bin must be positive"))
    isempty(simmaps) && return SimmapTransitionRate[]

    depths = _simmap_node_depths(tree)
    tree_height = _simmap_tree_height(tree, depths)
    nbins = max(1, ceil(Int, tree_height / Float64(bin)))
    nstates = maximum(_simmap_infer_nstates(simmap) for simmap in simmaps)
    labels = _simmap_state_labels(first(simmaps), nstates)

    total_counts = zeros(Float64, nbins, nstates, nstates)
    total_exposure = zeros(Float64, nbins, nstates)
    total_rates = zeros(Float64, nbins, nstates, nstates)
    nsims = length(simmaps)

    for simmap in simmaps
        counts = zeros(Float64, nbins, nstates, nstates)
        exposure = zeros(Float64, nbins, nstates)
        _accumulate_simmap_rate_stats!(counts, exposure, tree, simmap, depths, tree_height, Float64(bin))
        total_counts .+= counts
        total_exposure .+= exposure
        for bin_idx in 1:nbins, from_state in 1:nstates, to_state in 1:nstates
            from_state == to_state && continue
            exposure[bin_idx, from_state] > 0.0 && (total_rates[bin_idx, from_state, to_state] += counts[bin_idx, from_state, to_state] / exposure[bin_idx, from_state])
        end
    end

    rows = SimmapTransitionRate[]
    for bin_idx in 1:nbins, from_state in 1:nstates, to_state in 1:nstates
        from_state == to_state && continue
        from_label, to_label, transition = _simmap_transition_label(labels, from_state, to_state)
        push!(rows, SimmapTransitionRate(
            mya = bin_idx * Float64(bin),
            state_from = Int32(from_state),
            state_to = Int32(to_state),
            from_label = from_label,
            to_label = to_label,
            transition = transition,
            mean_count = total_counts[bin_idx, from_state, to_state] / nsims,
            mean_start_branch_length = total_exposure[bin_idx, from_state] / nsims,
            mean_rate = total_rates[bin_idx, from_state, to_state] / nsims,
        ))
    end
    return rows
end

transition_rates_through_time(tree::CompactTree, simmap::SimmapSample; kwargs...) = transition_rates_through_time(tree, [simmap]; kwargs...)

