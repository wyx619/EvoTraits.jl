using EvoTraits, Random, DataFrames

function documentation_data()
    rng = MersenneTwister(173)
    tree = sim_yule_tree(24; tree_height = 10.0, rng = rng)
    # Synthetic demonstration history.
    node_states = Int32[iseven(v) ? 2 : 1 for v in 1:tree.nnodes]
    node_states[tree.root] = 1
    starts = node_states[tree.parent_of_edge]
    ends = node_states[tree.child_of_edge]
    segments = [starts[e] == ends[e] ?
        [SimmapSegment(state = starts[e], length = tree.edge_length[e])] :
        [SimmapSegment(state = starts[e], length = tree.edge_length[e] / 2),
         SimmapSegment(state = ends[e], length = tree.edge_length[e] / 2)]
        for e in 1:tree.nedges]
    mapped = zeros(tree.nedges, 2)
    for e in 1:tree.nedges
        for segment in segments[e]
            mapped[e, segment.state] += segment.length
        end
    end
    sim = SimmapSample(success = true, nstates = 2, root_state = Int32(1),
        state_labels = ["open", "closed"],
        node_states = node_states, edge_start_states = starts,
        edge_end_states = ends, edge_segments = segments, mapped_edge = mapped)
    Sigma = [0.7 0.15; 0.15 0.5]
    X = sim_mvbm1(tree, Sigma; rng = rng)
    return (; tree, sim, X, y = X[:, 1])
end
