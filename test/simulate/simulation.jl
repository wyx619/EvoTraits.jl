# Helper: build a single-state edge_segments vector for every edge with the
# given state.
function _single_state_segment(tree::CompactTree, state::Integer)
    return [[SimmapSegment(state = Int32(state), length = tree.edge_length[e])] for e in 1:tree.nedges]
end

@testset "simulate trait dispatch" begin
    simtree = sim_yule_simtree(12; tree_height = 1.0, rng = MersenneTwister(701))
    tree = serialize_tree(simtree)
    nedges = tree.nedges

    # Build a 2-regime edge_segments for regime-aware models:
    # alternate state 1 / state 2 per edge.
    reg2_segments = [
        [SimmapSegment(state = Int32(iseven(i) ? 2 : 1), length = tree.edge_length[i])]
        for i in 1:nedges
    ]

    Sigma = [0.4 0.05; 0.05 0.2]
    Sigmas = [
        [0.4 0.05; 0.05 0.2],
        [0.2 -0.01; -0.01 0.3],
    ]
    A = [1.1 0.1; 0.1 0.9]
    theta = [0.2, -0.4]
    thetas = [[0.2, -0.4], [0.8, 0.1]]

    @test size(sim_trait(:BM1, tree, Sigma; rng = MersenneTwister(702))) == (tree.ntips, 2)
    @test size(sim_trait(:BMM, tree, reg2_segments, Sigmas; rng = MersenneTwister(703))) == (tree.ntips, 2)
    @test size(sim_trait(:EB, tree, Sigma, -0.2; rng = MersenneTwister(704))) == (tree.ntips, 2)
    @test size(sim_trait(:OU1, tree, A, Sigma, theta; rng = MersenneTwister(705))) == (tree.ntips, 2)
    @test size(sim_trait(:OUM, tree, reg2_segments, A, Sigma, thetas; rng = MersenneTwister(706))) == (tree.ntips, 2)

    @test sim_trait(:BM1, tree, Sigma; rng = MersenneTwister(707)) ==
          sim_mv_data(:mvBM1, tree, Sigma; rng = MersenneTwister(707))
    @test sim_trait(:OUM, tree, reg2_segments, A, Sigma, thetas; rng = MersenneTwister(708)) ==
          sim_mv_data(:mvOUM, tree, reg2_segments, A, Sigma, thetas; rng = MersenneTwister(708))

    Sigma1 = reshape([0.3], 1, 1)
    A1 = reshape([1.2], 1, 1)
    theta1 = [0.5]
    thetas1 = [[0.5], [-0.2]]
    @test size(sim_trait(:BM1, tree, Sigma1; rng = MersenneTwister(709))) == (tree.ntips, 1)
    @test size(sim_trait(:OU1, tree, A1, Sigma1, theta1; rng = MersenneTwister(710))) == (tree.ntips, 1)
    @test size(sim_trait(:OUM, tree, reg2_segments, A1, Sigma1, thetas1; rng = MersenneTwister(711))) == (tree.ntips, 1)

    bm_bundle = sim_mvbm1_dataset(
        8,
        Sigma;
        tree_height = 1.0,
        trait_names = ["x", "y"],
        rng = MersenneTwister(712),
    )
    @test bm_bundle.tree.ntips == 8
    @test size(bm_bundle.traits) == (8, 2)
    @test bm_bundle.trait_names == ["x", "y"]
    @test length(bm_bundle.tip_labels) == 8

    @test_throws ArgumentError sim_trait(:UNKNOWN, tree, Sigma)
end


