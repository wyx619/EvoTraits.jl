@testset "Multivariate simulation helpers" begin
    rng = MersenneTwister(20260420)
    simtree = simulate_yule_simtree(12; birth_rate = 1.2, tree_height = 2.5, rng = MersenneTwister(20260419))
    direct_tree = serialize_tree(simtree)
    yule_tree = simulate_yule_tree(12; birth_rate = 1.2, tree_height = 2.5, rng = MersenneTwister(20260419))
    yule_newick = to_newick(simtree)
    yule_path = joinpath(mktempdir(), "yule_tree.tre")
    write(yule_path, yule_newick)
    yule_from_newick = serialize_tree(read_tree(yule_path))
    @test direct_tree.ntips == 12
    @test yule_tree.ntips == 12
    @test yule_from_newick.ntips == 12
    @test direct_tree.nedges == 22
    @test isapprox(maximum(direct_tree.dist_from_root[direct_tree.tip_ids]), 2.5; atol = 1e-8)
    @test isapprox(maximum(direct_tree.dist_from_root[direct_tree.tip_ids]), minimum(direct_tree.dist_from_root[direct_tree.tip_ids]); atol = 1e-8)
    @test all(>=(0.0), direct_tree.edge_length)

    bd_simtree = simulate_birth_death_simtree(
        12;
        birth_rate = 1.4,
        death_rate = 0.3,
        tree_height = 2.5,
        rng = MersenneTwister(20260421),
    )
    bd_tree = serialize_tree(bd_simtree)
    bd_tip_depths = bd_tree.dist_from_root[bd_tree.tip_ids]
    @test bd_tree.ntips == 12
    @test isapprox(maximum(bd_tip_depths), 2.5; atol = 1e-8)
    @test isapprox(maximum(bd_tip_depths), minimum(bd_tip_depths); atol = 1e-8)
    @test all(>=(0.0), bd_tree.edge_length)

    bd_yule = simulate_birth_death_tree(
        12;
        birth_rate = 1.2,
        death_rate = 0.0,
        tree_height = 2.5,
        rng = MersenneTwister(20260419),
    )
    @test bd_yule.ntips == yule_tree.ntips
    @test bd_yule.nedges == yule_tree.nedges
    @test bd_yule.edge_length ≈ yule_tree.edge_length

    newick = simulate_ultrametric_newick(12; tree_height = 2.5, rng = rng)
    tree_path = joinpath(mktempdir(), "mv_sim_tree.tre")
    write(tree_path, newick)
    tree = serialize_tree(read_tree(tree_path))

    @test tree.ntips == 12
    tip_depths = tree.dist_from_root[tree.tip_ids]
    @test isapprox(maximum(tip_depths), minimum(tip_depths); atol = 1e-8)
    @test isapprox(only(unique(round.(tip_depths; digits = 8))), 2.5; atol = 1e-6)

    simtree3 = simulate_yule_simtree(12; tree_height = 1.7, rng = MersenneTwister(6))
    tree3 = serialize_tree(simtree3)
    @test tree3.ntips == 12
    # Build simple single-state edge segments as a basic validity test
    three_regime_segments = [[SimmapSegment(state = Int32(i % 3 + 1), length = tree3.edge_length[e])] for (e, i) in enumerate(1:tree3.nedges)]
    @test length(three_regime_segments) == tree3.nedges
    for edge in 1:tree3.nedges
        @test isapprox(sum(seg.length for seg in three_regime_segments[edge]), tree3.edge_length[edge]; atol = 1e-8)
    end

    Sigma = [
        0.8 0.2;
        0.2 0.6;
    ]
    traits = simulate_mvbm1(tree, Sigma; rng = MersenneTwister(7))
    @test size(traits) == (12, 2)

    bundle = simulate_mvbm1_dataset(
        10,
        Sigma;
        tree_height = 1.2,
        trait_names = ["x1", "x2"],
        rng = MersenneTwister(8),
    )
    @test bundle.tree.ntips == 10
    @test size(bundle.traits) == (10, 2)
    @test bundle.trait_names == ["x1", "x2"]
    @test length(bundle.tip_labels) == 10

    tree_path2 = joinpath(mktempdir(), "mv_regime_tree.tre")
    write(tree_path2, "((A:0.5,B:0.5):0.5,(C:0.5,D:0.5):0.5);")
    regime_tree = serialize_tree(read_tree(tree_path2))
    edge_segments = [
        [SimmapSegment(state = 1, length = 0.5)],
        [SimmapSegment(state = 2, length = 0.25), SimmapSegment(state = 1, length = 0.25)],
        [SimmapSegment(state = 2, length = 0.5)],
        [SimmapSegment(state = 1, length = 0.5)],
        [SimmapSegment(state = 1, length = 0.5)],
        [SimmapSegment(state = 2, length = 0.5)],
    ]
    Sig1 = [
        0.8 0.2;
        0.2 0.7;
    ]
    Sig2 = [
        0.5 0.1;
        0.1 0.6;
    ]
    bmm_traits = simulate_mvbmm(regime_tree, edge_segments, [Sig1, Sig2]; rng = MersenneTwister(9))
    @test size(bmm_traits) == (4, 2)
    @test size(simulate_mv_data(:mvBMM, regime_tree, edge_segments, [Sig1, Sig2]; rng = MersenneTwister(9))) == (4, 2)

    A = [
        1.0 0.2;
        0.0 0.8;
    ]
    Sigma = [
        0.6 0.15;
        0.15 0.7;
    ]
    theta = [0.0, 0.5]
    ou1_traits = simulate_mvou1(regime_tree, A, Sigma, theta; rng = MersenneTwister(10))
    @test size(ou1_traits) == (4, 2)
    @test size(simulate_mv_data(:mvOU1, regime_tree, A, Sigma, theta; rng = MersenneTwister(10))) == (4, 2)

    theta_regimes = [[0.0, 0.5], [0.8, -0.2]]
    oum_traits = simulate_mvoum(regime_tree, edge_segments, A, Sigma, theta_regimes; rng = MersenneTwister(11))
    @test size(oum_traits) == (4, 2)

    eb_traits = simulate_mveb(regime_tree, Sigma, -0.5; rng = MersenneTwister(12))
    @test size(eb_traits) == (4, 2)

    Sigma1 = reshape([0.7], 1, 1)
    A1 = reshape([0.9], 1, 1)
    theta1 = [0.2]
    one_trait_bm = simulate_mvbm1(regime_tree, Sigma1; rng = MersenneTwister(13))
    one_trait_ou = simulate_mvou1(regime_tree, A1, Sigma1, theta1; rng = MersenneTwister(14))
    one_trait_oum = simulate_mvoum(regime_tree, edge_segments, A1, Sigma1, [[0.1], [0.8]]; rng = MersenneTwister(15))
    one_trait_eb = simulate_mveb(regime_tree, Sigma1, -0.2; rng = MersenneTwister(16))
    @test size(one_trait_bm) == (4, 1)
    @test size(one_trait_ou) == (4, 1)
    @test size(one_trait_oum) == (4, 1)
    @test size(one_trait_eb) == (4, 1)

end

@testset "Multivariate hard limits" begin
    tmp = tempname() * ".tre"
    write(tmp, simulate_ultrametric_newick(4001; rng = MersenneTwister(202)))
    tree = serialize_tree(read_tree(tmp))
    oversized_trait = zeros(Float64, 4001, 2)
    A = [1.0 0.0; 0.0 1.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    theta = [0.0, 0.0]
    @test_throws ArgumentError mvou1_loglikelihood(tree, oversized_trait, A, Sigma, theta)
end






