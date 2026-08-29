@testset "EB fitting" begin
    eb_tree_path = joinpath(mktempdir(), "toy_eb_tree.tre")
    write(eb_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(eb_tree_path))
    trait = [1.0, 1.15, 2.0, 2.2]

    ll = eb_loglikelihood(toy_tree, trait, 0.4, -0.2)
    @test ll.success
    @test isfinite(ll.loglik)
    @test ll.nparams == 3

    fit = fit_eb(toy_tree, trait; max_iterations = 40)
    @test fit.success
    @test fit.model == :EB
    @test isfinite(fit.beta)
    @test isfinite(fit.theta)
    @test fit.sigma2 > 0.0
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
end

@testset "EB and EBM missing likelihood" begin
    tree_path = joinpath(mktempdir(), "toy_eb_missing_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(tree_path))
    trait = [1.0, NaN, 2.0, 2.2]

    eb = eb_loglikelihood(toy_tree, trait, 0.4, -0.2)
    @test eb.success
    @test isfinite(eb.loglik)
    fit = fit_eb(toy_tree, trait; max_iterations = 40)
    @test fit.success
    @test isfinite(fit.loglik)

    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    ebm = ebm_loglikelihood(toy_tree, trait, edge_segments, [0.2, 0.4], [-0.1, -0.2])
    @test ebm.success
    @test isfinite(ebm.loglik)
end

@testset "EBM fitting" begin
    ebm_tree_path = joinpath(mktempdir(), "toy_ebm_tree.tre")
    write(ebm_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(ebm_tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    ebm_ll = ebm_loglikelihood(toy_tree, trait, edge_segments, [0.2, 0.4], [-0.1, -0.2])
    @test ebm_ll.success
    @test ebm_ll.model == :EBM
    @test length(ebm_ll.sigma2) == 2
    @test length(ebm_ll.beta_regimes) == 2
    @test all(<=(0.0), ebm_ll.beta_regimes)
    @test ebm_ll.nparams == 5

    ebm_fit = fit_ebm(toy_tree, trait, edge_segments; max_iterations = 40)
    @test ebm_fit.success
    @test ebm_fit.model == :EBM
    @test length(ebm_fit.sigma2) == 2
    @test all(>(0.0), ebm_fit.sigma2)
    @test length(ebm_fit.beta_regimes) == 2
    @test all(<=(0.0), ebm_fit.beta_regimes)

    simmap = SimmapSample(
        success = true,
        nstates = 2,
        state_labels = ["blue", "red"],
        edge_segments = edge_segments,
        mapped_edge = [1.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 1.0],
    )
    ebm_simmap = fit_ebm(toy_tree, trait, simmap; max_iterations = 40)
    @test ebm_simmap.success
    @test ebm_simmap.regime_names == ["blue", "red"]
    @test ebm_simmap.model == :EBM

end

@testset "EB smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = serialize_tree(read_tree(tree_path))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))

    fit = fit_eb(big_tree, trait; max_iterations = 20)
    @test fit.success
    @test fit.model == :EB
    @test fit.sigma2 > 0.0
    @test isfinite(fit.loglik)
end






