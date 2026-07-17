@testset "BM1 fitting" begin
    bm_tree_path = joinpath(mktempdir(), "toy_bm1_tree.tre")
    write(bm_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = to_compact_tree(load_newick_tree(bm_tree_path))
    trait = [1.0, 1.2, 2.5, 2.8]

    ll = bm1_loglikelihood(toy_tree, trait, 0.5)
    @test ll.success
    @test isfinite(ll.loglik)
    @test isfinite(ll.root_state)
    @test ll.nparams == 2

    fit = fit_bm1(toy_tree, trait; max_iterations = 40)
    @test fit.success
    @test fit.model == :BM1
    @test fit.sigma2 > 0.0
    @test isfinite(fit.theta)
    @test isfinite(fit.root_state)
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
end

@testset "BM1 DataFrame entrypoint" begin
    tree_path = joinpath(mktempdir(), "toy_bm_dataframe_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = to_compact_tree(load_newick_tree(tree_path))
    df = DataFrame(taxon = ["A", "B", "C", "D"], lnH = [1.0, 1.2, 2.5, 2.8])

    fit = fit_bm1(toy_tree, df; max_iterations = 20)
    @test fit.success
    @test fit.trait_name == "lnH"
    @test occursin("lnH", sprint(show, MIME("text/plain"), fit))

    df_missing = DataFrame(taxon = ["C", "A", "D", "B"], lnH = [2.5, 1.0, missing, "NA"])
    missing_fit = fit_bm1(toy_tree, df_missing; max_iterations = 20)
    @test missing_fit.success
    @test missing_fit.trait_name == "lnH"
    @test isfinite(missing_fit.loglik)
end

@testset "BM1 and BMM missing likelihood" begin
    tree_path = joinpath(mktempdir(), "toy_bm_missing_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = to_compact_tree(load_newick_tree(tree_path))
    trait = [1.0, NaN, 2.5, 2.8]

    bm = bm1_loglikelihood(toy_tree, trait, 0.5)
    @test bm.success
    @test isfinite(bm.loglik)
    fit = fit_bm1(toy_tree, trait; max_iterations = 40)
    @test fit.success
    @test isfinite(fit.loglik)
    asr = estim_node(toy_tree, trait, fit)
    @test asr.success
    @test all(isfinite, asr.estimates)

    mapped = [
        1.0 0.0;
        1.0 0.0;
        0.0 1.0;
        0.0 1.0;
        1.0 0.0;
        0.0 1.0;
    ]
    bmm = bmm_loglikelihood(toy_tree, trait, mapped, [0.3, 0.8])
    @test bmm.success
    @test isfinite(bmm.loglik)
end

@testset "BM1 smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = to_compact_tree(load_newick_tree(tree_path))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))

    fit = fit_bm1(big_tree, trait; max_iterations = 30)
    @test fit.success
    @test fit.model == :BM1
    @test fit.sigma2 > 0.0
    @test isfinite(fit.loglik)
end


@testset "BMM fitting" begin
    bmm_tree_path = joinpath(mktempdir(), "toy_bmm_tree.tre")
    write(bmm_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = to_compact_tree(load_newick_tree(bmm_tree_path))
    trait = [1.0, 1.2, 2.5, 2.8]
    mapped = [
        1.0 0.0;
        1.0 0.0;
        0.0 1.0;
        0.0 1.0;
        1.0 0.0;
        0.0 1.0;
    ]

    ll = bmm_loglikelihood(toy_tree, trait, mapped, [0.3, 0.8])
    @test ll.success
    @test isfinite(ll.loglik)
    @test ll.nregimes == 2
    @test ll.nparams == 3

    fit = fit_bmm(toy_tree, trait, mapped; max_iterations = 40)
    @test fit.success
    @test fit.model == :BMM
    @test fit.nregimes == 2
    @test length(fit.sigma2) == 2
    @test all(>(0.0), fit.sigma2)
    @test isfinite(fit.theta)
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)

    simmap = SimmapSample(
        success = true,
        nstates = 2,
        state_labels = ["blue", "red"],
        edge_segments = [
            [SimmapSegment(state = 1, length = 1.0)],
            [SimmapSegment(state = 1, length = 1.0)],
            [SimmapSegment(state = 2, length = 1.0)],
            [SimmapSegment(state = 2, length = 1.0)],
            [SimmapSegment(state = 1, length = 1.0)],
            [SimmapSegment(state = 2, length = 1.0)],
        ],
        mapped_edge = mapped,
    )
    fit_simmap = fit_bmm(toy_tree, trait, simmap; max_iterations = 40)
    @test fit_simmap.success
    @test fit_simmap.regime_names == ["blue", "red"]
    @test isapprox(fit_simmap.loglik, fit.loglik; atol = 1e-6)
end

@testset "BMM smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = to_compact_tree(load_newick_tree(tree_path))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))
    mapped = zeros(Float64, big_tree.nedges, 4)
    for edge in 1:big_tree.nedges
        mapped[edge, mod1(edge, 4)] = big_tree.edge_length[edge]
    end

    fit = fit_bmm(big_tree, trait, mapped; max_iterations = 20)
    @test fit.success
    @test fit.model == :BMM
    @test fit.nregimes == 4
    @test length(fit.sigma2) == 4
    @test all(>(0.0), fit.sigma2)
    @test isfinite(fit.loglik)
end





