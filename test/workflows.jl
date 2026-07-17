@testset "Continuous model registries" begin
    continuous_models = Set([
        :BM1,
        :OU1,
        :EB,
        :BMM,
        :OUM,
        :OUMV,
        :OUMA,
        :OUMVA,
        :EBM,
    ])
    mv_models = Set([
        :mvBM1,
        :mvBMM,
        :mvEB,
        :mvOU1,
        :mvOUM,
        :mvOUMV,
        :mvOUMA,
        :mvOUMVA,
    ])

    c_registry = EvoTraits._continuous_model_registry()
    mv_registry = EvoTraits._multivariate_model_registry()

    @test Set(keys(c_registry)) == continuous_models
    @test Set(keys(mv_registry)) == mv_models
    @test c_registry[:BMM].context_kind == :regime_map
    @test c_registry[:OUMA].context_kind == :edge_segments
    @test mv_registry[:mvBMM].context_kind == :regime_map
    @test mv_registry[:mvOUM].context_kind == :edge_segments
end

@testset "AIC model comparison helpers" begin
    tree_path = joinpath(mktempdir(), "toy_aic_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = to_compact_tree(load_newick_tree(tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]

    fit_bm = fit_bm1(toy_tree, trait; max_iterations = 30)
    fit_ou = fit_ou1(toy_tree, trait; max_iterations = 30)
    eb_fit = fit_eb(toy_tree, trait; max_iterations = 30)

    table = aic_table(:BM1 => fit_bm, :OU1 => fit_ou, :EB => eb_fit)

    @test length(table) == 3
    @test table[1].delta_aic == 0.0
    @test issorted(getfield.(table, :aic))
    @test aic(fit_bm) == fit_bm.aic
    @test loglik(fit_ou) == fit_ou.loglik
    @test nparams(eb_fit) == eb_fit.nparams
    @test model(fit_bm) == :BM1
    @test best_model(table).name == table[1].name

    failed = ContinuousFitResult(model = :FAIL, success = false, aic = NaN, loglik = -Inf, nparams = 0)
    mixed = aic_table(:FAIL => failed, :BM1 => fit_bm)
    @test best_model(mixed).name == :BM1
end

@testset "Continuous workflow wrappers" begin
    tree_path = joinpath(mktempdir(), "toy_workflow_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = to_compact_tree(load_newick_tree(tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]

    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    workflow = fit_compare_estim(toy_tree, trait, [:BM1, :OU1, :EB]; max_iterations = 30)
    @test !isempty(workflow.aic_table)
    @test workflow.best_fit !== nothing
    @test workflow.asr !== nothing
    @test workflow.asr.success

    workflow_bmm = fit_compare_estim(toy_tree, trait, [:BM1, :BMM]; edge_segments = edge_segments, max_iterations = 30)
    @test workflow_bmm.asr !== nothing
    @test workflow_bmm.asr.success

    workflow_oum = fit_compare_estim(toy_tree, trait, [:OU1, :OUM, :OUMV, :OUMA, :OUMVA, :EBM]; edge_segments = edge_segments, max_iterations = 30)
    @test workflow_oum.asr !== nothing
    @test workflow_oum.asr.success

    @test_throws ArgumentError fit_compare_estim(toy_tree, trait, [:BM1, :BM1]; max_iterations = 10)
    @test_throws ArgumentError fit_compare_estim(toy_tree, trait, [:BM1, :OUM]; max_iterations = 10)
end

@testset "Multivariate workflow wrappers" begin
    tree = to_compact_tree(simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(500)))
    Sigma = [
        0.7 0.15;
        0.15 0.5;
    ]
    bm_data = simulate_mvbm1(tree, Sigma; rng = MersenneTwister(501))

    bm_workflow = fit_compare_multivariate(tree, bm_data, [:mvBM1, :mvEB]; max_iterations = 40, rel_tol = 1e-6)
    @test bm_workflow.best_fit !== nothing
    @test bm_workflow.asr !== nothing
    @test bm_workflow.best_name in (:mvBM1, :mvEB)

    ou_simtree = simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(502))
    ou_tree = to_compact_tree(ou_simtree)
    ou_root = ou_tree.root
    edge_segments = [
        [SimmapSegment(
            state = Int32(ou_tree.parent_of_edge[e] == ou_root ? 1 : (iseven(e) ? 2 : 1)),
            length = ou_tree.edge_length[e],
        )]
        for e in 1:ou_tree.nedges
    ]
    A = [
        0.9 0.1;
        0.1 1.0;
    ]
    theta_regimes = [[0.0, 0.2], [0.8, -0.1]]
    ou_data = simulate_mvoum(ou_tree, edge_segments, A, Sigma, theta_regimes; rng = MersenneTwister(503))

    ou_workflow = fit_compare_multivariate(
        ou_tree,
        ou_data,
        [:mvBM1, :mvBMM, :mvOUM];
        edge_segments = edge_segments,
        max_iterations = 40,
        rel_tol = 1e-6,
    )
    @test ou_workflow.best_name in (:mvBM1, :mvBMM, :mvOUM)
    @test ou_workflow.asr !== nothing
end
