@testset "OU1 fitting" begin
    ou_tree_path = joinpath(mktempdir(), "toy_ou1_tree.tre")
    write(ou_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(ou_tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]

    ll = ou1_loglikelihood(toy_tree, trait, 0.5, 0.3, 1.5)
    @test ll.success
    @test isfinite(ll.loglik)
    @test ll.nparams == 3

    fit = fit_ou1(toy_tree, trait; max_iterations = 40)
    @test fit.success
    @test fit.model == :OU1
    @test fit.alpha > 0.0
    @test isfinite(fit.theta)
    @test fit.sigma2 > 0.0
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
    @test fit.root_mean_mode == :theta
    @test fit.root_cov_mode == :fixed
end

@testset "OU missing likelihood" begin
    tree_path = joinpath(mktempdir(), "toy_ou_missing_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(tree_path))
    trait = [1.0, NaN, 2.0, 2.1]

    ou = ou1_loglikelihood(toy_tree, trait, 0.5, 0.3, 1.5)
    @test ou.success
    @test isfinite(ou.loglik)
    fit = fit_ou1(toy_tree, trait; max_iterations = 40)
    @test fit.success
    @test isfinite(fit.loglik)
    asr = estim_node(toy_tree, trait, fit)
    @test asr.success
    @test all(isfinite, asr.estimates)
end

@testset "OU1 smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = serialize_tree(read_tree(tree_path))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))

    fit = fit_ou1(big_tree, trait; max_iterations = 20)
    @test fit.success
    @test fit.model == :OU1
    @test fit.sigma2 > 0.0
    @test isfinite(fit.loglik)
end


@testset "OUM fitting" begin
    oum_tree_path = joinpath(mktempdir(), "toy_oum_tree.tre")
    write(oum_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(oum_tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    ll = oum_loglikelihood(toy_tree, trait, edge_segments, 0.5, 0.3, [1.0, 2.0])
    @test ll.success
    @test isfinite(ll.loglik)
    @test ll.nregimes == 2
    @test ll.nparams == 4

    fit = fit_oum(toy_tree, trait, edge_segments; max_iterations = 40)
    @test fit.success
    @test fit.model == :OUM
    @test fit.nregimes == 2
    @test fit.alpha > 0.0
    @test length(fit.theta_regimes) == 2
    @test length(fit.sigma2) == 1
    @test fit.sigma2[1] > 0.0
    @test isfinite(fit.theta)
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
    @test fit.root_mean_mode == :root_regime_theta
    @test fit.root_cov_mode == :fixed

    node_asr = estim_node(toy_tree, trait, fit; edge_segments = edge_segments)
    @test node_asr.success
    branch_asr = estim_branch_for_simmap(toy_tree, trait, fit; edge_segments = edge_segments)
    @test branch_asr.success
    @test length(branch_asr.edge_ids) == sum(length, edge_segments)
    @test length(branch_asr.start_estimates) == sum(length, edge_segments)
    @test length(branch_asr.midpoint_estimates) == sum(length, edge_segments)
    @test length(branch_asr.end_estimates) == sum(length, edge_segments)
    @test all(isfinite, branch_asr.start_estimates)
    @test all(isfinite, branch_asr.midpoint_estimates)
    @test all(isfinite, branch_asr.end_estimates)
    @test all(>=(0.0), branch_asr.start_variances)
    @test all(>=(0.0), branch_asr.midpoint_variances)
    @test all(>=(0.0), branch_asr.end_variances)
end

@testset "OUM DataFrame and SIMMAP entrypoint" begin
    tree_path = joinpath(mktempdir(), "toy_oum_dataframe_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(tree_path))
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        state_labels = ["simple", "scalariform"],
        edge_segments = edge_segments,
        mapped_edge = [1.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 1.0],
    )
    df = DataFrame(taxon = ["A", "B", "C", "D"], lnVD = [1.0, 1.2, 2.4, 2.7])

    fit = fit_oum(toy_tree, df, simmap; max_iterations = 20)
    @test fit.success
    @test fit.trait_name == "lnVD"
    @test fit.regime_names == ["simple", "scalariform"]
    shown = sprint(show, MIME("text/plain"), fit)
    @test occursin("simple", shown)
    @test occursin("scalariform", shown)

    trait = [1.0, 1.2, 2.4, 2.7]
    vector_fit = fit_oum(toy_tree, trait, simmap; max_iterations = 20, trait_name = "lnVD")
    @test vector_fit.success
    @test vector_fit.trait_name == "lnVD"
    @test vector_fit.regime_names == ["simple", "scalariform"]
end


@testset "OUMV fitting" begin
    oum_tree_path = joinpath(mktempdir(), "toy_oumv_tree.tre")
    write(oum_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(oum_tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    ll = oumv_loglikelihood(toy_tree, trait, edge_segments, 0.5, [0.2, 0.4], [1.0, 2.0])
    @test ll.success
    @test isfinite(ll.loglik)
    @test ll.model == :OUMV
    @test ll.nregimes == 2
    @test ll.nparams == 5
    @test length(ll.sigma2) == 2
    @test length(ll.theta_regimes) == 2
    @test ll.alpha > 0.0

    fit = fit_oumv(toy_tree, trait, edge_segments; max_iterations = 40)
    @test fit.success
    @test fit.model == :OUMV
    @test fit.nregimes == 2
    @test fit.alpha > 0.0
    @test length(fit.sigma2) == 2
    @test all(>(0.0), fit.sigma2)
    @test length(fit.theta_regimes) == 2
    @test fit.root_mean_mode == :root_regime_theta
    @test fit.root_cov_mode == :fixed
end

@testset "OUMA and OUMVA fitting" begin
    oum_tree_path = joinpath(mktempdir(), "toy_ouma_tree.tre")
    write(oum_tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(oum_tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    ll_ouma = ouma_loglikelihood(toy_tree, trait, edge_segments, [0.4, 0.8], 0.3, [1.0, 2.0])
    @test ll_ouma.success
    @test ll_ouma.model == :OUMA
    @test length(ll_ouma.alpha_regimes) == 2
    @test isnan(ll_ouma.alpha)
    @test length(ll_ouma.sigma2) == 1

    ouma_fit = fit_ouma(toy_tree, trait, edge_segments; max_iterations = 40)
    @test ouma_fit.success
    @test ouma_fit.model == :OUMA
    @test length(ouma_fit.alpha_regimes) == 2
    @test all(>(0.0), ouma_fit.alpha_regimes)
    @test length(ouma_fit.sigma2) == 1
    @test length(ouma_fit.theta_regimes) == 2

    ll_oumva = oumva_loglikelihood(toy_tree, trait, edge_segments, [0.4, 0.8], [0.2, 0.4], [1.0, 2.0])
    @test ll_oumva.success
    @test ll_oumva.model == :OUMVA
    @test length(ll_oumva.alpha_regimes) == 2
    @test length(ll_oumva.sigma2) == 2
    @test length(ll_oumva.theta_regimes) == 2

    oumva_fit = fit_oumva(toy_tree, trait, edge_segments; max_iterations = 40)
    @test oumva_fit.success
    @test oumva_fit.model == :OUMVA
    @test length(oumva_fit.alpha_regimes) == 2
    @test all(>(0.0), oumva_fit.alpha_regimes)
    @test length(oumva_fit.sigma2) == 2
    @test all(>(0.0), oumva_fit.sigma2)
    @test length(oumva_fit.theta_regimes) == 2

    simmap = SimmapSample(
        success = true,
        nstates = 2,
        state_labels = ["blue", "red"],
        edge_segments = edge_segments,
        mapped_edge = [1.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 1.0],
    )
    @test fit_oumv(toy_tree, trait, simmap; max_iterations = 40).regime_names == ["blue", "red"]
    @test fit_ouma(toy_tree, trait, simmap; max_iterations = 40).regime_names == ["blue", "red"]
    @test fit_oumva(toy_tree, trait, simmap; max_iterations = 40).regime_names == ["blue", "red"]
end

@testset "OUM smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = serialize_tree(read_tree(tree_path))
    states = Int32[mod1(i, 4) for i in 1:big_tree.ntips]
    priors = tip_priors_from_states(big_tree, states, 4)
    Q = [
        -0.30 0.10 0.10 0.10;
         0.10 -0.30 0.10 0.10;
         0.10 0.10 -0.30 0.10;
         0.10 0.10 0.10 -0.30;
    ]
    simmap = simmap_sample(big_tree, priors, Q; root_prior = :flat, rng = MersenneTwister(123))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))

    fit = fit_oum(big_tree, trait, simmap.edge_segments; max_iterations = 20)
    @test fit.success
    @test fit.model == :OUM
    @test fit.nregimes == 4
    @test fit.alpha > 0.0
    @test length(fit.theta_regimes) == 4
    @test fit.sigma2[1] > 0.0
    @test isfinite(fit.loglik)
end

@testset "OUM tolerates tiny simmap residue on zero-length internal edge" begin
    tree_path = joinpath(mktempdir(), "toy_oum_zero_internal.tre")
    write(tree_path, "((A:1,B:1):1,(C:2,D:2):0);")
    tree = serialize_tree(read_tree(tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]

    edge_segments = [SimmapSegment[] for _ in 1:tree.nedges]
    for edge in 1:tree.nedges
        child = Int(tree.child_of_edge[edge])
        state = tree.is_tip[child] ? 2 : 1
        len = tree.edge_length[edge]
        if len == 0.0 && !tree.is_tip[child]
            edge_segments[edge] = [SimmapSegment(state = Int32(state), length = 1.4901161193847656e-8)]
        else
            edge_segments[edge] = [SimmapSegment(state = Int32(state), length = len)]
        end
    end

    fit = fit_oum(tree, trait, edge_segments; max_iterations = 20, rel_tol = 1e-6)
    @test fit.success
    @test isfinite(fit.loglik)
end

@testset "Univariate OU multistart preserves the serial optimum" begin
    objective = x -> (x[1] - 1.75)^2 + (x[2] + 0.25)^2
    candidates = [
        [-4.0, 3.0],
        [2.0, 2.0],
        [0.5, -1.0],
        [4.0, -3.0],
    ]
    serial = EvoTraits._continuous_two_stage_multistart_serial(
        objective,
        candidates;
        max_iterations = 20,
        polish_iterations = 20,
        rel_tol = 1e-8,
    )
    automatic = EvoTraits._continuous_two_stage_multistart(
        objective,
        candidates;
        max_iterations = 20,
        polish_iterations = 20,
        rel_tol = 1e-8,
    )
    @test isapprox(EvoTraits._continuous_result_minimum(automatic), EvoTraits._continuous_result_minimum(serial); atol = 1e-10)
    @test isapprox(EvoTraits._continuous_result_minimizer(automatic), EvoTraits._continuous_result_minimizer(serial); atol = 1e-8)
end






