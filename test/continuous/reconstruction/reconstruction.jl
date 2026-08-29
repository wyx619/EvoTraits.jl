@testset "Branch posterior for BM-derived multi-regime models" begin
    tree_path = joinpath(mktempdir(), "toy_branch_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    trait = [1.0, 1.2, 1.8, 2.0]
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 0.4), SimmapSegment(state = 2, length = 0.6)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 0.5), SimmapSegment(state = 1, length = 0.5)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    ebm_fit = fit_ebm(tree, trait, edge_segments; max_iterations = 30)
    ebm_branch = estim_branch_for_simmap(tree, trait, ebm_fit; edge_segments = edge_segments)
    @test ebm_branch.success
    @test ebm_branch.model == :EBM
    @test length(ebm_branch.start_estimates) == sum(length, edge_segments)
    @test all(isfinite, ebm_branch.end_estimates)
end

@testset "estim_branch_for_simmap accepts SimmapSample directly" begin
    tree_path = joinpath(mktempdir(), "toy_branch_simmap_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))

    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    mapped_edge = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges, seg in edge_segments[edge]
        mapped_edge[edge, Int(seg.state)] += seg.length
    end
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        root_state = Int32(1),
        state_labels = ["regime1", "regime2"],
        node_states = Int32[1, 1, 2, 1, 2, 2, 2],
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
    )

    trait_uni = [1.0, 1.1, 2.0, 2.1]
    fit_ebm_res = fit_ebm(tree, trait_uni, edge_segments; max_iterations = 20)
    fit_oum_res = fit_oum(tree, trait_uni, edge_segments; max_iterations = 20)
    branch_ebm_kw = estim_branch_for_simmap(tree, trait_uni, fit_ebm_res; edge_segments = edge_segments)
    branch_ebm_simmap = estim_branch_for_simmap(tree, trait_uni, fit_ebm_res, simmap)
    branch_oum_kw = estim_branch_for_simmap(tree, trait_uni, fit_oum_res; edge_segments = edge_segments)
    branch_oum_simmap = estim_branch_for_simmap(tree, trait_uni, fit_oum_res, simmap)
    @test isapprox(branch_ebm_simmap.start_estimates, branch_ebm_kw.start_estimates; atol = 1e-10)
    @test isapprox(branch_oum_simmap.start_estimates, branch_oum_kw.start_estimates; atol = 1e-10)

    trait_mv = [
        1.0 0.2;
        1.2 0.4;
        1.8 1.0;
        2.0 1.3;
    ]
    sigma1 = [
        0.8 0.2;
        0.2 0.6;
    ]
    sigma2 = [
        0.5 0.1;
        0.1 0.7;
    ]
    fit_mvoum_res = fit_mvoum(tree, trait_mv, edge_segments; max_iterations = 20, rel_tol = 1e-5)
    fit_mvbmm_res = mvbmm_loglikelihood(tree, trait_mv, mapped_edge, [sigma1, sigma2])
    branch_mvoum_kw = estim_branch_for_simmap(tree, trait_mv, fit_mvoum_res; edge_segments = edge_segments)
    branch_mvoum_simmap = estim_branch_for_simmap(tree, trait_mv, fit_mvoum_res, simmap)
    branch_mvbmm_kw = estim_branch_for_simmap(tree, trait_mv, fit_mvbmm_res; edge_segments = edge_segments)
    branch_mvbmm_simmap = estim_branch_for_simmap(tree, trait_mv, fit_mvbmm_res, simmap)
    @test isapprox(branch_mvoum_simmap.start_estimates, branch_mvoum_kw.start_estimates; atol = 1e-10)
    @test isapprox(branch_mvbmm_simmap.start_estimates, branch_mvbmm_kw.start_estimates; atol = 1e-10)
end

@testset "Multivariate OU branch posterior" begin
    simtree = simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(510))
    tree = serialize_tree(simtree)
    root = tree.root
    edge_segments = [
        [SimmapSegment(
            state = Int32(tree.parent_of_edge[e] == root ? 1 : (iseven(e) ? 2 : 1)),
            length = tree.edge_length[e],
        )]
        for e in 1:tree.nedges
    ]
    A = [
        0.9 0.1;
        0.1 1.0;
    ]
    Sigma = [
        0.7 0.15;
        0.15 0.5;
    ]
    theta_regimes = [[0.0, 0.2], [0.8, -0.1]]
    trait = simulate_mvoum(tree, edge_segments, A, Sigma, theta_regimes; rng = MersenneTwister(511))
    fit = fit_mvoum(tree, trait, edge_segments; max_iterations = 20, rel_tol = 1e-5)
    branch = estim_branch_for_simmap(tree, trait, fit; edge_segments = edge_segments)
    @test branch.success
    @test branch.model == :mvOUM
    @test length(branch.edge_ids) == sum(length, edge_segments)
    @test size(branch.start_estimates, 2) == 2
    @test size(branch.midpoint_estimates, 2) == 2
    @test size(branch.end_estimates, 2) == 2
    @test size(branch.start_covariances, 2) == 2
    @test size(branch.end_covariances, 3) == 2

    missing_trait = copy(trait)
    missing_trait[3, 1] = NaN
    missing_trait[7, 2] = NaN
    missing_trait[11, :] .= NaN
    missing_branch = estim_branch_for_simmap(tree, missing_trait, fit; edge_segments = edge_segments)
    @test missing_branch.success
    @test length(missing_branch.edge_ids) == sum(length, edge_segments)
    @test all(isfinite, missing_branch.start_estimates)
    @test all(isfinite, missing_branch.midpoint_estimates)
    @test all(isfinite, missing_branch.end_estimates)
end

@testset "Advanced multivariate OU branch posterior" begin
    simtree = simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(512))
    tree = serialize_tree(simtree)
    root = tree.root
    edge_segments = [
        [SimmapSegment(
            state = Int32(tree.parent_of_edge[e] == root ? 1 : (iseven(e) ? 2 : 1)),
            length = tree.edge_length[e],
        )]
        for e in 1:tree.nedges
    ]
    A = [
        0.9 0.1;
        0.1 1.0;
    ]
    A_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    A_regimes[:, :, 1] = [
        0.85 0.08;
        0.08 1.05;
    ]
    A_regimes[:, :, 2] = [
        1.15 0.04;
        0.04 0.95;
    ]
    Sigma = [
        0.7 0.15;
        0.15 0.5;
    ]
    Sigma_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    Sigma_regimes[:, :, 1] = [
        0.65 0.12;
        0.12 0.45;
    ]
    Sigma_regimes[:, :, 2] = [
        0.55 0.10;
        0.10 0.70;
    ]
    theta_regimes = [
        0.0 0.2;
        0.8 -0.1;
    ]
    trait = simulate_mvoum(tree, edge_segments, A, Sigma, eachrow(theta_regimes) |> collect; rng = MersenneTwister(513))

    fits = (
        mvoumv_loglikelihood(tree, trait, edge_segments, A, Sigma_regimes, theta_regimes),
        mvouma_loglikelihood(tree, trait, edge_segments, A_regimes, Sigma, theta_regimes),
        mvoumva_loglikelihood(tree, trait, edge_segments, A_regimes, Sigma_regimes, theta_regimes),
    )
    missing_trait = copy(trait)
    missing_trait[2, 1] = NaN
    missing_trait[5, 2] = NaN
    missing_trait[9, :] .= NaN

    for fit in fits
        branch = estim_branch_for_simmap(tree, trait, fit; edge_segments = edge_segments)
        @test branch.success
        @test branch.model == fit.model
        @test length(branch.edge_ids) == sum(length, edge_segments)
        @test size(branch.start_estimates, 2) == 2
        @test size(branch.start_covariances, 2) == 2
        @test all(isfinite, branch.start_estimates)
        @test all(isfinite, branch.midpoint_estimates)
        @test all(isfinite, branch.end_estimates)

        missing_branch = estim_branch_for_simmap(tree, missing_trait, fit; edge_segments = edge_segments)
        @test missing_branch.success
        @test all(isfinite, missing_branch.start_estimates)
        @test all(isfinite, missing_branch.midpoint_estimates)
        @test all(isfinite, missing_branch.end_estimates)
    end
end

@testset "Multivariate BM branch posterior" begin
    tree_path = joinpath(mktempdir(), "toy_mvbm_branch_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    trait = [
        1.0 0.2;
        1.2 0.4;
        1.8 1.0;
        2.0 1.3;
    ]
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 0.4), SimmapSegment(state = 2, length = 0.6)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 0.5), SimmapSegment(state = 1, length = 0.5)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    mapped = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges, seg in edge_segments[edge]
        mapped[edge, Int(seg.state)] += seg.length
    end
    sigma1 = [
        0.8 0.2;
        0.2 0.6;
    ]
    sigma2 = [
        0.5 0.1;
        0.1 0.7;
    ]

    bm_fit = mvbm1_loglikelihood(tree, trait, sigma1)
    eb_fit = mveb_loglikelihood(tree, trait, sigma1, -0.3)
    bmm_fit = mvbmm_loglikelihood(tree, trait, mapped, [sigma1, sigma2])

    for fit in (bm_fit, eb_fit, bmm_fit)
        branch = estim_branch_for_simmap(tree, trait, fit; edge_segments = edge_segments)
        @test branch.success
        @test branch.model == fit.model
        @test length(branch.edge_ids) == sum(length, edge_segments)
        @test size(branch.start_estimates, 2) == 2
        @test size(branch.midpoint_estimates, 2) == 2
        @test size(branch.end_estimates, 2) == 2
        @test size(branch.start_covariances, 2) == 2
        @test all(isfinite, branch.start_estimates)
        @test all(isfinite, branch.midpoint_estimates)
        @test all(isfinite, branch.end_estimates)

        missing_trait = copy(trait)
        missing_trait[2, 1] = NaN
        missing_trait[4, :] .= NaN
        missing_branch = estim_branch_for_simmap(tree, missing_trait, fit; edge_segments = edge_segments)
        @test missing_branch.success
        @test all(isfinite, missing_branch.start_estimates)
        @test all(isfinite, missing_branch.midpoint_estimates)
        @test all(isfinite, missing_branch.end_estimates)
    end
end


@testset "Missing tip reconstruction fields" begin
    tree_path = joinpath(mktempdir(), "toy_missing_tip_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))

    trait = [1.0, NaN, 2.0, 2.2]
    fit = fit_bm1(tree, trait; max_iterations = 30)
    asr = estim_node(tree, trait, fit)
    @test asr.success
    @test asr.tip_ids == Int32.(tree.tip_ids)
    @test asr.tip_labels == tree.node_labels[tree.tip_ids]
    @test length(asr.time_from_root) == tree.ntips - 1
    @test length(asr.time_before_present) == tree.ntips - 1
    @test length(asr.tip_estimates) == tree.ntips
    @test length(asr.tip_variances) == tree.ntips
    @test asr.tip_observed == .!isnan.(trait)
    @test isfinite(asr.tip_estimates[2])
    @test isfinite(asr.tip_variances[2])
    @test asr.tip_variances[2] > 0.0

    mv_trait = [
        1.0 0.2;
        NaN 0.4;
        1.8 NaN;
        NaN NaN;
    ]
    sigma = [
        0.7 0.1;
        0.1 0.5;
    ]
    mv_fit = mvbm1_loglikelihood(tree, mv_trait, sigma)
    mv_asr = estim_node(tree, mv_trait, mv_fit)
    @test mv_asr.success
    @test mv_asr.tip_ids == Int32.(tree.tip_ids)
    @test mv_asr.tip_labels == tree.node_labels[tree.tip_ids]
    @test length(mv_asr.time_from_root) == tree.ntips - 1
    @test length(mv_asr.time_before_present) == tree.ntips - 1
    @test size(mv_asr.tip_estimates) == size(mv_trait)
    @test size(mv_asr.tip_covariances) == (tree.ntips, 2, 2)
    @test size(mv_asr.all_node_covariances) == (tree.nnodes, 2, 2)
    @test mv_asr.tip_observed_mask == .!isnan.(mv_trait)
    @test all(isfinite, mv_asr.tip_estimates)
    @test all(isfinite, mv_asr.tip_covariances)
end


@testset "Continuous ancestral reconstruction" begin
    tree_path = joinpath(mktempdir(), "toy_estim_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    toy_tree = serialize_tree(read_tree(tree_path))
    trait = [1.0, 1.1, 2.0, 2.1]

    mapped = [
        1.0 0.0;
        1.0 0.0;
        0.0 1.0;
        0.0 1.0;
        1.0 0.0;
        0.0 1.0;
    ]

    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    bm_fit = fit_bm1(toy_tree, trait; max_iterations = 30)
    bmm_fit = fit_bmm(toy_tree, trait, mapped; max_iterations = 30)
    ou_fit = fit_ou1(toy_tree, trait; max_iterations = 30)
    oum_fit = fit_oum(toy_tree, trait, edge_segments; max_iterations = 30)
    oumv_fit = fit_oumv(toy_tree, trait, edge_segments; max_iterations = 30)
    ouma_fit = fit_ouma(toy_tree, trait, edge_segments; max_iterations = 30)
    oumva_fit = fit_oumva(toy_tree, trait, edge_segments; max_iterations = 30)
    ebm_fit = fit_ebm(toy_tree, trait, edge_segments; max_iterations = 30)
    eb_fit = fit_eb(toy_tree, trait; max_iterations = 30)

    bm_asr = estim_node(toy_tree, trait, bm_fit)
    bmm_asr = estim_node(toy_tree, trait, bmm_fit; edge_segments = edge_segments)
    bmm_compat_asr = estim_node(toy_tree, trait, bmm_fit; mapped_edge = mapped)
    ou_asr = estim_node(toy_tree, trait, ou_fit)
    oum_asr = estim_node(toy_tree, trait, oum_fit; edge_segments = edge_segments)
    oumv_asr = estim_node(toy_tree, trait, oumv_fit; edge_segments = edge_segments)
    ouma_asr = estim_node(toy_tree, trait, ouma_fit; edge_segments = edge_segments)
    oumva_asr = estim_node(toy_tree, trait, oumva_fit; edge_segments = edge_segments)
    ebm_asr = estim_node(toy_tree, trait, ebm_fit; edge_segments = edge_segments)
    eb_asr = estim_node(toy_tree, trait, eb_fit)

    @test isapprox(bmm_asr.estimates, bmm_compat_asr.estimates; atol = 1e-10)

    for asr in (bm_asr, bmm_asr, ou_asr, oum_asr, oumv_asr, ouma_asr, oumva_asr, ebm_asr, eb_asr)
        @test asr.success
        @test length(asr.node_ids) == toy_tree.ntips - 1
        @test length(asr.estimates) == toy_tree.ntips - 1
        @test all(isfinite, asr.estimates)
        @test all(x -> isfinite(x) && x >= 0.0, asr.variances)
        @test length(asr.all_node_estimates) == toy_tree.nnodes
        @test length(asr.all_node_variances) == toy_tree.nnodes
    end

    root_index = findfirst(==(toy_tree.root), bm_asr.node_ids)
    @test root_index !== nothing
    @test isapprox(bm_asr.estimates[root_index], bm_fit.root_state; atol = 1e-8)
end

@testset "estim_node accepts SimmapSample directly" begin
    tree_path = joinpath(mktempdir(), "toy_estim_simmap_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))

    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    mapped_edge = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges, seg in edge_segments[edge]
        mapped_edge[edge, Int(seg.state)] += seg.length
    end
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        root_state = Int32(1),
        state_labels = ["regime1", "regime2"],
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
    )

    trait_uni = [1.0, 1.1, 2.0, 2.1]
    fit_bmm_res = fit_bmm(tree, trait_uni, mapped_edge; max_iterations = 30)
    fit_oum_res = fit_oum(tree, trait_uni, edge_segments; max_iterations = 30)
    asr_bmm_kw = estim_node(tree, trait_uni, fit_bmm_res; mapped_edge = mapped_edge)
    asr_bmm_simmap = estim_node(tree, trait_uni, fit_bmm_res, simmap)
    asr_oum_kw = estim_node(tree, trait_uni, fit_oum_res; edge_segments = edge_segments)
    asr_oum_simmap = estim_node(tree, trait_uni, fit_oum_res, simmap)
    @test isapprox(asr_bmm_simmap.estimates, asr_bmm_kw.estimates; atol = 1e-10)
    @test isapprox(asr_oum_simmap.estimates, asr_oum_kw.estimates; atol = 1e-10)

    trait_mv = [
        1.0 0.2;
        1.2 0.4;
        1.8 1.0;
        2.0 1.3;
    ]
    sigma1 = [
        0.8 0.2;
        0.2 0.6;
    ]
    sigma2 = [
        0.5 0.1;
        0.1 0.7;
    ]
    fit_mvbmm_res = mvbmm_loglikelihood(tree, trait_mv, mapped_edge, [sigma1, sigma2])
    fit_mvoum_res = fit_mvoum(tree, trait_mv, edge_segments; max_iterations = 20, rel_tol = 1e-5)
    asr_mvbmm_kw = estim_node(tree, trait_mv, fit_mvbmm_res; mapped_edge = mapped_edge)
    asr_mvbmm_simmap = estim_node(tree, trait_mv, fit_mvbmm_res, simmap)
    asr_mvoum_kw = estim_node(tree, trait_mv, fit_mvoum_res; edge_segments = edge_segments)
    asr_mvoum_simmap = estim_node(tree, trait_mv, fit_mvoum_res, simmap)
    @test isapprox(asr_mvbmm_simmap.estimates, asr_mvbmm_kw.estimates; atol = 1e-10)
    @test isapprox(asr_mvoum_simmap.estimates, asr_mvoum_kw.estimates; atol = 1e-10)
end

@testset "estim_node_table integrates phyloref" begin
    tree_path = joinpath(mktempdir(), "toy_estim_table_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    map = build_phyloref(tree)
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    mapped_edge = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges, seg in edge_segments[edge]
        mapped_edge[edge, Int(seg.state)] += seg.length
    end
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        root_state = Int32(1),
        state_labels = ["regime1", "regime2"],
        node_states = Int32[1, 1, 2, 1, 2, 2, 2],
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
    )

    trait_uni = [1.0, 1.1, 2.0, 2.1]
    fit_uni = fit_bm1(tree, trait_uni; max_iterations = 20)
    asr_uni = estim_node(tree, trait_uni, fit_uni)
    tbl_uni = estim_node_table(tree, asr_uni; map = map)
    tbl_uni_simmap = estim_node_table(tree, asr_uni; map = map, simmap = simmap)
    @test nrow(tbl_uni) == tree.ntips - 1
    @test issubset(Set(["node_id", "R_node_id", "tipX", "tipY", "estimate", "variance", "se"]), Set(String.(names(tbl_uni))))
    @test "time_from_root" in names(tbl_uni)
    @test "time_before_present" in names(tbl_uni)
    @test Set(tbl_uni.R_node_id) == Set(EvoTraits.R_node_id(tree, Int(node); map = map) for node in asr_uni.node_ids)
    @test issubset(Set(["regime_id", "regime"]), Set(String.(names(tbl_uni_simmap))))
    expected_regime_id = [Int(simmap.node_states[node]) for node in tbl_uni_simmap.node_id]
    expected_regime = [simmap.state_labels[id] for id in expected_regime_id]
    @test tbl_uni_simmap.regime_id == expected_regime_id
    @test tbl_uni_simmap.regime == expected_regime

    trait_mv = [
        1.0 0.2;
        1.2 0.4;
        1.8 1.0;
        2.0 1.3;
    ]
    sigma = [
        0.7 0.1;
        0.1 0.5;
    ]
    fit_mv = mvbm1_loglikelihood(tree, trait_mv, sigma)
    asr_mv = estim_node(tree, trait_mv, fit_mv)
    tbl_mv = estim_node_table(tree, asr_mv; map = map)
    tbl_mv_named = estim_node_table(tree, asr_mv, fit_mv; map = map, simmap = simmap)
    @test nrow(tbl_mv) == tree.ntips - 1
    @test issubset(Set(["node_id", "R_node_id", "tipX", "tipY", "estimate_1", "estimate_2", "variance_1", "variance_2", "se_1", "se_2"]), Set(String.(names(tbl_mv))))
    @test "time_from_root" in names(tbl_mv)
    @test "time_before_present" in names(tbl_mv)
    @test Set(tbl_mv.R_node_id) == Set(EvoTraits.R_node_id(tree, Int(node); map = map) for node in asr_mv.node_ids)
    @test issubset(Set(["estimate_trait1", "estimate_trait2", "variance_trait1", "variance_trait2", "se_trait1", "se_trait2"]), Set(String.(names(tbl_mv_named))))
    @test issubset(Set(["regime_id", "regime"]), Set(String.(names(tbl_mv_named))))
    expected_mv_regime_id = [Int(simmap.node_states[node]) for node in tbl_mv_named.node_id]
    @test tbl_mv_named.regime_id == expected_mv_regime_id
end

@testset "estim_branch_table integrates phyloref" begin
    tree_path = joinpath(mktempdir(), "toy_branch_table_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    map = build_phyloref(tree)

    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]
    mapped_edge = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges, seg in edge_segments[edge]
        mapped_edge[edge, Int(seg.state)] += seg.length
    end
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        root_state = Int32(1),
        state_labels = ["regime1", "regime2"],
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
    )

    trait_uni = [1.0, 1.1, 2.0, 2.1]
    fit_uni = fit_oum(tree, trait_uni, edge_segments; max_iterations = 20)
    branch_uni = estim_branch_for_simmap(tree, trait_uni, fit_uni, simmap)
    tbl_uni = estim_branch_table(tree, branch_uni; map = map)
    @test nrow(tbl_uni) == sum(length, edge_segments)
    @test issubset(
        Set(["edge_id", "edge_R_id", "parent_R_node_id", "child_R_node_id", "tipX", "tipY", "descendant_signature", "start_estimate", "midpoint_estimate", "end_estimate"]),
        Set(String.(names(tbl_uni))),
    )
    @test "time_from_root_start" in names(tbl_uni)
    @test "time_before_present_end" in names(tbl_uni)
    @test !("absolute_start" in names(tbl_uni))
    @test !("absolute_end" in names(tbl_uni))
    @test !("absolute_midpoint" in names(tbl_uni))

    trait_mv = [
        1.0 0.2;
        1.2 0.4;
        1.8 1.0;
        2.0 1.3;
    ]
    fit_mv = fit_mvoum(tree, trait_mv, edge_segments; max_iterations = 20, rel_tol = 1e-5)
    branch_mv = estim_branch_for_simmap(tree, trait_mv, fit_mv, simmap)
    tbl_mv = estim_branch_table(tree, branch_mv; map = map)
    tbl_mv_named = estim_branch_table(tree, branch_mv, fit_mv; map = map)
    @test nrow(tbl_mv) == sum(length, edge_segments)
    @test issubset(
        Set(["edge_id", "edge_R_id", "parent_R_node_id", "child_R_node_id", "tipX", "tipY", "descendant_signature", "start_estimate_1", "start_estimate_2", "end_estimate_1", "end_estimate_2"]),
        Set(String.(names(tbl_mv))),
    )
    @test "time_from_root_midpoint" in names(tbl_mv)
    @test "time_before_present_start" in names(tbl_mv)
    @test !("absolute_start" in names(tbl_mv))
    @test !("absolute_end" in names(tbl_mv))
    @test !("absolute_midpoint" in names(tbl_mv))
    @test issubset(
        Set(["start_estimate_trait1", "start_estimate_trait2", "midpoint_estimate_trait1", "end_estimate_trait2"]),
        Set(String.(names(tbl_mv_named))),
    )
end

@testset "BM1 ancestral reconstruction smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = serialize_tree(read_tree(tree_path))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))

    fit = fit_bm1(big_tree, trait; max_iterations = 20)
asr = estim_node(big_tree, trait, fit)

    @test asr.success
    @test length(asr.node_ids) == big_tree.ntips - 1
    @test all(isfinite, asr.estimates)
end

@testset "OUM ancestral reconstruction smoke test on real tree" begin
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
    simmap = simmap_sample(big_tree, priors, Q; root_prior = :flat, rng = MersenneTwister(321))
    trait = collect(range(0.0, 1.0; length = big_tree.ntips))
    fit = fit_oum(big_tree, trait, simmap.edge_segments; max_iterations = 20)
asr = estim_node(big_tree, trait, fit; edge_segments = simmap.edge_segments)

    @test asr.success
    @test length(asr.node_ids) == big_tree.ntips - 1
    @test all(isfinite, asr.estimates)
end







