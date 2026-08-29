@testset "mvBM1 fitting" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(401)))
    Sigma_true = [
        0.8  0.25;
        0.25 0.6;
    ]
    traits = simulate_mvbm1(tree, Sigma_true; rng = MersenneTwister(402))

    fixed = mvbm1_loglikelihood(tree, traits, Sigma_true)
    @test fixed.success
    @test isfinite(fixed.loglik)
    @test fixed.model == :mvBM1
    @test fixed.ntraits == 2
    @test size(fixed.sigma) == (2, 2)
    @test length(fixed.theta) == 2

    fit = fit_mvbm1(tree, traits; max_iterations = 100, rel_tol = 1e-6)
    @test fit.success
    @test fit.model == :mvBM1
    @test fit.ntraits == 2
    @test size(fit.sigma) == (2, 2)
    @test isapprox(fit.sigma, fit.sigma'; atol = 1e-8)
    @test minimum(eigvals(Symmetric(fit.sigma))) > 0.0
    @test length(fit.theta) == 2
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
end

@testset "mvBM1 p>=2 and missing likelihood" begin
    tree = serialize_tree(simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(310)))
    Sigma3 = [
        0.8 0.2 0.1;
        0.2 0.7 0.15;
        0.1 0.15 0.6;
    ]
    traits3 = simulate_mvbm1(tree, Sigma3; rng = MersenneTwister(311))
    fixed3 = mvbm1_loglikelihood(tree, traits3, Sigma3)
    @test fixed3.success
    @test fixed3.ntraits == 3
    @test length(fixed3.theta) == 3
    @test isfinite(fixed3.loglik)

    missing_traits = copy(traits3)
    missing_traits[2, 1] = NaN
    missing_traits[5, 2] = NaN
    missing_traits[7, :] .= NaN
    missing_fixed = mvbm1_loglikelihood(tree, missing_traits, Sigma3)
    @test missing_fixed.success
    @test missing_fixed.ntraits == 3
    @test isfinite(missing_fixed.loglik)

    missing_fit = fit_mvbm1(tree, missing_traits; guess_sigma = Sigma3, max_iterations = 30, rel_tol = 1e-6)
    @test missing_fit.success
    missing_asr = estim_node(tree, missing_traits, missing_fit)
    @test missing_asr.success
    @test size(missing_asr.estimates) == (tree.ntips - 1, 3)
    @test size(missing_asr.all_node_estimates) == (tree.nnodes, 3)
    @test all(isfinite, missing_asr.estimates)
    @test all(isfinite, missing_asr.all_node_estimates)
end

@testset "mvBM1 ancestral reconstruction" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(403)))
    Sigma = [
        0.8 0.25;
        0.25 0.6;
    ]
    traits = simulate_mvbm1(tree, Sigma; rng = MersenneTwister(404))

    fit = fit_mvbm1(tree, traits; max_iterations = 100, rel_tol = 1e-6)
    asr = estim_node(tree, traits, fit)
    @test asr.success
    @test asr.model == :mvBM1
    @test length(asr.node_ids) == tree.ntips - 1
    @test size(asr.estimates) == (tree.ntips - 1, 2)
    @test size(asr.node_covariances) == (tree.ntips - 1, 2, 2)
    @test size(asr.all_node_estimates) == (tree.nnodes, 2)
    for i in axes(asr.node_covariances, 1)
        cov_i = Symmetric(asr.node_covariances[i, :, :])
        @test isapprox(Matrix(cov_i), Matrix(cov_i)'; atol = 1e-8)
        @test minimum(eigvals(cov_i)) >= -1e-8
    end
end


@testset "mvBMM fitting and ancestral reconstruction" begin
    simtree = simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(405))
    tree = serialize_tree(simtree)
    edge_segments = [[SimmapSegment(state = Int32(iseven(e) ? 2 : 1), length = tree.edge_length[e])] for e in 1:tree.nedges]
    Sig1 = [
        0.8 0.25;
        0.25 0.6;
    ]
    Sig2 = [
        0.45 0.12;
        0.12 0.7;
    ]
    traits = simulate_mvbmm(tree, edge_segments, [Sig1, Sig2]; rng = MersenneTwister(406))
    mapped = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges, seg in edge_segments[edge]
        mapped[edge, Int(seg.state)] += seg.length
    end

    fixed = mvbmm_loglikelihood(tree, traits, mapped, [Sig1, Sig2])
    @test fixed.success
    @test fixed.model == :mvBMM
    @test fixed.ntraits == 2
    @test fixed.nregimes == 2
    @test size(fixed.sigma) == (2, 2, 2)
    @test size(fixed.theta) == (1, 2)
    @test isfinite(fixed.loglik)

    fit = fit_mvbmm(tree, traits, mapped; guess_sigma = [Sig1, Sig2], max_iterations = 80, rel_tol = 1e-6)
    @test fit.success
    @test fit.model == :mvBMM
    @test fit.ntraits == 2
    @test fit.nregimes == 2
    @test size(fit.sigma) == (2, 2, 2)
    @test size(fit.theta) == (1, 2)
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
    for r in 1:2
        Sigma_r = Symmetric(fit.sigma[:, :, r])
        @test minimum(eigvals(Sigma_r)) > 0.0
    end

    simmap = SimmapSample(
        success = true,
        nstates = 2,
        state_labels = ["blue", "red"],
        edge_segments = edge_segments,
        mapped_edge = mapped,
    )
    simmap_fit = fit_mvbmm(tree, traits, simmap; trait_names = ["x", "y"], guess_sigma = [Sig1, Sig2], max_iterations = 40, rel_tol = 1e-6)
    @test simmap_fit.success
    @test simmap_fit.trait_names == ["x", "y"]
    @test simmap_fit.regime_names == ["blue", "red"]

    asr = estim_node(tree, traits, fit; edge_segments = edge_segments)
    @test asr.success
    @test asr.model == :mvBMM
    @test length(asr.node_ids) == tree.ntips - 1
    @test size(asr.estimates) == (tree.ntips - 1, 2)
    @test size(asr.node_covariances) == (tree.ntips - 1, 2, 2)
end

@testset "multivariate BM family initializes without complete rows" begin
    tree_path = joinpath(mktempdir(), "mv_missing_no_complete_rows.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    traits = [
        1.0 NaN;
        NaN 1.2;
        2.4 NaN;
        NaN 2.8;
    ]
    mapped = [
        1.0 0.0;
        1.0 0.0;
        0.0 1.0;
        0.0 1.0;
        1.0 0.0;
        0.0 1.0;
    ]

    bm1 = fit_mvbm1(tree, traits; max_iterations = 20, rel_tol = 1e-6)
    @test bm1.success
    @test isfinite(bm1.loglik)

    bmm = fit_mvbmm(tree, traits, mapped; max_iterations = 20, rel_tol = 1e-6)
    @test bmm.success
    @test isfinite(bmm.loglik)
end

@testset "mvEB fitting and ancestral reconstruction" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(407)))
    Sigma = [
        0.8 0.25;
        0.25 0.6;
    ]

    eb_traits = simulate_mveb(tree, Sigma, -0.5; rng = MersenneTwister(21))
    eb_fit = fit_mveb(tree, eb_traits; max_iterations = 60, rel_tol = 1e-6)
    @test eb_fit.success
    @test eb_fit.model == :mvEB
    @test eb_fit.ntraits == 2
    @test isfinite(eb_fit.loglik)
    eb_asr = estim_node(tree, eb_traits, eb_fit)
    @test eb_asr.success
    @test size(eb_asr.estimates, 2) == 2

end






