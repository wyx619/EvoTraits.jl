# Helper to build edge_segments for N regimes that satisfy the
# root-outgoing-edges-same-initial-state constraint.
function _test_edge_segments(tree::CompactTree, nregimes::Int)
    root = tree.root
    segs = Vector{Vector{SimmapSegment}}(undef, tree.nedges)
    for e in 1:tree.nedges
        if tree.parent_of_edge[e] == root
            state = 1
        else
            state = ((e - 1) % nregimes) + 1
        end
        segs[e] = [SimmapSegment(state = Int32(state), length = tree.edge_length[e])]
    end
    return segs
end

@testset "mvOU1 fitting and ancestral reconstruction" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(420)))

    A_true = [
        1.0 0.15;
        0.15 0.9;
    ]
    Sigma_true = [
        0.8 0.25;
        0.25 0.6;
    ]
    theta_true = [0.2, -0.1]
    traits = simulate_mvou1(tree, A_true, Sigma_true, theta_true; rng = MersenneTwister(225))

    fixed = mvou1_loglikelihood(tree, traits, A_true, Sigma_true, theta_true)
    @test fixed.success
    @test fixed.model == :mvOU1
    @test fixed.ntraits == 2
    @test size(fixed.A) == (2, 2, 1)
    @test size(fixed.Sigma) == (2, 2, 1)
    @test size(fixed.theta) == (1, 2)
    @test isfinite(fixed.loglik)

    fit = fit_mvou1(tree, traits; max_iterations = 80, rel_tol = 1e-6)
    @test fit.success
    @test fit.model == :mvOU1
    @test fit.ntraits == 2
    @test size(fit.A) == (2, 2, 1)
    @test size(fit.Sigma) == (2, 2, 1)
    @test size(fit.theta) == (1, 2)
    @test fit.root_mean_mode == :theta
    @test fit.root_cov_mode == :fixed
    @test minimum(eigvals(Symmetric(fit.A[:, :, 1]))) > 0.0
    @test minimum(eigvals(Symmetric(fit.Sigma[:, :, 1]))) > 0.0

    asr = estim_node(tree, traits, fit)
    @test asr.success
    @test asr.model == :mvOU1
    @test length(asr.node_ids) == tree.ntips - 1
    @test size(asr.estimates) == (tree.ntips - 1, 2)
    @test size(asr.node_covariances) == (tree.ntips - 1, 2, 2)

end

@testset "mvOU1 p>=2 and missing likelihood" begin
    tree = serialize_tree(simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(320)))
    A3 = [
        1.0 0.1 0.0;
        0.1 0.9 0.08;
        0.0 0.08 1.1;
    ]
    Sigma3 = [
        0.8 0.2 0.1;
        0.2 0.7 0.15;
        0.1 0.15 0.6;
    ]
    theta3 = [0.2, -0.1, 0.35]
    traits3 = simulate_mvou1(tree, A3, Sigma3, theta3; rng = MersenneTwister(321))
    fixed3 = mvou1_loglikelihood(tree, traits3, A3, Sigma3, theta3)
    @test fixed3.success
    @test fixed3.ntraits == 3
    @test size(fixed3.A) == (3, 3, 1)
    @test isfinite(fixed3.loglik)

    missing_traits = copy(traits3)
    missing_traits[3, 1] = NaN
    missing_traits[6, 3] = NaN
    missing_traits[9, :] .= NaN
    missing_fixed = mvou1_loglikelihood(tree, missing_traits, A3, Sigma3, theta3)
    @test missing_fixed.success
    @test missing_fixed.ntraits == 3
    @test isfinite(missing_fixed.loglik)

    missing_fit = fit_mvou1(
        tree,
        missing_traits;
        max_iterations = 30,
        rel_tol = 1e-6,
    )
    @test missing_fit.success
    missing_asr = estim_node(tree, missing_traits, missing_fit)
    @test missing_asr.success
    @test size(missing_asr.estimates) == (tree.ntips - 1, 3)
    @test size(missing_asr.all_node_estimates) == (tree.nnodes, 3)
    @test all(isfinite, missing_asr.estimates)
    @test all(isfinite, missing_asr.all_node_estimates)
end

@testset "mvOU1 near-BM alpha remains numerically valid" begin
    tree = serialize_tree(simulate_yule_simtree(40; tree_height = 1.0, rng = MersenneTwister(326)))
    A = [
        1.0681614698559785 -0.14938292367880018 0.023523131284298147;
        -0.14938292367880018 2.057137236721755 0.014946846700781051;
        0.023523131284298147 0.014946846700781051 0.0006814504191377855;
    ]
    Sigma = [
        0.9594705738576089 -0.11600454912447339 0.003452658636509885;
        -0.11600454912447339 0.6264525042174498 0.00208287579991228;
        0.003452658636509885 0.00208287579991228 0.10424604447740397;
    ]
    theta = [3.9248426837990906, 1.987459530806447, -0.06310212097264915]

    @test minimum(eigvals(Symmetric(A))) > 0.0
    traits = simulate_mvou1(tree, A, Sigma, theta; rng = MersenneTwister(327))
    fixed = mvou1_loglikelihood(tree, traits, A, Sigma, theta)
    @test fixed.success
    @test isfinite(fixed.loglik)
end

@testset "mvOU1 initializes without complete rows" begin
    tree_path = joinpath(mktempdir(), "mvou_missing_no_complete_rows.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    traits = [
        1.0 NaN;
        NaN 1.2;
        2.4 NaN;
        NaN 2.8;
    ]
    fit = fit_mvou1(tree, traits; max_iterations = 20, rel_tol = 1e-6)
    @test fit.success
    @test isfinite(fit.loglik)
end

@testset "mvOUM diagonal likelihood uses stationary design" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(421)))

    edge_segments = _test_edge_segments(tree, 2)
    rng = MersenneTwister(233)
    trait1 = randn(rng, tree.ntips)
    trait2 = randn(rng, tree.ntips)
    traits = hcat(trait1, trait2)

    alpha1 = 0.4
    alpha2 = 2.2
    sigma1 = 0.7
    sigma2 = 1.4
    theta1 = [-0.5, 0.8]
    theta2 = [1.2, -0.3]
    A = Matrix(Diagonal([alpha1, alpha2]))
    Sigma = Matrix(Diagonal([sigma1, sigma2]))
    theta = hcat(theta1, theta2)

    mv = mvoum_loglikelihood(tree, traits, edge_segments, A, Sigma, theta)
    @test mv.success
    @test isfinite(mv.loglik)
    @test mv.root_mean_mode == :stationary_design
    @test mv.root_cov_mode == :fixed
end

@testset "mvOUM fitting and ancestral reconstruction" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(422)))

    A_true = [
        0.9 0.12;
        0.12 1.1;
    ]
    Sigma_true = [
        0.7 0.2;
        0.2 0.5;
    ]
    edge_segments = _test_edge_segments(tree, 2)
    theta_regimes = [
        0.0 0.3;
        0.8 -0.2;
    ]
    traits = simulate_mvoum(tree, edge_segments, A_true, Sigma_true, eachrow(theta_regimes) |> collect; rng = MersenneTwister(227))

    fixed = mvoum_loglikelihood(tree, traits, edge_segments, A_true, Sigma_true, theta_regimes)
    @test fixed.success
    @test fixed.model == :mvOUM
    @test fixed.ntraits == 2
    @test fixed.nregimes == 2
    @test size(fixed.A) == (2, 2, 1)
    @test size(fixed.Sigma) == (2, 2, 1)
    @test size(fixed.theta) == (2, 2)
    @test isfinite(fixed.loglik)

    fit = fit_mvoum(tree, traits, edge_segments; max_iterations = 60, rel_tol = 1e-6)
    @test fit.success
    @test fit.model == :mvOUM
    @test fit.ntraits == 2
    @test fit.nregimes == 2
    @test size(fit.theta) == (2, 2)
    @test fit.root_mean_mode == :stationary_design
    @test fit.root_cov_mode == :fixed

    asr = estim_node(tree, traits, fit; edge_segments = edge_segments)
    @test asr.success
    @test asr.model == :mvOUM
    @test length(asr.node_ids) == tree.ntips - 1
    @test size(asr.estimates) == (tree.ntips - 1, 2)
    @test size(asr.node_covariances) == (tree.ntips - 1, 2, 2)
end

@testset "mvOUM DataFrame and SimmapSample metadata" begin
    ou_simtree = simulate_yule_simtree(14; tree_height = 1.0, rng = MersenneTwister(2241))
    tree = serialize_tree(ou_simtree)
    edge_segments = _test_edge_segments(tree, 2)
    A = [0.8 0.1; 0.1 0.9]
    Sigma = [0.5 0.05; 0.05 0.4]
    theta = [[0.0, 0.2], [0.7, -0.1]]
    traits = simulate_mvoum(tree, edge_segments, A, Sigma, theta; rng = MersenneTwister(2242))
    df = DataFrame(
        species = tree.tip_labels,
        plate = fill("simple", tree.ntips),
        lnH = traits[:, 1],
        lnVD = traits[:, 2],
    )
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        state_labels = ["simple", "scalariform"],
        edge_segments = edge_segments,
    )

    fit = fit_mvoum(tree, df, simmap; max_iterations = 30, rel_tol = 1e-5)
    @test fit.success
    @test fit.trait_names == ["lnH", "lnVD"]
    @test fit.regime_names == ["simple", "scalariform"]

    matrix_fit = fit_mvoum(tree, traits, simmap; trait_names = ["lnH", "lnVD"], max_iterations = 30, rel_tol = 1e-5)
    @test matrix_fit.success
    @test matrix_fit.trait_names == ["lnH", "lnVD"]
    @test matrix_fit.regime_names == ["simple", "scalariform"]

    shown = sprint(show, MIME("text/plain"), fit)
    @test occursin("lnH", shown)
    @test occursin("lnVD", shown)
    @test occursin("simple", shown)
    @test occursin("theta by regime", shown)
    @test occursin("alpha (A)", shown)
    @test occursin("sigma", shown)
end

@testset "mvOUMV/OUMA/OUMVA fixed likelihood degeneracies" begin
    tree = serialize_tree(simulate_yule_simtree(16; tree_height = 1.0, rng = MersenneTwister(425)))
    edge_segments = _test_edge_segments(tree, 3)
    rng = MersenneTwister(427)
    traits = randn(rng, tree.ntips, 2)

    A = [
        0.9 0.12;
        0.12 1.1;
    ]
    Sigma = [
        0.7 0.2;
        0.2 0.5;
    ]
    theta = [
        -0.2 0.4;
         0.5 0.1;
         1.0 -0.3;
    ]
    A_regimes = repeat(reshape(A, 2, 2, 1), 1, 1, 3)
    Sigma_regimes = repeat(reshape(Sigma, 2, 2, 1), 1, 1, 3)

    oum = mvoum_loglikelihood(tree, traits, edge_segments, A, Sigma, theta)
    oumv = mvoumv_loglikelihood(tree, traits, edge_segments, A, Sigma_regimes, theta)
    ouma = mvouma_loglikelihood(tree, traits, edge_segments, A_regimes, Sigma, theta)
    oumva = mvoumva_loglikelihood(tree, traits, edge_segments, A_regimes, Sigma_regimes, theta)

    @test oum.success
    @test oumv.success
    @test ouma.success
    @test oumva.success
    @test isapprox(oumv.loglik, oum.loglik; atol = 1e-8)
    @test isapprox(ouma.loglik, oum.loglik; atol = 1e-8)
    @test isapprox(oumva.loglik, oum.loglik; atol = 1e-8)
    @test isapprox(oumva.loglik, oumv.loglik; atol = 1e-8)
    @test isapprox(oumva.loglik, ouma.loglik; atol = 1e-8)
end

@testset "mvOUMV/OUMA/OUMVA p=1 fixed likelihood matches univariate" begin
    tree_path = joinpath(mktempdir(), "mvou_p1_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    y = [1.0, 1.1, 2.0, 2.1]
    X = reshape(y, :, 1)
    edge_segments = [
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
        [SimmapSegment(state = 1, length = 1.0)],
        [SimmapSegment(state = 2, length = 1.0)],
    ]

    alpha = 0.5
    alpha_regimes = [0.4, 0.8]
    sigma2 = 0.3
    sigma2_regimes = [0.2, 0.4]
    theta_regimes = [1.0, 2.0]
    theta_matrix = reshape(theta_regimes, :, 1)
    A = reshape([alpha], 1, 1)
    A_regimes = reshape(alpha_regimes, 1, 1, :)
    Sigma = reshape([sigma2], 1, 1)
    Sigma_regimes = reshape(sigma2_regimes, 1, 1, :)

    uni_oumv = oumv_loglikelihood(tree, y, edge_segments, alpha, sigma2_regimes, theta_regimes)
    mv_oumv = mvoumv_loglikelihood(tree, X, edge_segments, A, Sigma_regimes, theta_matrix; root_mean_mode = :root_regime_theta)
    @test uni_oumv.success
    @test mv_oumv.success
    @test isapprox(mv_oumv.loglik, uni_oumv.loglik; atol = 1e-8)

    uni_ouma = ouma_loglikelihood(tree, y, edge_segments, alpha_regimes, sigma2, theta_regimes)
    mv_ouma = mvouma_loglikelihood(tree, X, edge_segments, A_regimes, Sigma, theta_matrix; root_mean_mode = :root_regime_theta)
    @test uni_ouma.success
    @test mv_ouma.success
    @test isapprox(mv_ouma.loglik, uni_ouma.loglik; atol = 1e-8)

    uni_oumva = oumva_loglikelihood(tree, y, edge_segments, alpha_regimes, sigma2_regimes, theta_regimes)
    mv_oumva = mvoumva_loglikelihood(tree, X, edge_segments, A_regimes, Sigma_regimes, theta_matrix; root_mean_mode = :root_regime_theta)
    @test uni_oumva.success
    @test mv_oumva.success
    @test isapprox(mv_oumva.loglik, uni_oumva.loglik; atol = 1e-8)
end

function _test_ou_segment_step(x, A, Sigma, theta, len, rng)
    Phi = exp(-A * len)
    p = length(x)
    Sstat = reshape((kron(Matrix{Float64}(I, p, p), A) + kron(A, Matrix{Float64}(I, p, p))) \ vec(Sigma), p, p)
    Q = (Sstat - Phi * Sstat * Phi' + (Sstat - Phi * Sstat * Phi')') / 2
    cholQ = cholesky(Symmetric(Q + 1e-10 * Matrix{Float64}(I, p, p)))
    return theta + Phi * (x - theta) + cholQ.L * randn(rng, p)
end

function _test_simulate_mvou_family(tree, edge_segments, A_regimes, Sigma_regimes, theta_regimes; rng)
    p = size(theta_regimes, 2)
    root_regime = Int(edge_segments[tree.first_child_edge[tree.root]][1].state)
    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= theta_regimes[root_regime, :]
    for node in tree.preorder
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        for edge in first_edge:tree.last_child_edge[node]
            current = Vector{Float64}(@view node_states[node, :])
            for seg in edge_segments[edge]
                r = Int(seg.state)
                current = _test_ou_segment_step(
                    current,
                    A_regimes[:, :, r],
                    Sigma_regimes[:, :, r],
                    Vector{Float64}(@view theta_regimes[r, :]),
                    seg.length,
                    rng,
                )
            end
            node_states[tree.child_of_edge[edge], :] .= current
        end
    end
    X = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        X[i, :] .= node_states[tip, :]
    end
    return X
end

@testset "mvOUMV/OUMA/OUMVA fixed likelihood favors simulation parameters" begin
    tree = serialize_tree(simulate_yule_simtree(80; tree_height = 1.0, rng = MersenneTwister(428)))
    edge_segments = _test_edge_segments(tree, 2)
    theta = [
        -0.6 0.4;
         1.1 -0.3;
    ]

    A_shared = [
        1.4 0.08;
        0.08 0.9;
    ]
    A_alt = [
        0.45 0.02;
        0.02 1.8;
    ]
    A_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    A_regimes[:, :, 1] = A_shared
    A_regimes[:, :, 2] = A_alt
    A_wrong = repeat(reshape([
        0.25 0.0;
        0.0 0.25;
    ], 2, 2, 1), 1, 1, 2)

    Sigma_shared = [
        0.25 0.04;
        0.04 0.18;
    ]
    Sigma_alt = [
        1.2 0.25;
        0.25 0.85;
    ]
    Sigma_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    Sigma_regimes[:, :, 1] = Sigma_shared
    Sigma_regimes[:, :, 2] = Sigma_alt
    Sigma_wrong = repeat(reshape([
        2.0 0.0;
        0.0 0.08;
    ], 2, 2, 1), 1, 1, 2)

    X_oumv = _test_simulate_mvou_family(
        tree,
        edge_segments,
        repeat(reshape(A_shared, 2, 2, 1), 1, 1, 2),
        Sigma_regimes,
        theta;
        rng = MersenneTwister(430),
    )
    oumv_true = mvoumv_loglikelihood(tree, X_oumv, edge_segments, A_shared, Sigma_regimes, theta; root_mean_mode = :root_regime_theta)
    oumv_wrong = mvoumv_loglikelihood(tree, X_oumv, edge_segments, A_shared, Sigma_wrong, theta; root_mean_mode = :root_regime_theta)
    @test oumv_true.success
    @test oumv_wrong.success
    @test oumv_true.loglik > oumv_wrong.loglik

    X_ouma = _test_simulate_mvou_family(
        tree,
        edge_segments,
        A_regimes,
        repeat(reshape(Sigma_shared, 2, 2, 1), 1, 1, 2),
        theta;
        rng = MersenneTwister(431),
    )
    ouma_true = mvouma_loglikelihood(tree, X_ouma, edge_segments, A_regimes, Sigma_shared, theta; root_mean_mode = :root_regime_theta)
    ouma_wrong = mvouma_loglikelihood(tree, X_ouma, edge_segments, A_wrong, Sigma_shared, theta; root_mean_mode = :root_regime_theta)
    @test ouma_true.success
    @test ouma_wrong.success
    @test ouma_true.loglik > ouma_wrong.loglik

    X_oumva = _test_simulate_mvou_family(tree, edge_segments, A_regimes, Sigma_regimes, theta; rng = MersenneTwister(432))
    oumva_true = mvoumva_loglikelihood(tree, X_oumva, edge_segments, A_regimes, Sigma_regimes, theta; root_mean_mode = :root_regime_theta)
    oumva_wrong = mvoumva_loglikelihood(tree, X_oumva, edge_segments, A_wrong, Sigma_wrong, theta; root_mean_mode = :root_regime_theta)
    @test oumva_true.success
    @test oumva_wrong.success
    @test oumva_true.loglik > oumva_wrong.loglik
end

@testset "mvOUMV fitting and ancestral reconstruction" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(423)))

    A_true = [
        0.95 0.10;
        0.10 1.05;
    ]
    Sigma_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    Sigma_regimes[:, :, 1] = [
        0.7 0.18;
        0.18 0.45;
    ]
    Sigma_regimes[:, :, 2] = [
        0.45 0.12;
        0.12 0.75;
    ]
    edge_segments = _test_edge_segments(tree, 2)
    theta_regimes = [
        0.1 0.25;
        0.9 -0.15;
    ]
    traits = simulate_mvoum(tree, edge_segments, A_true, Sigma_regimes[:, :, 1], eachrow(theta_regimes) |> collect; rng = MersenneTwister(229))

    fixed = mvoumv_loglikelihood(tree, traits, edge_segments, A_true, Sigma_regimes, theta_regimes)
    @test fixed.success
    @test fixed.model == :mvOUMV
    @test fixed.ntraits == 2
    @test fixed.nregimes == 2
    @test size(fixed.A) == (2, 2, 1)
    @test size(fixed.Sigma) == (2, 2, 2)
    @test size(fixed.theta) == (2, 2)

    fit = fit_mvoumv(tree, traits, edge_segments; max_iterations = 40, rel_tol = 1e-6)
    @test fit.success
    @test fit.model == :mvOUMV
    @test fit.ntraits == 2
    @test fit.nregimes == 2
    @test size(fit.Sigma) == (2, 2, 2)
    @test size(fit.theta) == (2, 2)
    @test fit.root_mean_mode == :stationary_design
    @test fit.root_cov_mode == :fixed

    asr = estim_node(tree, traits, fit; edge_segments = edge_segments)
    @test asr.success
    @test asr.model == :mvOUMV
    @test length(asr.node_ids) == tree.ntips - 1
    @test size(asr.estimates) == (tree.ntips - 1, 2)
    @test size(asr.node_covariances) == (tree.ntips - 1, 2, 2)
end

@testset "fit_mvoumv profiles theta and dominates nested mvOUM on simulated data" begin
    tree = serialize_tree(simulate_yule_simtree(40; tree_height = 1.0, rng = MersenneTwister(433)))
    edge_segments = _test_edge_segments(tree, 2)
    A = [
        1.1 0.06;
        0.06 0.85;
    ]
    Sigma_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    Sigma_regimes[:, :, 1] = [
        0.25 0.04;
        0.04 0.18;
    ]
    Sigma_regimes[:, :, 2] = [
        1.0 0.18;
        0.18 0.65;
    ]
    theta = [
        -0.4 0.35;
         0.9 -0.25;
    ]
    traits = _test_simulate_mvou_family(
        tree,
        edge_segments,
        repeat(reshape(A, 2, 2, 1), 1, 1, 2),
        Sigma_regimes,
        theta;
        rng = MersenneTwister(435),
    )

    oum_fit = fit_mvoum(tree, traits, edge_segments; max_iterations = 80, rel_tol = 1e-6)
    oumv_fit = fit_mvoumv(tree, traits, edge_segments; max_iterations = 220, rel_tol = 1e-6)
    @test oum_fit.success
    @test oumv_fit.success
    @test oumv_fit.converged
    @test size(oumv_fit.theta) == (2, 2)
    @test size(oumv_fit.Sigma) == (2, 2, 2)
    @test oumv_fit.loglik >= oum_fit.loglik - 1e-6
    @test all(r -> minimum(eigvals(Symmetric(oumv_fit.Sigma[:, :, r]))) > 0.0, 1:oumv_fit.nregimes)
end

@testset "mvOUMA and mvOUMVA fitting and ancestral reconstruction" begin
    tree = serialize_tree(simulate_yule_simtree(18; tree_height = 1.0, rng = MersenneTwister(424)))

    A_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    A_regimes[:, :, 1] = [
        0.85 0.08;
        0.08 1.05;
    ]
    A_regimes[:, :, 2] = [
        1.15 0.05;
        0.05 0.95;
    ]
    Sigma_true = [
        0.65 0.18;
        0.18 0.55;
    ]
    Sigma_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    Sigma_regimes[:, :, 1] = [
        0.62 0.16;
        0.16 0.52;
    ]
    Sigma_regimes[:, :, 2] = [
        0.48 0.10;
        0.10 0.70;
    ]
    edge_segments = _test_edge_segments(tree, 2)
    theta_regimes = [
        0.05 0.20;
        0.85 -0.10;
    ]
    traits = simulate_mvoum(tree, edge_segments, A_regimes[:, :, 1], Sigma_true, eachrow(theta_regimes) |> collect; rng = MersenneTwister(231))

    fixed_ouma = mvouma_loglikelihood(tree, traits, edge_segments, A_regimes, Sigma_true, theta_regimes)
    @test fixed_ouma.success
    @test fixed_ouma.model == :mvOUMA
    @test size(fixed_ouma.A) == (2, 2, 2)
    @test size(fixed_ouma.Sigma) == (2, 2, 1)

    fit_ouma = fit_mvouma(tree, traits, edge_segments; max_iterations = 40, rel_tol = 1e-6)
    @test fit_ouma.success
    @test fit_ouma.model == :mvOUMA
    @test size(fit_ouma.A) == (2, 2, 2)
    @test size(fit_ouma.Sigma) == (2, 2, 1)

    asr_ouma = estim_node(tree, traits, fit_ouma; edge_segments = edge_segments)
    @test asr_ouma.success
    @test asr_ouma.model == :mvOUMA
    @test size(asr_ouma.estimates) == (tree.ntips - 1, 2)

    fixed_oumva = mvoumva_loglikelihood(tree, traits, edge_segments, A_regimes, Sigma_regimes, theta_regimes)
    @test fixed_oumva.success
    @test fixed_oumva.model == :mvOUMVA
    @test size(fixed_oumva.A) == (2, 2, 2)
    @test size(fixed_oumva.Sigma) == (2, 2, 2)

    fit_oumva = fit_mvoumva(tree, traits, edge_segments; max_iterations = 40, rel_tol = 1e-6)
    @test fit_oumva.success
    @test fit_oumva.model == :mvOUMVA
    @test size(fit_oumva.A) == (2, 2, 2)
    @test size(fit_oumva.Sigma) == (2, 2, 2)
    @test fit_oumva.root_mean_mode == :stationary_design
    @test fit_oumva.root_cov_mode == :fixed

    asr_oumva = estim_node(tree, traits, fit_oumva; edge_segments = edge_segments)
    @test asr_oumva.success
    @test asr_oumva.model == :mvOUMVA
    @test size(asr_oumva.estimates) == (tree.ntips - 1, 2)
    @test size(asr_oumva.node_covariances) == (tree.ntips - 1, 2, 2)
end

@testset "mvOUM tolerates tiny simmap residue on zero-length internal edge" begin
    tree_path = joinpath(mktempdir(), "toy_mvoum_zero_internal.tre")
    write(tree_path, "((A:1,B:1):1,(C:2,D:2):0);")
    tree = serialize_tree(read_tree(tree_path))
    traits = [
        1.0 0.2;
        1.1 0.3;
        2.0 0.9;
        2.1 1.1;
    ]

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

    fit = fit_mvoum(tree, traits, edge_segments; max_iterations = 20, rel_tol = 1e-6)
    @test fit.success
    @test isfinite(fit.loglik)
end

@testset "mvOU1 schur supports asymmetric A for p=2" begin
    tree = serialize_tree(simulate_yule_simtree(24; tree_height = 1.0, rng = MersenneTwister(6101)))
    A_true = [
        1.1 0.8;
        -0.2 0.6;
    ]
    Sigma_true = [
        0.7 0.15;
        0.15 0.5;
    ]
    theta_true = [0.3, -0.2]
    traits = simulate_mvou1(tree, A_true, Sigma_true, theta_true; rng = MersenneTwister(6102))

    fixed = mvou1_loglikelihood(tree, traits, A_true, Sigma_true, theta_true; A_decomp = :schur)
    @test fixed.success
    @test fixed.A_decomp == :schur
    @test fixed.nparams == 4 + 3 + 2
    @test !isapprox(fixed.A[1, 2, 1], fixed.A[2, 1, 1]; atol = 1e-10)
    @test isfinite(fixed.loglik)

    fit = fit_mvou1(tree, traits; A_decomp = :schur, max_iterations = 40, rel_tol = 1e-6)
    @test fit.success
    @test fit.A_decomp == :schur
    @test isfinite(fit.loglik)
end

@testset "mvOUM schur supports asymmetric shared A for p=2" begin
    tree = serialize_tree(simulate_yule_simtree(24; tree_height = 1.0, rng = MersenneTwister(6201)))
    edge_segments = _test_edge_segments(tree, 2)
    A_true = [
        0.9 0.7;
        -0.15 0.5;
    ]
    Sigma_true = [
        0.6 0.12;
        0.12 0.45;
    ]
    theta_regimes = [
        -0.3 0.2;
         0.8 -0.1;
    ]
    traits = simulate_mvoum(tree, edge_segments, A_true, Sigma_true, eachrow(theta_regimes) |> collect; rng = MersenneTwister(6202))

    fixed = mvoum_loglikelihood(tree, traits, edge_segments, A_true, Sigma_true, theta_regimes; A_decomp = :schur)
    @test fixed.success
    @test fixed.A_decomp == :schur
    @test fixed.nparams == 4 + 3 + 4
    @test !isapprox(fixed.A[1, 2, 1], fixed.A[2, 1, 1]; atol = 1e-10)
    @test isfinite(fixed.loglik)

    fit = fit_mvoum(tree, traits, edge_segments; A_decomp = :schur, max_iterations = 40, rel_tol = 1e-6)
    @test fit.success
    @test fit.A_decomp == :schur
    @test isfinite(fit.loglik)
end

@testset "schur rejects p!=2" begin
    tree = serialize_tree(simulate_yule_simtree(12; tree_height = 1.0, rng = MersenneTwister(6301)))
    A3 = [
        1.0 0.1 0.0;
        0.2 0.9 0.1;
        0.0 -0.1 0.8;
    ]
    Sigma3 = [
        0.8 0.1 0.0;
        0.1 0.7 0.1;
        0.0 0.1 0.6;
    ]
    theta3 = [0.1, -0.2, 0.3]
    X = simulate_mvou1(tree, A3, Sigma3, theta3; rng = MersenneTwister(6302))
    @test_throws ArgumentError mvou1_loglikelihood(tree, X, A3, Sigma3, theta3; A_decomp = :schur)
    @test_throws ArgumentError fit_mvou1(tree, X; A_decomp = :schur, max_iterations = 5)
end

@testset "mvOUMV/OUMA/OUMVA schur fixed likelihood supports asymmetric A" begin
    tree = serialize_tree(simulate_yule_simtree(20; tree_height = 1.0, rng = MersenneTwister(6401)))
    edge_segments = _test_edge_segments(tree, 2)
    A_shared = [
        0.95 0.65;
        -0.12 0.55;
    ]
    A_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    A_regimes[:, :, 1] = [
        0.9 0.6;
        -0.1 0.5;
    ]
    A_regimes[:, :, 2] = [
        1.1 -0.25;
        0.3 0.7;
    ]
    Sigma_regimes = Array{Float64, 3}(undef, 2, 2, 2)
    Sigma_regimes[:, :, 1] = [0.6 0.1; 0.1 0.4]
    Sigma_regimes[:, :, 2] = [0.7 0.12; 0.12 0.5]
    Sigma_shared = [0.55 0.08; 0.08 0.45]
    theta_regimes = [
        -0.2 0.3;
         0.9 -0.1;
    ]
    traits = simulate_mvoum(tree, edge_segments, A_shared, Sigma_shared, eachrow(theta_regimes) |> collect; rng = MersenneTwister(6402))

    oumv = mvoumv_loglikelihood(tree, traits, edge_segments, A_shared, Sigma_regimes, theta_regimes; A_decomp = :schur)
    @test oumv.success
    @test oumv.A_decomp == :schur
    @test isfinite(oumv.loglik)

    ouma = mvouma_loglikelihood(tree, traits, edge_segments, A_regimes, Sigma_shared, theta_regimes; A_decomp = :schur)
    @test ouma.success
    @test ouma.A_decomp == :schur
    @test isfinite(ouma.loglik)

    oumva = mvoumva_loglikelihood(tree, traits, edge_segments, A_regimes, Sigma_regimes, theta_regimes; A_decomp = :schur)
    @test oumva.success
    @test oumva.A_decomp == :schur
    @test isfinite(oumva.loglik)
end







