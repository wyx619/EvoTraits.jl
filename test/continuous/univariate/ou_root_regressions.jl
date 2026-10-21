using Test, EvoTraits, LinearAlgebra, Random

set_engine_blas_threads!(1)

function ou_root_fixture()
    path = joinpath(mktempdir(), "root_regression.tre")
    write(path, "(((A:1,B:1):1,C:2):1,(D:1,(E:0.4,F:0.4):0.6):2);")
    tree = load_tree(path)
    segments = [[SimmapSegment(state = 1, length = t)] for t in tree.edge_length]
    for edge in 1:tree.nedges
        child = tree.child_of_edge[edge]
        if tree.is_tip[child] && tree.node_labels[child] in ("B", "D", "F")
            t = tree.edge_length[edge]
            segments[edge] = [SimmapSegment(state = 1, length = t / 4),
                SimmapSegment(state = 2, length = 3t / 4)]
        end
    end
    y = [Dict("A" => 0.2, "B" => 0.7, "C" => 1.8, "D" => 1.1, "E" => -0.3, "F" => 0.8)[t]
        for t in tree.tip_labels]
    return tree, segments, y
end

# Independent dense oracle, deliberately restricted to tiny test trees.
function ou_root_dense(tree, segments, spec, alpha, sigma2, y)
    k = 2
    W, G = zeros(tree.nnodes, k), zeros(tree.nnodes, tree.nnodes)
    root = Int(tree.root)
    if spec.root_mean_mode in (:theta, :regime)
        W[root, 1] = 1.0
    end
    if spec.root_cov_mode === :random
        G[root, root] = sqrt(sigma2[1] / (2alpha[1]))
    end
    for parent in tree.preorder
        tree.is_tip[parent] && continue
        for edge in tree.first_child_edge[parent]:tree.last_child_edge[parent]
            child = Int(tree.child_of_edge[edge])
            a, v = 1.0, 0.0
            B = zeros(k)
            for segment in segments[edge]
                r = Int(segment.state)
                ar = alpha[spec.alpha_mode === :shared ? 1 : r]
                sr = sigma2[spec.sigma_mode === :shared ? 1 : r]
                phi = exp(-ar * segment.length)
                B .*= phi
                B[r] -= expm1(-ar * segment.length)
                a *= phi
                v = phi^2 * v - sr * expm1(-2ar * segment.length) / (2ar)
            end
            W[child, :] = a * W[parent, :] + B
            G[child, :] = a * G[parent, :]
            G[child, child] += sqrt(max(v, 1e-12))
        end
    end
    if spec.root_mean_mode === :stationary
        W[root, 1] = 1.0
        for node in 1:tree.nnodes
            node == root && continue
            W[node, :] ./= sum(W[node, :])
        end
    end
    observed = findall(!isnan, y)
    tips = Int.(tree.tip_ids[observed])
    covariance = G * G'
    V = covariance[tips, tips]
    factor = cholesky(Symmetric(V))
    design = factor.L \ W[tips, :]
    response = factor.L \ y[observed]
    coefficients = pinv(design; rtol = sqrt(eps())) * response
    return (; W, G, covariance, factor, coefficients, design, response, tips, observed)
end

@testset "OU branch interior agrees with dense conditioning" begin
    tree, segments, complete = ou_root_fixture()
    for (model, likelihood, alpha, sigma2) in (
        (:OUM, oum_loglikelihood, [0.5], [0.4]),
        (:OUMV, oumv_loglikelihood, [0.5], [0.4, 0.7]),
        (:OUMA, ouma_loglikelihood, [0.5, 0.9], [0.4]),
        (:OUMVA, oumva_loglikelihood, [0.5, 0.9], [0.4, 0.7]),
    ), mode in (:regime, :stationary_design), missing_tip in (false, true)
        y = copy(complete)
        missing_tip && (y[2] = NaN)
        a_arg = length(alpha) == 1 ? only(alpha) : alpha
        s_arg = length(sigma2) == 1 ? only(sigma2) : sigma2
        spec = EvoTraits.ou_spec(model; root_mean_mode = mode, root_cov_mode = :random)
        # Legacy extensions remain covered through the internal specification;
        # public OUMV/OUMA/OUMVA now enforce the fixed-root contract.
        cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
        bundle = EvoTraits.OUParameterBundle(theta = [0.3, 1.4], alpha = alpha, sigma2 = sigma2)
        profile = EvoTraits._ou_loglikelihood(tree, y, spec, bundle; cache)
        fit = EvoTraits._ou_multiregime_result(spec, bundle, profile, cache.nregimes)
        oracle = ou_root_dense(tree, segments, spec, alpha, sigma2, y)
        coefficients = [0.3, 1.4]
        means = oracle.W * coefficients
        residual = y[oracle.observed] - means[oracle.tips]
        branch = estim_branch_for_simmap(tree, y, fit; edge_segments = segments)
        @test branch.success
        for i in eachindex(branch.edge_ids)
            edge = Int(branch.edge_ids[i])
            parent, child = Int(tree.parent_of_edge[edge]), Int(tree.child_of_edge[edge])
            time = branch.branch_midpoint[i]
            elapsed, pa, pb, pv, sa, tv = 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
            for seg in segments[edge]
                r = Int(seg.state)
                ar = alpha[length(alpha) == 1 ? 1 : r]
                sr = sigma2[length(sigma2) == 1 ? 1 : r]
                duration = clamp(time - elapsed, 0.0, seg.length)
                phi = exp(-ar * duration)
                pa *= phi
                pb = phi * pb - expm1(-ar * duration) * coefficients[r]
                pv = phi^2 * pv - sr * expm1(-2ar * duration) / (2ar)
                sa *= exp(-ar * (seg.length - duration))
                tv = exp(-2ar * seg.length) * tv - sr * expm1(-2ar * seg.length) / (2ar)
                elapsed += seg.length
            end
            cross = pa * oracle.covariance[parent, oracle.tips] +
                (sa * pv / sqrt(tv)) * oracle.G[oracle.tips, child]
            expected_mean = pa * means[parent] + pb + dot(cross, oracle.factor \ residual)
            expected_variance = pa^2 * oracle.covariance[parent, parent] + pv - dot(cross, oracle.factor \ cross)
            @test isapprox(branch.midpoint_estimates[i], expected_mean; atol = 2e-11)
            @test isapprox(branch.midpoint_variances[i], expected_variance; atol = 2e-11)
        end
    end
end

@testset "Stable scalar Gaussian information" begin
    for p in (1e-20, 1e-16, 1e-8, 1.0, 1e8), v in (1e-10, 1.0, 1e8)
        h, a, b = 2p, 0.5, 1.0
        result = EvoTraits._scalar_info_to_parent(p, h, 0.0, a, b, v)
        reference = setprecision(256) do
            P, H, A, B, V = BigFloat.((p, h, a, b, v))
            D = 1 + V * P
            (Float64(A^2 * P / D), Float64(A * (H - B * P) / D),
                Float64(-log1p(V * P) / 2 + (V * H^2 / 2 + B * H - B^2 * P / 2) / D))
        end
        @test result.success
        @test result.precision > 0
        @test isapprox(result.precision, reference[1]; rtol = 2e-15)
        @test isapprox(result.linear, reference[2]; rtol = 2e-15)
        @test isapprox(result.logconst, reference[3]; rtol = 3e-14, atol = 1e-30)
    end
    root = EvoTraits._scalar_root_info_loglik(1e-20, 2e-20, 0.0, 26.0, 0.5, false)
    ref = setprecision(256) do
        p, h, m, v = BigFloat.((1e-20, 2e-20, 26.0, 0.5))
        Float64(-log1p(v * p) / 2 + (v * h^2 / 2 + m * h - m^2 * p / 2) / (1 + v * p))
    end
    @test isapprox(root.loglik, ref; rtol = 2e-15)
    @test EvoTraits._edge_predict_to_child(1.0, Inf, 0.0, 2.0, 0.5) == (mean = 2.0, var = 0.5)
end

@testset "OU profiled GLS versus dense Gaussian" begin
    tree, segments, complete = ou_root_fixture()
    cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
    for model in (:OUM, :OUMV, :OUMA, :OUMVA),
        mean_mode in (:regime, :stationary_design),
        covariance_mode in (:fixed, :random), missing_tip in (false, true),
        alpha_scale in (1e-7, 0.5, 10.0, 1000.0)
        spec = EvoTraits.ou_spec(model; root_mean_mode = mean_mode, root_cov_mode = covariance_mode)
        alpha = spec.alpha_mode === :shared ? [alpha_scale] : [alpha_scale, 1.7alpha_scale]
        sigma2 = spec.sigma_mode === :shared ? [0.4] : [0.4, 0.7]
        y = copy(complete)
        missing_tip && (y[2] = NaN)
        context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
        evaluation = EvoTraits._ou_profile_parameters(context, vcat(alpha, sigma2), EvoTraits.OUMeanWorkspace(context);
            diagnostics = true)
        @test evaluation.profile.success
        oracle = ou_root_dense(tree, segments, spec, alpha, sigma2, y)
        coefficients = evaluation.bundle.theta
        # Near rank loss, the fitted means and likelihood are more informative
        # than large, ill-conditioned individual coefficient differences.
        @test isapprox(oracle.design * coefficients, oracle.design * oracle.coefficients; rtol = 2e-7, atol = 2e-7)
        residual = oracle.response - oracle.design * oracle.coefficients
        expected = -(sum(abs2, residual) + logdet(oracle.factor) + length(oracle.observed) * log(2pi)) / 2
        @test isapprox(evaluation.profile.loglik, expected; rtol = 2e-8, atol = 2e-7)
        @test evaluation.profile.mean_diagnostics.rank == rank(oracle.design; rtol = sqrt(eps()))
    end
end

@testset "Removed independent root rejects fitting and reconstruction" begin
    tree, segments, y = ou_root_fixture()
    for (model, likelihood, alpha, sigma2) in (
        (:OUM, oum_loglikelihood, 0.5, 0.4),
        (:OUMV, oumv_loglikelihood, 0.5, [0.4, 0.7]),
        (:OUMA, ouma_loglikelihood, [0.5, 0.9], 0.4),
        (:OUMVA, oumva_loglikelihood, [0.5, 0.9], [0.4, 0.7]),
    )
        @test_throws ArgumentError likelihood(tree, y, segments, alpha, sigma2, [0.3, 1.4]; root_mean_mode = :free_theta0)
        @test_throws ArgumentError likelihood(tree, y, segments, alpha, sigma2, [0.3, 1.4];
            root_mean_mode = :free_theta0, theta0 = 2.3)
        @test_throws ArgumentError EvoTraits.ou_spec(model; root_mean_mode = :free_theta0)
        fitter = getfield(EvoTraits, Symbol("fit_", lowercase(string(model))))
        @test_throws ArgumentError fitter(tree, y, segments; root_mean_mode = :free_theta0)
        for theta0 in (nothing, 2.3)
            unsupported = ContinuousMultiRegimeResult(model = model, success = true,
                alpha = alpha isa Real ? alpha : NaN,
                alpha_regimes = alpha isa Real ? Float64[] : alpha,
                sigma2 = sigma2 isa Real ? [sigma2] : sigma2, theta_regimes = [0.3, 1.4],
                root_mean_mode = :free_theta0, root_cov_mode = :fixed, theta0 = theta0)
            @test_throws ArgumentError estim_node(tree, y, unsupported; edge_segments = segments)
            @test_throws ArgumentError estim_branch_for_simmap(tree, y, unsupported; edge_segments = segments)
        end
    end
end

@testset "OUM-family fitted means and diagnostics" begin
    tree, segments, y = ou_root_fixture()
    for fitter in (fit_oum, fit_oumv, fit_ouma, fit_oumva)
        fit = fitter(tree, y, segments; max_iterations = 60)
        @test fit.success
        @test fit.theta0 === nothing
        @test fit.mean_diagnostics !== nothing
        @test isapprox(fit.aic, 2fit.nparams - 2fit.loglik; atol = 1e-10)
        @test estim_node(tree, y, fit; edge_segments = segments).success
    end
    cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
    spec = EvoTraits.ou_spec(:OUMVA; root_mean_mode = :regime, root_cov_mode = :random)
    context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
    weak = EvoTraits._ou_profile_parameters(context, [1000.0, 1700.0, 0.4, 0.7], EvoTraits.OUMeanWorkspace(context);
        diagnostics = true)
    @test weak.profile.mean_diagnostics.rank == 2
    @test weak.profile.mean_diagnostics.ncoefficients == 2
    @test all(weak.profile.mean_diagnostics.identifiable)
    @test weak.bundle.theta0 === nothing
end

@testset "Profiled-root covariance and incomplete clades" begin
    tree, segments, complete = ou_root_fixture()
    cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
    alpha, sigma2 = [0.3, 0.7], [0.4, 0.8]
    for mode in (:regime,), covariance_mode in (:free, :nonstationary),
        nmissing in (0, 2, 5)
        y = copy(complete)
        y[1:nmissing] .= NaN
        spec = EvoTraits.ou_spec(:OUMVA; root_mean_mode = mode, root_cov_mode = covariance_mode)
        context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
        evaluation = EvoTraits._ou_profile_parameters(context, vcat(alpha, sigma2), EvoTraits.OUMeanWorkspace(context);
            diagnostics = true)
        @test evaluation.profile.success
        # Profile the unconstrained root as an extra dense GLS coefficient.
        base = EvoTraits.ou_spec(:OUMVA; root_mean_mode = :stationary_design, root_cov_mode = :fixed)
        oracle = ou_root_dense(tree, segments, base, alpha, sigma2, y)
        attenuation = ones(tree.nnodes)
        for node in tree.preorder
            tree.is_tip[node] && continue
            for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
                child = Int(tree.child_of_edge[edge])
                attenuation[child] = attenuation[node] * exp(-sum(alpha[Int(s.state)] * s.length for s in segments[edge]))
            end
        end
        root_design = oracle.factor.L \ attenuation[oracle.tips]
        full_design = hcat(oracle.design, root_design)
        dense_coefficients = pinv(full_design; rtol = sqrt(eps())) * oracle.response
        residual = oracle.response - full_design * dense_coefficients
        expected = -(sum(abs2, residual) + logdet(oracle.factor) + length(oracle.observed) * log(2pi)) / 2
        @test isapprox(evaluation.profile.loglik, expected; atol = 1e-10)
    end
    for covariance_mode in (:fixed, :random)
        y = copy(complete)
        y[1:5] .= NaN
        spec = EvoTraits.ou_spec(:OUMVA; root_mean_mode = :regime, root_cov_mode = covariance_mode)
        context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
        evaluation = EvoTraits._ou_profile_parameters(context, vcat(alpha, sigma2), EvoTraits.OUMeanWorkspace(context);
            diagnostics = true)
        @test evaluation.profile.success
        @test evaluation.profile.mean_diagnostics.rank == 1
    end
end

@testset "Univariate shared OU root combinations" begin
    tree, segments, complete = ou_root_fixture()
    cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
    for mean_mode in (:regime, :stationary), cov_mode in (:fixed, :random),
        missing_tip in (false, true), alpha in (0.001, 0.5)
        y = copy(complete)
        missing_tip && (y[2] = NaN)
        theta, sigma2 = [0.3, 1.4], 0.4
        fit = oum_loglikelihood(tree, y, segments, alpha, sigma2, theta;
            root_mean_mode = mean_mode, root_cov_mode = cov_mode)
        @test fit.success
        @test fit.root_mean_mode === mean_mode
        @test fit.root_cov_mode === cov_mode
        @test fit.nparams == 4
        spec = EvoTraits.ou_spec(:OUM; root_mean_mode = mean_mode, root_cov_mode = cov_mode)
        oracle = ou_root_dense(tree, segments, spec, [alpha], [sigma2], y)
        means = oracle.W * theta
        residual = y[oracle.observed] - means[oracle.tips]
        expected_loglik = -(dot(residual, oracle.factor \ residual) + logdet(oracle.factor) +
            length(residual) * log(2pi)) / 2
        @test isapprox(fit.loglik, expected_loglik; atol = 2e-10)
        expected_mean = means + oracle.covariance[:, oracle.tips] * (oracle.factor \ residual)
        expected_cov = oracle.covariance - oracle.covariance[:, oracle.tips] *
            (oracle.factor \ oracle.covariance[oracle.tips, :])
        node = estim_node(tree, y, fit; edge_segments = segments)
        @test node.success
        @test isapprox(node.all_node_estimates, expected_mean; atol = 2e-10)
        @test isapprox(node.all_node_variances, diag(expected_cov); atol = 2e-10)
        context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
        evaluation = EvoTraits._ou_profile_parameters(context, [alpha, sigma2],
            EvoTraits.OUMeanWorkspace(context); diagnostics = true)
        profiled_residual = oracle.response - oracle.design * oracle.coefficients
        expected_profiled = -(sum(abs2, profiled_residual) + logdet(oracle.factor) +
            length(residual) * log(2pi)) / 2
        @test isapprox(evaluation.profile.loglik, expected_profiled; atol = 2e-10)
    end
    for model in (:OU1, :OUM), mode in (:root_regime_theta, :normalized_design, :free_theta0)
        @test_throws ArgumentError EvoTraits._validate_shared_ou_root(model, mode, :fixed)
    end
    for model in (:OU1, :OUM), mode in (:regime, :stationary)
        @test_throws ArgumentError EvoTraits._validate_shared_ou_root(model, mode, :stationary)
    end
end
