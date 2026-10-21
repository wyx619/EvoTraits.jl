using Test, EvoTraits, LinearAlgebra, TOML

@testset "OUwie 3.02 fixed-root reference" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ouwie_roots.toml"))
    path = joinpath(mktempdir(), "ouwie_roots.tre")
    write(path, fixture["newick"])
    tree = load_tree(path)
    @test tree.tip_labels == fixture["tips"]
    segments = [[SimmapSegment(state = state, length = duration)
        for (state, duration) in zip(edge["states"], edge["durations"])]
        for edge in fixture["edge_segments"]]
    cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
    @test cache.root_regime == 2
    for reference in fixture["cases"]
        model = Symbol(reference["model"])
        if reference["free_root"]
            @test_throws ArgumentError EvoTraits.ou_spec(model; root_mean_mode = :free_theta0)
            continue
        end
        spec = EvoTraits.ou_spec(model; root_mean_mode = :regime,
            root_cov_mode = :fixed)
        alpha, sigma2, theta = reference["alpha"], reference["sigma2"], reference["theta"]
        W = permutedims(hcat(reference["W"]...))
        V = permutedims(hcat(reference["V"]...))
        y = copy(fixture["y"])
        reference["missing_tip"] && (y[only(findall(==("C"), tree.tip_labels))] = NaN)
        observed = findall(!isnan, y)
        context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
        evaluation = EvoTraits._ou_profile_parameters(context, vcat(alpha, sigma2),
            EvoTraits.OUMeanWorkspace(context); diagnostics = true)
        @test evaluation.profile.success
        @test evaluation.profile.mean_diagnostics.rank == reference["mean_rank"]
        @test evaluation.profile.mean_diagnostics.ncoefficients == 3
        coefficients = evaluation.bundle.theta
        @test isapprox(W * coefficients, W * reference["expected_coefficients"]; atol = 1e-9, rtol = 1e-9)
        @test isapprox(evaluation.profile.loglik, reference["profiled_loglik"]; atol = 1e-9, rtol = 1e-10)
        likelihood = getfield(EvoTraits, Symbol(lowercase(reference["model"]) * "_loglikelihood"))
        fit = likelihood(tree, y, segments, length(alpha) == 1 ? only(alpha) : alpha,
            length(sigma2) == 1 ? only(sigma2) : sigma2, theta;
            root_mean_mode = spec.root_mean_mode, root_cov_mode = :fixed)
        @test fit.success
        @test isapprox(fit.loglik, reference["fixed_loglik"]; atol = 1e-9, rtol = 1e-10)
        @test fit.nparams == length(alpha) + length(sigma2) + length(theta)
        @test fit.aic == -2fit.loglik + 2fit.nparams
        @test fit.theta0 === nothing
        @test fit.root_state == theta[2]
        expected = W * theta
        @test isapprox(V, V'; atol = 1e-14)
        dense_loglik = -(dot(y[observed] - expected[observed],
            V[observed, observed] \ (y[observed] - expected[observed])) +
            logdet(Symmetric(V[observed, observed])) + length(observed) * log(2pi)) / 2
        @test isapprox(fit.loglik, dense_loglik; atol = 1e-9, rtol = 1e-10)
    end
end
