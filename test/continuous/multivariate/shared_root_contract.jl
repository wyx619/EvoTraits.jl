using Test, EvoTraits, LinearAlgebra, Random

function _shared_root_fixture(p = 2)
    tree = simulate_yule_tree(12; tree_height = 1.0, rng = MersenneTwister(2911))
    segments = [[SimmapSegment(state = 2, length = 0.3t),
        SimmapSegment(state = 1 + e % 2, length = 0.25t),
        SimmapSegment(state = 1 + (e + 1) % 2, length = 0.45t)]
        for (e, t) in enumerate(tree.edge_length)]
    return tree, segments, randn(MersenneTwister(2912), tree.ntips, p)
end

# Dense propagation and conditioning are confined to these small test trees.
function _shared_root_oracle(tree, data, segments, A, Sigma, mode, covariance_mode; ou1 = false)
    p, k = size(data, 2), ou1 ? 1 : 2
    q = p * k
    block(n) = ((Int(n) - 1) * p + 1):(Int(n) * p)
    W, C = zeros(tree.nnodes * p, q), zeros(tree.nnodes * p, tree.nnodes * p)
    root = block(tree.root)
    columns = (((ou1 ? 0 : 1) * p + 1):(k * p))
    if mode === :regime || ou1
        W[root, columns] .= Matrix{Float64}(I, p, p)
    end
    stationary = reshape((kron(Matrix{Float64}(I, p, p), A) +
        kron(A, Matrix{Float64}(I, p, p))) \ vec(Sigma), p, p)
    covariance_mode === :random && (C[root, root] .= stationary)
    for parent in tree.preorder
        tree.is_tip[parent] && continue
        for edge in tree.first_child_edge[parent]:tree.last_child_edge[parent]
            child, ancestor = block(tree.child_of_edge[edge]), block(parent)
            F, Q, D = Matrix{Float64}(I, p, p), zeros(p, p), zeros(p, q)
            parts = ou1 ? [SimmapSegment(state = 1, length = tree.edge_length[edge])] : segments[edge]
            for seg in parts
                E = exp(seg.length * [-A Sigma; zeros(p, p) A'])
                Phi = E[1:p, 1:p]
                Q = Phi * Q * Phi' + E[1:p, (p + 1):end] * Phi'
                F = Phi * F
                D = Phi * D
                cols = ((Int(seg.state) - 1) * p + 1):(Int(seg.state) * p)
                D[:, cols] .+= Matrix{Float64}(I, p, p) - Phi
            end
            W[child, :] .= F * W[ancestor, :] + D
            C[child, :] .= F * C[ancestor, :]
            C[:, child] .= C[child, :]'
            C[child, child] .= F * C[ancestor, ancestor] * F' + Q
        end
    end
    if mode === :stationary && !ou1
        W[root, ((k - 1) * p + 1):(k * p)] .= A
        for row in axes(W, 1)
            @views W[row, :] ./= sum(W[row, :])
        end
    end
    observed, response = Int[], Float64[]
    for (i, tip) in enumerate(tree.tip_ids), j in 1:p
        isnan(data[i, j]) && continue
        push!(observed, first(block(tip)) + j - 1)
        push!(response, data[i, j])
    end
    factor = cholesky(Symmetric(C[observed, observed]))
    X, y = factor.L \ W[observed, :], factor.L \ response
    coefficients = pinv(X; rtol = sqrt(eps())) * y
    return (; W, C, observed, response, factor, X, y, coefficients, p, k, block)
end

function _shared_root_fixed(tree, data, segments, A, Sigma, mode, covariance_mode, coefficients; ou1 = false, A_decomp = :cholesky)
    p, k = size(data, 2), ou1 ? 1 : 2
    theta = permutedims(reshape(coefficients[1:(k * p)], p, k))
    kwargs = (; root_mean_mode = mode, root_cov_mode = covariance_mode, A_decomp)
    return ou1 ? mvou1_loglikelihood(tree, data, A, Sigma, vec(theta); kwargs...) :
        mvoum_loglikelihood(tree, data, segments, A, Sigma, theta; kwargs...)
end

@testset "Shared OU root likelihood, GLS and Gaussian ASR" begin
    for p in (1, 2, 3), ou1 in (false, true), mode in (:regime, :stationary),
        covariance_mode in (:fixed, :random), missing in (false, true),
        asymmetric in (p == 2 ? (false, true) : (false,))
        tree, segments, data = _shared_root_fixture(p)
        missing && (data[1, 1] = NaN; data[3, :] .= NaN)
        A = Matrix(Diagonal(collect(range(0.45, 1.15; length = p + 1))[1:p])) .+ 0.07
        if asymmetric
            A = [0.45 0.12; 0.0 0.8]
        end
        decomposition = asymmetric ? :decomposable_positive : :cholesky
        Sigma = Matrix{Float64}(I, p, p) .* 0.5 .+ 0.06
        oracle = _shared_root_oracle(tree, data, segments, A, Sigma, mode, covariance_mode; ou1)
        spec = EvoTraits._mvou_public_spec(ou1 ? :mvOU1 : :mvOUM;
            root_mean_mode = mode, root_cov_mode = covariance_mode, A_decomp = decomposition)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = ou1 ? nothing : segments)
        bundle = EvoTraits.MVOUParameterBundle(theta = zeros(p * oracle.k), A = A, Sigma = Sigma)
        prof = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec; diagnostics = true)
        @test prof.success
        coefficients = prof.theta
        @test isapprox(oracle.X * coefficients, oracle.X * oracle.coefficients; atol = 2e-9)
        @test prof.mean_diagnostics.rank == rank(oracle.X; rtol = sqrt(eps()))
        fixed = _shared_root_fixed(tree, data, segments, A, Sigma, mode, covariance_mode, coefficients; ou1, A_decomp = decomposition)
        @test fixed.success
        residual = oracle.response - oracle.W[oracle.observed, :] * coefficients
        ll = -(dot(residual, oracle.factor \ residual) + logdet(oracle.factor) + length(residual) * log(2pi)) / 2
        @test isapprox(fixed.loglik, ll; atol = 2e-9)
        @test isapprox(prof.loglik, ll; atol = 2e-9)
        @test fixed.nparams == (asymmetric ? p^2 : div(p * (p + 1), 2)) +
            div(p * (p + 1), 2) + length(coefficients)
        means = oracle.W * coefficients
        conditional = means + oracle.C[:, oracle.observed] * (oracle.factor \ residual)
        covariance = oracle.C - oracle.C[:, oracle.observed] * (oracle.factor \ oracle.C[oracle.observed, :])
        node = ou1 ? estim_node(tree, data, fixed) : estim_node(tree, data, fixed; edge_segments = segments)
        @test node.success
        @test isapprox(node.all_node_estimates, permutedims(reshape(conditional, p, :)); atol = 2e-8)
        for n in 1:tree.nnodes
            @test isapprox(node.all_node_covariances[n, :, :], covariance[oracle.block(n), oracle.block(n)]; atol = 2e-8)
        end
        if !ou1
            branch = estim_branch_for_simmap(tree, data, fixed; edge_segments = segments)
            @test branch.success
            for i in eachindex(branch.edge_ids)
                e = branch.edge_ids[i]
                parent, child = tree.parent_of_edge[e], tree.child_of_edge[e]
                branch.segment_indices[i] == 1 && @test isapprox(branch.start_estimates[i, :], node.all_node_estimates[parent, :]; atol = 2e-8)
                branch.segment_indices[i] == length(segments[e]) && @test isapprox(branch.end_estimates[i, :], node.all_node_estimates[child, :]; atol = 2e-8)
                t = branch.absolute_midpoint[i]
                prefix = SimmapSegment[]
                remaining = branch.branch_midpoint[i]
                for seg in segments[e]
                    duration = min(remaining, seg.length)
                    duration > 0 && push!(prefix, SimmapSegment(state = seg.state, length = duration))
                    remaining -= duration
                end
                D = mode === :stationary ? zeros(p, length(coefficients)) : copy(oracle.W[oracle.block(parent), :])
                if mode === :stationary
                    # Recover the unnormalized parent's root-to-node design independently.
                    raw = _shared_root_oracle(tree, data, segments, A, Sigma, :stationary_design, covariance_mode)
                    D .= raw.W[oracle.block(parent), :]
                end
                for seg in prefix
                    Phi = exp(-A * seg.length)
                    D = Phi * D
                    D[:, ((Int(seg.state) - 1) * p + 1):(Int(seg.state) * p)] .+= Matrix{Float64}(I, p, p) - Phi
                end
                if mode === :stationary
                    for row in axes(D, 1)
                        @views D[row, :] ./= sum(D[row, :])
                    end
                end
                duration = branch.branch_midpoint[i]
                Phi = exp(-A * duration)
                stationary = reshape((kron(Matrix{Float64}(I, p, p), A) + kron(A, Matrix{Float64}(I, p, p))) \ vec(Sigma), p, p)
                Q = stationary - Phi * stationary * Phi'
                cross = Phi * oracle.C[oracle.block(parent), oracle.observed]
                for (col, idx) in enumerate(oracle.observed)
                    tipnode = div(idx - 1, p) + 1
                    ancestor = tipnode
                    while ancestor != tree.root && ancestor != child
                        ancestor = tree.parent_of_node[ancestor]
                    end
                    if ancestor == child
                        suffix = exp(-A * (tree.dist_from_root[tipnode] - t))
                        trait = mod(idx - 1, p) + 1
                        cross[:, col] .+= (Q * suffix')[:, trait]
                    end
                end
                expected = D * coefficients + cross * (oracle.factor \ residual)
                @test isapprox(branch.midpoint_estimates[i, :], expected; atol = 3e-8)
                point_cov = Phi * oracle.C[oracle.block(parent), oracle.block(parent)] * Phi' + Q
                expected_cov = point_cov - cross * (oracle.factor \ cross')
                @test isapprox(branch.midpoint_covariances[i, :, :], expected_cov; atol = 3e-8)
            end
        end
        if p == 1
            uni = ou1 ? ou1_loglikelihood(tree, data[:, 1], A[1], Sigma[1], coefficients[1];
                root_mean_mode = mode, root_cov_mode = covariance_mode) :
                oum_loglikelihood(tree, data[:, 1], segments, A[1], Sigma[1], coefficients[1:2];
                    root_mean_mode = mode, root_cov_mode = covariance_mode)
            @test isapprox(uni.loglik, fixed.loglik; atol = 2e-9)
            unode = ou1 ? estim_node(tree, data[:, 1], uni) : estim_node(tree, data[:, 1], uni; edge_segments = segments)
            @test isapprox(unode.all_node_estimates, node.all_node_estimates[:, 1]; atol = 2e-8)
            @test isapprox(uni.root_state, unode.all_node_estimates[tree.root]; atol = 2e-8)
            if !ou1
                ubranch = estim_branch_for_simmap(tree, data[:, 1], uni; edge_segments = segments)
                @test isapprox(ubranch.midpoint_estimates, branch.midpoint_estimates[:, 1]; atol = 2e-8)
            end
        end
    end
end

@testset "Shared OU inputs, fitted likelihood replay and saved results" begin
    tree, segments, data = _shared_root_fixture()
    removed = EvoTraits.OUSpec(model = :OU1, root_mean_mode = :free_theta0, root_cov_mode = :fixed)
    @test_throws ArgumentError EvoTraits._ou_fit(tree, data[:, 1], removed)
    @test_throws ArgumentError EvoTraits._ou_fit_with_starts(tree, data[:, 1], removed)
    removed_mv = EvoTraits.MVOUSpec(model = :mvOU1, root_mean_mode = :free_theta0, root_cov_mode = :fixed)
    @test_throws ArgumentError EvoTraits._mvou_precalc(tree, removed_mv)
    removed_precalc = EvoTraits.MVOUPrecalc(branch_lengths = tree.edge_length, root_mean_mode = :free_theta0)
    bundle = EvoTraits.MVOUParameterBundle(A = Matrix{Float64}(I, 2, 2), Sigma = Matrix{Float64}(I, 2, 2))
    @test_throws ArgumentError EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, bundle, removed_precalc, removed_mv)
    for model in (:mvOU1, :mvOUM), mode in (:regime, :stationary), cov in (:fixed, :random)
        ou1 = model === :mvOU1
        fitter = ou1 ? fit_mvou1 : fit_mvoum
        args = ou1 ? (tree, data) : (tree, data, segments)
        fit = fitter(args...; max_iterations = 30, optimization = :L_BFGS,
            root_mean_mode = mode, root_cov_mode = cov)
        @test fit.success
        @test fit.mean_diagnostics !== nothing
        coefficients = vec(fit.theta')
        @test fit.theta0 === nothing
        replay = _shared_root_fixed(tree, data, segments, fit.A[:, :, 1], fit.Sigma[:, :, 1], mode, cov, coefficients; ou1)
        @test isapprox(fit.loglik, replay.loglik; atol = 1e-7)
        @test isapprox(fit.aic, replay.aic; atol = 1e-7)
    end
    for mode in (:regime, :stationary), cov in (:fixed, :random)
        fit = fit_ou1(tree, data[:, 1]; max_iterations = 20, root_mean_mode = mode, root_cov_mode = cov)
        @test fit.success
        replay = ou1_loglikelihood(tree, data[:, 1], fit.alpha, fit.sigma2, fit.theta;
            root_mean_mode = mode, root_cov_mode = cov, theta0 = fit.theta0)
        @test isapprox(fit.loglik, replay.loglik; atol = 1e-7)
        @test fit.nparams == 3
        @test fit.theta0 === nothing
        oum = fit_oum(tree, data[:, 1], segments; max_iterations = 20,
            root_mean_mode = mode, root_cov_mode = cov)
        @test oum.success
        replay_oum = oum_loglikelihood(tree, data[:, 1], segments, oum.alpha, oum.sigma2[1], oum.theta_regimes;
            root_mean_mode = mode, root_cov_mode = cov, theta0 = oum.theta0)
        @test isapprox(oum.loglik, replay_oum.loglik; atol = 1e-7)
        @test isapprox(oum.aic, replay_oum.aic; atol = 1e-7)
        @test oum.mean_diagnostics.ncoefficients == 2
        @test oum.theta0 === nothing
    end
    for model in (:OU1, :OUM, :mvOU1, :mvOUM), mode in (:theta, :stationary_design, :free_theta0, :invalid, :root_regime_theta, :normalized_design), cov in (:fixed, :random, :free, :nonstationary)
        @test_throws ArgumentError EvoTraits._validate_shared_ou_root(model, mode, cov)
    end
    for model in (:OU1, :OUM, :mvOU1, :mvOUM), mode in (:regime, :stationary), cov in (:free, :nonstationary, :stationary)
        @test_throws ArgumentError EvoTraits._validate_shared_ou_root(model, mode, cov)
    end
    for ou1 in (false, true)
        ul = ou1 ? ou1_loglikelihood : oum_loglikelihood
        uf = ou1 ? fit_ou1 : fit_oum
        ml = ou1 ? mvou1_loglikelihood : mvoum_loglikelihood
        mf = ou1 ? fit_mvou1 : fit_mvoum
        ua = ou1 ? (tree, data[:, 1]) : (tree, data[:, 1], segments)
        ma = ou1 ? (tree, data) : (tree, data, segments)
        up = ou1 ? (0.5, 0.4, 0.7) : (0.5, 0.4, [0.7, 0.8])
        mp = ou1 ? (Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2), [0.7, 0.8]) :
            (Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2), [0.7 0.8; 0.2 0.5])
        @test_throws ArgumentError ul(ua..., up...; root_mean_mode = :free_theta0)
        @test_throws ArgumentError ul(ua..., up...; theta0 = 1.0)
        @test_throws ArgumentError ul(ua..., up...; root_mean_mode = :free_theta0, theta0 = NaN)
        @test_throws ArgumentError ml(ma..., mp...; root_mean_mode = :free_theta0)
        @test_throws ArgumentError ml(ma..., mp...; theta0 = ones(2))
        @test_throws ArgumentError ml(ma..., mp...; root_mean_mode = :free_theta0, theta0 = ones(1))
        @test_throws ArgumentError ml(ma..., mp...; root_mean_mode = :free_theta0, theta0 = [NaN, 0.0])
        for cov in (:fixed, :random)
            @test_throws ArgumentError uf(ua...; root_mean_mode = :free_theta0, root_cov_mode = cov)
            @test_throws ArgumentError mf(ma...; root_mean_mode = :free_theta0, root_cov_mode = cov)
            @test_throws ArgumentError ul(ua..., up...; root_mean_mode = :free_theta0, root_cov_mode = cov, theta0 = 1.0)
            @test_throws ArgumentError ml(ma..., mp...; root_mean_mode = :free_theta0, root_cov_mode = cov, theta0 = ones(2))
        end
        @test_throws ArgumentError uf(ua...; root_mean_mode = :stationary_design)
        @test_throws ArgumentError mf(ma...; root_mean_mode = :theta)
        @test_throws ArgumentError uf(ua...; root_cov_mode = :free)
        @test_throws ArgumentError mf(ma...; root_cov_mode = :nonstationary)
    end
    @test_throws DomainError mvoum_loglikelihood(tree, data, segments, [1.0 -1.0; -1.0 2.0],
        Matrix{Float64}(I, 2, 2), [0.2 0.3; 0.4 0.5]; root_mean_mode = :stationary)
    for theta0 in (nothing, 1.0)
        unsupported = ContinuousFitResult(model = :OU1, success = true, alpha = 0.5, sigma2 = 0.4,
            theta = 0.7, root_mean_mode = :free_theta0, root_cov_mode = :fixed, theta0 = theta0)
        @test_throws ArgumentError estim_node(tree, data[:, 1], unsupported)
        unsupported_mv = MVContinuousOUResult(model = :mvOU1, success = true, ntraits = 2, theta = [0.7 0.8],
            A = reshape(Matrix{Float64}(I, 2, 2), 2, 2, 1), Sigma = reshape(Matrix{Float64}(I, 2, 2), 2, 2, 1),
            root_mean_mode = :free_theta0, root_cov_mode = :fixed, theta0 = theta0 === nothing ? nothing : ones(2))
        @test_throws ArgumentError estim_node(tree, data, unsupported_mv)
    end
end
