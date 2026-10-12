using Statistics

# Dense Gaussian conditioning is an independent oracle for small trees only.
function _test_affine_tree_reference(tree, data, root_mean, root_cov, processes)
    p = size(data, 2)
    block(node) = ((Int(node) - 1) * p + 1):(Int(node) * p)
    means = zeros(tree.nnodes * p)
    covariance = zeros(length(means), length(means))
    root = block(tree.root)
    means[root] .= root_mean
    covariance[root, root] .= root_cov
    for parent in tree.preorder
        tree.is_tip[parent] && continue
        for edge in tree.first_child_edge[parent]:tree.last_child_edge[parent]
            child = block(tree.child_of_edge[edge])
            ancestor = block(parent)
            Phi, Q, mu = processes[edge]
            means[child] .= Phi * means[ancestor] + mu
            covariance[child, :] .= Phi * covariance[ancestor, :]
            covariance[:, child] .= covariance[child, :]'
            covariance[child, child] .= Phi * covariance[ancestor, ancestor] * Phi' + Q
        end
    end
    observed = Int[]
    values = Float64[]
    for (i, tip) in enumerate(tree.tip_ids), j in 1:p
        isnan(data[i, j]) && continue
        push!(observed, first(block(tip)) + j - 1)
        push!(values, data[i, j])
    end
    C = cholesky(Symmetric(covariance[observed, observed]))
    residual = values - means[observed]
    loglik = -0.5 * (dot(residual, C \ residual) +
        2sum(log, diag(C.L)) + length(values) * log(2pi))
    posterior_mean = means + covariance[:, observed] * (C \ residual)
    posterior_cov = covariance - covariance[:, observed] * (C \ covariance[observed, :])
    node_covariances = zeros(tree.nnodes, p, p)
    for node in 1:tree.nnodes
        node_covariances[node, :, :] .= posterior_cov[block(node), block(node)]
    end
    return (; loglik, means = permutedims(reshape(posterior_mean, p, tree.nnodes)),
        covariance = node_covariances)
end

function _test_affine_edge(A, Sigma, theta, duration)
    p = size(A, 1)
    exponential = exp(duration * [-A Sigma; zeros(p, p) A'])
    Phi = exponential[1:p, 1:p]
    Q = exponential[1:p, (p + 1):end] * Phi'
    mu = (Matrix{Float64}(I, p, p) - Phi) * theta
    return Phi, (Q + Q') / 2, mu
end

function _test_affine_processes(segments, As, Sigmas, thetas)
    p = size(first(As), 1)
    return map(segments) do edge
        Phi = Matrix{Float64}(I, p, p)
        Q = zeros(p, p)
        mu = zeros(p)
        for segment in edge
            r = Int(segment.state)
            F, V, b = _test_affine_edge(As[r], Sigmas[r], thetas[r], segment.length)
            mu = F * mu + b
            Q = F * Q * F' + V
            Phi = F * Phi
        end
        (Phi, Q, mu)
    end
end

function _test_two_regime_segments(tree; root_regime = 2)
    return [
        [SimmapSegment(state = Int32(tree.parent_of_edge[e] == tree.root ? root_regime : 1 + e % 2),
            length = 0.35tree.edge_length[e]),
         SimmapSegment(state = Int32(1 + (e + 1) % 2), length = 0.65tree.edge_length[e])]
        for e in 1:tree.nedges
    ]
end

function _test_fixed_oum(model, tree, data, segments, As, Sigmas, theta; kwargs...)
    if model === :mvOUM
        return mvoum_loglikelihood(tree, data, segments, As[1], Sigmas[1], theta; kwargs...)
    elseif model === :mvOUMV
        return mvoumv_loglikelihood(tree, data, segments, As[1], cat(Sigmas...; dims = 3), theta; kwargs...)
    elseif model === :mvOUMA
        return mvouma_loglikelihood(tree, data, segments, cat(As...; dims = 3), Sigmas[1], theta; kwargs...)
    end
    return mvoumva_loglikelihood(tree, data, segments,
        cat(As...; dims = 3), cat(Sigmas...; dims = 3), theta; kwargs...)
end

@testset "OUM-family root means and covariance match independent Gaussian conditioning" begin
    tree = simulate_yule_tree(12; tree_height = 1.0, rng = MersenneTwister(731))
    A = [0.9 0.2; 0.2 0.6]
    Sigma = [0.7 -0.12; -0.12 1.1]
    theta = [-0.7 0.4; 1.2 -0.9]
    original = randn(MersenneTwister(732), tree.ntips, 2)
    for nregimes in (1, 2)
        segments = nregimes == 2 ? _test_two_regime_segments(tree) :
            [[SimmapSegment(state = Int32(1), length = tree.edge_length[e])] for e in 1:tree.nedges]
        theta_r = theta[1:nregimes, :]
        root_regime = nregimes
        for model in (:mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA)
            As = [r == 1 || model in (:mvOUM, :mvOUMV) ? A : 1.4A for r in 1:nregimes]
            Sigmas = [r == 1 || model in (:mvOUM, :mvOUMA) ? Sigma : 0.6Sigma for r in 1:nregimes]
            processes = _test_affine_processes(segments, As, Sigmas, collect(eachrow(theta_r)))
            for mean_mode in (:stationary_design, :root_regime_theta), cov_mode in (:fixed, :stationary), missing in (false, true)
                data = copy(original)
                if missing
                    data[1, 1] = NaN
                    data[3, 2] = NaN
                    data[5, :] .= NaN
                end
                root_mean = mean_mode === :stationary_design ? zeros(2) : vec(theta_r[root_regime, :])
                root_cov = cov_mode === :fixed ? zeros(2, 2) :
                    reshape((kron(Matrix{Float64}(I, 2, 2), As[root_regime]) +
                        kron(As[root_regime], Matrix{Float64}(I, 2, 2))) \ vec(Sigmas[root_regime]), 2, 2)
                reference = _test_affine_tree_reference(tree, data, root_mean, root_cov, processes)
                fixed = _test_fixed_oum(model, tree, data, segments, As, Sigmas, theta_r;
                    root_mean_mode = mean_mode, root_cov_mode = cov_mode)
                @test fixed.success
                @test isapprox(fixed.loglik, reference.loglik; atol = 1e-8, rtol = 1e-10)
                spec = EvoTraits._mvou_spec_with_root(model;
                    root_mean_mode = mean_mode, root_cov_mode = cov_mode)
                precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
                direct = EvoTraits._mvou_profile_dispatch(tree, data, EvoTraits._mvou_asr_bundle(fixed), precalc)
                @test isapprox(direct.loglik, reference.loglik; atol = 1e-8, rtol = 1e-10)
                asr = estim_node(tree, data, fixed; edge_segments = segments)
                @test asr.success
                @test isapprox(asr.all_node_estimates, reference.means; atol = 1e-8, rtol = 1e-9)
                @test isapprox(asr.all_node_covariances, reference.covariance; atol = 1e-8, rtol = 1e-9)
            end
        end
    end
end

@testset "Explicit legacy theta design agrees across likelihood and ASR" begin
    tree = simulate_yule_tree(12; tree_height = 1.0, rng = MersenneTwister(733))
    segments = _test_two_regime_segments(tree)
    A, Sigma = [0.9 0.2; 0.2 0.6], [0.7 -0.12; -0.12 1.1]
    theta = [-0.7 0.4; 1.2 -0.9]
    data = randn(MersenneTwister(734), tree.ntips, 2)
    handlers = (EvoTraits._mvoum_tree_pruning_profile, EvoTraits._mvoumv_tree_pruning_profile,
        EvoTraits._mvouma_tree_pruning_profile, EvoTraits._mvoumva_tree_pruning_profile)
    for (model, handler) in zip((:mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA), handlers)
        fixed = _test_fixed_oum(model, tree, data, segments, [A, 1.4A], [Sigma, 0.6Sigma], theta;
            root_mean_mode = :theta)
        spec = EvoTraits._mvou_spec_with_root(model; root_mean_mode = :theta)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
        profiled = handler(tree, data, EvoTraits._mvou_asr_bundle(fixed), precalc)
        @test isapprox(fixed.loglik, profiled.loglik; atol = 1e-8, rtol = 1e-10)
        @test estim_node(tree, data, fixed; edge_segments = segments).success
    end
end

@testset "BM-boundary theta profiling and workspace reuse preserve the likelihood" begin
    tree = simulate_yule_tree(16; tree_height = 1.0, rng = MersenneTwister(739))
    segments = _test_two_regime_segments(tree)
    rotation = [cos(0.7) -sin(0.7); sin(0.7) cos(0.7)]
    A = rotation * Diagonal([1.2e-8, 0.9]) * rotation'
    Sigma = [0.7 -0.12; -0.12 1.1]
    data = randn(MersenneTwister(740), tree.ntips, 2)
    for model in (:mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA), missing in (false, true)
        traits = copy(data)
        missing && (traits[1, 1] = NaN)
        spec = EvoTraits.mvou_spec(model)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
        ws = EvoTraits._mvou_theta_profile_workspace(tree, 2, 4)
        for (delta, sigma_scale) in ((0.0, 1.0), (0.0, 1.2), (1e-13, 1.2))
            As = [A + delta * Matrix{Float64}(I, 2, 2), 1.3A]
            Sigmas = [sigma_scale * Sigma, 0.8Sigma]
            initial = _test_fixed_oum(model, tree, traits, segments, As, Sigmas, zeros(2, 2))
            bundle = EvoTraits._mvou_asr_bundle(initial)
            prof = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, traits, bundle, precalc, spec, ws)
            @test prof.success
            theta = permutedims(reshape(copy(prof.theta), 2, 2))
            fixed = _test_fixed_oum(model, tree, traits, segments, As, Sigmas, theta)
            @test isapprox(prof.loglik, fixed.loglik; atol = 1e-7, rtol = 1e-10)
            uncached = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, traits, bundle, precalc, spec)
            @test isapprox(prof.loglik, uncached.loglik; atol = 1e-7, rtol = 1e-10)
        end
    end
end

@testset "mvOUBM affine ASR matches independent Gaussian conditioning" begin
    tree = simulate_yule_tree(16; tree_height = 1.0, rng = MersenneTwister(735))
    segments = _test_two_regime_segments(tree)
    A = [1.1 0.25; 0.25 0.7]
    B = reshape([0.45, -0.3], 2, 1)
    SigmaY, SigmaX = [0.8 0.1; 0.1 0.5], reshape([0.6], 1, 1)
    mPsi = [-0.7 1.3; 0.5 -0.2]
    vX0 = [0.8]
    H = [A B; zeros(1, 3)]
    Sigma = [SigmaY zeros(2, 1); zeros(1, 2) SigmaX]
    processes = _test_affine_processes(segments, [H, H], [Sigma, Sigma],
        [vcat(mPsi[:, r], 0.0) for r in 1:2])
    root_mean = vcat(mPsi[:, 2] - A \ (B * vX0), vX0)
    original = randn(MersenneTwister(736), tree.ntips, 3)
    for missing in (false, true)
        data = copy(original)
        if missing
            data[1, 1] = NaN
            data[2, 3] = NaN
            data[3, :] .= NaN
        end
        reference = _test_affine_tree_reference(tree, data, root_mean, zeros(3, 3), processes)
        fixed = mvoubm_loglikelihood(tree, data, segments, A, B, SigmaY, SigmaX, mPsi; vX0)
        @test fixed.success
        @test isapprox(fixed.loglik, reference.loglik; atol = 1e-8, rtol = 1e-10)
        asr = estim_node(tree, data, fixed; edge_segments = segments)
        @test asr.success
        @test isapprox(asr.all_node_estimates, reference.means; atol = 1e-8, rtol = 1e-9)
        @test isapprox(asr.all_node_covariances, reference.covariance; atol = 1e-8, rtol = 1e-9)
        @test isapprox(asr.estimates, asr.all_node_estimates[asr.node_ids, :]; atol = 1e-12)
        @test isapprox(asr.tip_estimates, asr.all_node_estimates[tree.tip_ids, :]; atol = 1e-12)
        observed = .!isnan.(data)
        @test isapprox(asr.tip_estimates[observed], data[observed]; atol = 1e-12)
        @test asr.tip_observed_mask == observed
    end
end

@testset "mvOUBM initialization handles sparse and missing observations" begin
    tree = simulate_yule_tree(16; tree_height = 1.0, rng = MersenneTwister(737))
    segments = _test_two_regime_segments(tree)
    spec = EvoTraits.MVOUSpec(model = :mvOUBM)
    precalc = EvoTraits._mvou_build_precalc(tree, spec; edge_segments = segments)
    original = randn(MersenneTwister(738), tree.ntips, 4)
    patterns = [copy(original) for _ in 1:6]
    patterns[2][1:3, 1] .= NaN
    patterns[2][4:6, 3] .= NaN
    patterns[2][7, :] .= NaN
    patterns[3][1:2:end, 1] .= NaN
    patterns[3][2:2:end, 2] .= NaN
    patterns[4][precalc.tip_terminal_regime .== 2, 2] .= NaN
    patterns[5][2:end, 4] .= NaN
    patterns[6] .= 0.5
    for data in patterns
        before = copy(data)
        candidates = EvoTraits._mvoubm_initial_candidates(tree, data, precalc, 2, 2, :cholesky)
        @test !isempty(candidates)
        @test all(c -> all(isfinite, c), candidates)
        @test isequal(data, before)
        params = EvoTraits._mvoubm_unpack_params(first(candidates), 2, 2, 2, :cholesky)
        @test isposdef(Symmetric(params.SigmaY))
        @test isposdef(Symmetric(params.SigmaX))
        @test isapprox(params.vX0, [mean(filter(!isnan, data[:, j])) for j in 3:4]; atol = 1e-12)
        for r in 1:2, j in 1:2
            values = filter(!isnan, data[precalc.tip_terminal_regime .== r, j])
            expected = mean(isempty(values) ? filter(!isnan, data[:, j]) : values)
            @test isapprox(params.mPsi[j, r], expected; atol = 1e-12)
        end
    end
    data = copy(original[:, 1:2])
    data[1:3, 1] .= NaN
    data[4:6, 2] .= NaN
    data[7, :] .= NaN
    before = copy(data)
    fit = fit_mvoubm(tree, data, segments; kY = 1, max_iterations = 500, rel_tol = 1e-6)
    @test fit.success
    @test fit.nobs == count(!isnan, data)
    @test isequal(data, before)
    fixed = mvoubm_loglikelihood(tree, data, segments, fit.A[:, :, 1], fit.B,
        fit.Sigma[1:1, 1:1, 1], fit.Sigma[2:2, 2:2, 1], permutedims(fit.theta); vX0 = fit.vX0)
    @test isapprox(fit.loglik, fixed.loglik; atol = 1e-8, rtol = 1e-10)
    @test estim_node(tree, data, fit; edge_segments = segments).success
    invalid = copy(data)
    invalid[:, 2] .= NaN
    @test_throws ArgumentError fit_mvoubm(tree, invalid, segments)
end
