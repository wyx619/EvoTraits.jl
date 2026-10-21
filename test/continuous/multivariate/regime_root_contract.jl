using Test, EvoTraits, LinearAlgebra, Random, TOML, DataFrames

set_engine_blas_threads!(1)

function _regime_root_fixture(p = 2)
    tree = simulate_yule_tree(14; tree_height = 1.0, rng = MersenneTwister(1191))
    segments = [[SimmapSegment(state = 2, length = 0.2t),
        SimmapSegment(state = 1 + e % 2, length = 0.45t),
        SimmapSegment(state = 1 + (e + 1) % 2, length = 0.35t)]
        for (e, t) in enumerate(tree.edge_length)]
    return tree, segments, randn(MersenneTwister(1192), tree.ntips, p)
end

function _regime_root_bundle(model, A, Sigma; p = 2)
    As = model === :mvOUMV ? zeros(0, 0, 0) : cat(A, p == 2 ? [1.1 -0.12; -0.12 0.5] : 1.7A; dims = 3)
    Ss = model === :mvOUMA ? zeros(0, 0, 0) : cat(Sigma, 0.65Sigma; dims = 3)
    return EvoTraits.MVOUParameterBundle(theta = zeros(2p), A = A, Sigma = Sigma,
        A_regimes = As, Sigma_regimes = Ss)
end

function _regime_root_fixed(model, tree, data, segments, bundle, theta; kwargs...)
    if model === :mvOUMV
        return mvoumv_loglikelihood(tree, data, segments, bundle.A, bundle.Sigma_regimes, theta; kwargs...)
    elseif model === :mvOUMA
        return mvouma_loglikelihood(tree, data, segments, bundle.A_regimes, bundle.Sigma, theta; kwargs...)
    end
    return mvoumva_loglikelihood(tree, data, segments, bundle.A_regimes, bundle.Sigma_regimes, theta; kwargs...)
end

# Independent dense propagation is restricted to these tiny test trees.
function _regime_root_dense(tree, data, segments, bundle, mode)
    p = size(data, 2)
    q = 2p
    block(node) = ((Int(node) - 1) * p + 1):(Int(node) * p)
    W = zeros(tree.nnodes * p, q)
    C = zeros(tree.nnodes * p, tree.nnodes * p)
    root = block(tree.root)
    root_columns = (p + 1):2p
    W[root, root_columns] .= Matrix{Float64}(I, p, p)
    for parent in tree.preorder
        tree.is_tip[parent] && continue
        for edge in tree.first_child_edge[parent]:tree.last_child_edge[parent]
            child, ancestor = block(tree.child_of_edge[edge]), block(parent)
            Phi, Q, design = Matrix{Float64}(I, p, p), zeros(p, p), zeros(p, q)
            for seg in segments[edge]
                r = Int(seg.state)
                A = isempty(bundle.A_regimes) ? bundle.A : bundle.A_regimes[:, :, r]
                Sigma = isempty(bundle.Sigma_regimes) ? bundle.Sigma : bundle.Sigma_regimes[:, :, r]
                F = exp(-A * seg.length)
                stationary = reshape((kron(Matrix{Float64}(I, p, p), A) +
                    kron(A, Matrix{Float64}(I, p, p))) \ vec(Sigma), p, p)
                V = stationary - F * stationary * F'
                Q = F * Q * F' + V
                Phi = F * Phi
                design = F * design
                design[:, ((r - 1) * p + 1):(r * p)] .+= Matrix{Float64}(I, p, p) - F
            end
            W[child, :] .= Phi * W[ancestor, :] + design
            C[child, :] .= Phi * C[ancestor, :]
            C[:, child] .= C[child, :]'
            C[child, child] .= Phi * C[ancestor, ancestor] * Phi' + Q
        end
    end
    observed, response = Int[], Float64[]
    for (i, tip) in enumerate(tree.tip_ids), j in 1:p
        isnan(data[i, j]) && continue
        push!(observed, first(block(tip)) + j - 1)
        push!(response, data[i, j])
    end
    factor = cholesky(Symmetric(C[observed, observed]))
    design, y = factor.L \ W[observed, :], factor.L \ response
    coefficients = pinv(design; rtol = sqrt(eps())) * y
    residual = y - design * coefficients
    loglik = -(sum(abs2, residual) + logdet(factor) + length(y) * log(2pi)) / 2
    return (; W, C, observed, response, factor, design, y, coefficients, loglik)
end

function _regime_point_kernel(segments, bundle, from, to, q)
    p = size(bundle.A, 1)
    Phi, Q, D = Matrix{Float64}(I, p, p), zeros(p, p), zeros(p, q)
    elapsed = 0.0
    for segment in segments
        duration = max(0.0, min(to, elapsed + segment.length) - max(from, elapsed))
        elapsed += segment.length
        duration == 0.0 && continue
        r = Int(segment.state)
        A = isempty(bundle.A_regimes) ? bundle.A : bundle.A_regimes[:, :, r]
        Sigma = isempty(bundle.Sigma_regimes) ? bundle.Sigma : bundle.Sigma_regimes[:, :, r]
        exponential = exp(duration * [-A Sigma; zeros(p, p) A'])
        F = exponential[1:p, 1:p]
        V = exponential[1:p, (p + 1):end] * F'
        Q = F * Q * F' + V
        Phi = F * Phi
        D = F * D
        D[:, ((r - 1) * p + 1):(r * p)] .+= Matrix{Float64}(I, p, p) - F
    end
    return (; Phi, Q, D)
end

@testset "Regime OU public root contract" begin
    tree, segments, data = _regime_root_fixture()
    for (uni, mv) in ((:OUMV, :mvOUMV), (:OUMA, :mvOUMA), (:OUMVA, :mvOUMVA))
        A, Sigma = [0.7 0.12; 0.12 0.9], [0.5 -0.08; -0.08 0.8]
        bundle = _regime_root_bundle(mv, A, Sigma)
        unilik = getfield(EvoTraits, Symbol(lowercase(string(uni)), "_loglikelihood"))
        unifit = getfield(EvoTraits, Symbol("fit_", lowercase(string(uni))))
        mvfit = getfield(EvoTraits, Symbol("fit_", lowercase(string(mv))))
        alpha = uni === :OUMV ? 0.7 : [0.7, 0.9]
        sigma = uni === :OUMA ? 0.5 : [0.5, 0.8]
        for mean in (:theta, :stationary_design, :regime, :free_theta0),
            covariance in (:fixed, :random, :free, :nonstationary)
            mean === :regime && covariance === :fixed && continue
            @test_throws ArgumentError unilik(tree, data[:, 1], segments, alpha, sigma, [0.1, 0.5];
                root_mean_mode = mean, root_cov_mode = covariance)
            @test_throws ArgumentError unifit(tree, data[:, 1], segments;
                root_mean_mode = mean, root_cov_mode = covariance)
            @test_throws ArgumentError _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2);
                root_mean_mode = mean, root_cov_mode = covariance)
            @test_throws ArgumentError mvfit(tree, data, segments;
                root_mean_mode = mean, root_cov_mode = covariance)
        end
        @test_throws ArgumentError unilik(tree, data[:, 1], segments, alpha, sigma, [0.1, 0.5]; theta0 = 2.0)
        @test_throws ArgumentError unilik(tree, data[:, 1], segments, alpha, sigma, [0.1, 0.5]; root_mean_mode = :free_theta0, theta0 = NaN)
        @test_throws ArgumentError unilik(tree, data[:, 1], segments, alpha, sigma, [0.1, 0.5]; root_mean_mode = :free_theta0, theta0 = 2.0)
        @test_throws ArgumentError _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2); root_mean_mode = :free_theta0)
        @test_throws ArgumentError _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2); root_mean_mode = :free_theta0, theta0 = [1.0])
        @test_throws ArgumentError _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2); root_mean_mode = :free_theta0, theta0 = [NaN, 0.0])
        @test_throws ArgumentError _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2); root_mean_mode = :free_theta0, theta0 = [1.0, 2.0])
        @test_throws ArgumentError _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2); theta0 = zeros(2))
        default = _regime_root_fixed(mv, tree, data, segments, bundle, zeros(2, 2))
        @test default.root_mean_mode === :regime
        @test default.root_cov_mode === :fixed
    end
end

@testset "Matrix root GLS, fixed likelihood, node and branch ASR" begin
    tree, segments, original = _regime_root_fixture()
    for model in (:mvOUMV, :mvOUMA, :mvOUMVA), mode in (:regime,),
        missing in (false, true), decomposition in (:cholesky, :diagonal, :decomposable_positive)
        data = copy(original)
        if missing
            data[1, 1] = NaN
            data[2, 2] = NaN
            data[3:5, :] .= NaN
        end
        A = decomposition === :diagonal ? [0.7 0.0; 0.0 0.9] :
            decomposition === :cholesky ? [0.7 0.12; 0.12 0.9] : [0.7 0.25; 0.0 0.9]
        Sigma = [0.5 -0.08; -0.08 0.8]
        base = _regime_root_bundle(model, A, Sigma)
        As = isempty(base.A_regimes) ? base.A_regimes : cat(A,
            decomposition === :diagonal ? [1.1 0.0; 0.0 0.5] :
            decomposition === :cholesky ? [1.1 -0.12; -0.12 0.5] : [1.1 0.0; 0.15 0.5]; dims = 3)
        bundle = EvoTraits.MVOUParameterBundle(theta = base.theta, A = A, Sigma = Sigma,
            A_regimes = As, Sigma_regimes = base.Sigma_regimes)
        spec = EvoTraits._mvou_spec_with_root(model; root_mean_mode = mode, A_decomp = decomposition)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
        oracle = _regime_root_dense(tree, data, segments, bundle, mode)
        ws = EvoTraits._mvou_fixed_mean_workspace(tree, 2, size(oracle.W, 2); nregimes = 2)
        prof = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec, ws; diagnostics = true)
        @test prof.success
        coefficients = prof.theta
        @test isapprox(oracle.design * coefficients, oracle.design * oracle.coefficients; atol = 1e-9, rtol = 1e-9)
        @test isapprox(prof.loglik, oracle.loglik; atol = 1e-9, rtol = 1e-10)
        @test prof.mean_diagnostics.rank == rank(oracle.design; rtol = sqrt(eps()))
        theta = permutedims(reshape(prof.theta, 2, 2))
        fit = _regime_root_fixed(model, tree, data, segments, bundle, theta;
            root_mean_mode = mode, theta0 = prof.theta0, A_decomp = decomposition)
        @test fit.success
        @test isapprox(fit.loglik, prof.loglik; atol = 1e-10)
        @test fit.nparams == (decomposition === :diagonal ? 2 : decomposition === :cholesky ? 3 : 4) *
            (model === :mvOUMV ? 1 : 2) + 3 * (model === :mvOUMA ? 1 : 2) + length(coefficients)
        shift = [12.0, -7.0]
        shifted = _regime_root_fixed(model, tree, data .+ shift', segments, bundle, theta .+ shift';
            root_mean_mode = mode,
            A_decomp = decomposition)
        @test isapprox(shifted.loglik, fit.loglik; atol = 1e-9)
        means = oracle.W * coefficients
        residual = oracle.response - means[oracle.observed]
        cross = oracle.C[:, oracle.observed]
        posterior = means + cross * (oracle.factor \ residual)
        covariance = oracle.C - cross * (oracle.factor \ cross')
        asr = estim_node(tree, data, fit; edge_segments = segments)
        @test asr.success
        @test isapprox(asr.all_node_estimates, permutedims(reshape(posterior, 2, :)); atol = 1e-9)
        for node in 1:tree.nnodes
            block = (2node - 1):2node
            @test isapprox(asr.all_node_covariances[node, :, :], covariance[block, block]; atol = 1e-9)
        end
        branch = estim_branch_for_simmap(tree, data, fit; edge_segments = segments)
        @test branch.success
        midpoint_mean_gap, midpoint_cov_gap = 0.0, 0.0
        for i in eachindex(branch.edge_ids)
            edge = branch.edge_ids[i]
            branch.branch_start[i] == 0.0 && @test isapprox(branch.start_estimates[i, :],
                asr.all_node_estimates[tree.parent_of_edge[edge], :]; atol = 1e-9)
            isapprox(branch.branch_end[i], tree.edge_length[edge]; atol = 1e-14) && @test isapprox(branch.end_estimates[i, :],
                asr.all_node_estimates[tree.child_of_edge[edge], :]; atol = 1e-9)
            parent, child = Int(tree.parent_of_edge[edge]), Int(tree.child_of_edge[edge])
            pa, ch = (2parent - 1):2parent, (2child - 1):2child
            time, duration = branch.branch_midpoint[i], tree.edge_length[edge]
            prefix = _regime_point_kernel(segments[edge], bundle, 0.0, time, length(coefficients))
            suffix = _regime_point_kernel(segments[edge], bundle, time, duration, length(coefficients))
            total = _regime_point_kernel(segments[edge], bundle, 0.0, duration, length(coefficients))
            Cpa, Cch = oracle.C[pa, oracle.observed], oracle.C[ch, oracle.observed]
            cross = prefix.Phi * Cpa + prefix.Q * suffix.Phi' * (total.Q \ (Cch - total.Phi * Cpa))
            point_mean = (prefix.Phi * oracle.W[pa, :] + prefix.D) * coefficients
            point_cov = prefix.Phi * oracle.C[pa, pa] * prefix.Phi' + prefix.Q
            expected_mean = point_mean + cross * (oracle.factor \ residual)
            expected_cov = point_cov - cross * (oracle.factor \ cross')
            midpoint_mean_gap = max(midpoint_mean_gap, maximum(abs, expected_mean - branch.midpoint_estimates[i, :]))
            midpoint_cov_gap = max(midpoint_cov_gap, maximum(abs, expected_cov - branch.midpoint_covariances[i, :, :]))
        end
        @test midpoint_mean_gap < 1e-9
        @test midpoint_cov_gap < 1e-9
        # A change must invalidate the cached design before it can be reused.
        changed = EvoTraits.MVOUParameterBundle(A = 1.03bundle.A, Sigma = bundle.Sigma,
            A_regimes = 1.03bundle.A_regimes, Sigma_regimes = bundle.Sigma_regimes)
        changed_reference = _regime_root_dense(tree, data, segments, changed, mode)
        changed_profile = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, changed, precalc, spec, ws)
        @test isapprox(changed_profile.loglik, changed_reference.loglik; atol = 1e-9)
        again = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec, ws)
        @test again.loglik == prof.loglik
        @test again.theta == prof.theta
        @test again.theta0 == prof.theta0
    end
end

@testset "Three correlated traits use the same root contract" begin
    tree, segments, data = _regime_root_fixture(3)
    data[1:2:end, 1] .= NaN
    data[2:2:end, 2] .= NaN
    A = [0.9 0.12 0.03; 0.12 0.7 -0.07; 0.03 -0.07 1.2]
    A2 = [0.6 -0.08 0.11; -0.08 1.1 0.04; 0.11 0.04 0.8]
    Sigma = [0.5 -0.08 0.04; -0.08 0.8 0.15; 0.04 0.15 0.6]
    for model in (:mvOUMV, :mvOUMA, :mvOUMVA), mode in (:regime,)
        bundle = EvoTraits.MVOUParameterBundle(A = A, Sigma = Sigma,
            A_regimes = model === :mvOUMV ? zeros(0, 0, 0) : cat(A, A2; dims = 3),
            Sigma_regimes = model === :mvOUMA ? zeros(0, 0, 0) : cat(Sigma, 0.7Sigma; dims = 3))
        spec = EvoTraits._mvou_spec_with_root(model; root_mean_mode = mode)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
        reference = _regime_root_dense(tree, data, segments, bundle, mode)
        prof = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec; diagnostics = true)
        @test prof.success
        @test isapprox(prof.loglik, reference.loglik; atol = 1e-9)
        @test prof.mean_diagnostics.rank == rank(reference.design; rtol = sqrt(eps()))
        theta = permutedims(reshape(prof.theta, 3, 2))
        fixed = _regime_root_fixed(model, tree, data, segments, bundle, theta; root_mean_mode = mode, theta0 = prof.theta0)
        @test isapprox(fixed.loglik, prof.loglik; atol = 1e-9)
        @test estim_node(tree, data, fixed; edge_segments = segments).success
    end
end

@testset "Weak and saturated attraction preserve rank diagnostics" begin
    tree, segments, data = _regime_root_fixture()
    for model in (:mvOUMV, :mvOUMA, :mvOUMVA), scale in (1e-7, 0.5, 100.0), nmissing in (0, 13)
        traits = copy(data)
        traits[1:nmissing, :] .= NaN
        base = _regime_root_bundle(model, scale * [0.7 0.12; 0.12 0.9], [0.5 -0.08; -0.08 0.8])
        As = isempty(base.A_regimes) ? base.A_regimes : cat(base.A, scale * [1.1 -0.12; -0.12 0.5]; dims = 3)
        bundle = EvoTraits.MVOUParameterBundle(A = base.A, Sigma = base.Sigma,
            A_regimes = As, Sigma_regimes = base.Sigma_regimes)
        spec = EvoTraits._mvou_spec_with_root(model; root_mean_mode = :regime)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
        reference = _regime_root_dense(tree, traits, segments, bundle, :regime)
        prof = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, traits, bundle, precalc, spec; diagnostics = true)
        @test prof.success
        @test isapprox(prof.loglik, reference.loglik; atol = 2e-7, rtol = 2e-8)
        @test prof.mean_diagnostics.rank == rank(reference.design; rtol = sqrt(eps()))
        @test prof.mean_diagnostics.ncoefficients == 4
        if nmissing > 0
            @test prof.mean_diagnostics.rank < 4
            @test !all(prof.mean_diagnostics.identifiable)
        end
    end
end

@testset "One-dimensional matrix roots inherit the OUwie reference" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "..", "univariate", "fixtures", "ouwie_roots.toml"))
    tree = mktemp() do path, io
        write(io, fixture["newick"])
        close(io)
        load_tree(path)
    end
    segments = [[SimmapSegment(state = state, length = duration)
        for (state, duration) in zip(edge["states"], edge["durations"])] for edge in fixture["edge_segments"]]
    for ref in fixture["cases"]
        ref["model"] == "OUM" && continue
        model = Symbol("mv", ref["model"])
        if ref["free_root"]
            @test_throws ArgumentError EvoTraits._mvou_spec_with_root(model; root_mean_mode = :free_theta0)
            continue
        end
        mode = :regime
        data = reshape(copy(fixture["y"]), :, 1)
        ref["missing_tip"] && (data[only(findall(==("C"), tree.tip_labels)), 1] = NaN)
        bundle = EvoTraits.MVOUParameterBundle(A = reshape(ref["alpha"][1:1], 1, 1),
            Sigma = reshape(ref["sigma2"][1:1], 1, 1),
            A_regimes = model === :mvOUMV ? zeros(0, 0, 0) : reshape(ref["alpha"], 1, 1, :),
            Sigma_regimes = model === :mvOUMA ? zeros(0, 0, 0) : reshape(ref["sigma2"], 1, 1, :))
        fixed = _regime_root_fixed(model, tree, data, segments, bundle, reshape(ref["theta"], :, 1);
            root_mean_mode = mode)
        @test fixed.success
        @test isapprox(fixed.loglik, ref["fixed_loglik"]; atol = 1e-9, rtol = 1e-10)
        spec = EvoTraits._mvou_spec_with_root(model; root_mean_mode = mode)
        precalc = EvoTraits._mvou_precalc(tree, spec; edge_segments = segments)
        prof = EvoTraits._mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec; diagnostics = true)
        @test prof.success
        @test isapprox(prof.loglik, ref["profiled_loglik"]; atol = 1e-9, rtol = 1e-10)
        @test prof.mean_diagnostics.rank == ref["mean_rank"]
    end
end

@testset "Root fit diagnostics and dispatch" begin
    tree, segments, data = _regime_root_fixture()
    simmap = SimmapSample(success = true, nstates = 2, root_state = Int32(2),
        state_labels = ["first", "root"], edge_segments = segments)
    for model in (:mvOUMV, :mvOUMA, :mvOUMVA), missing in (false, true)
        traits = copy(data)
        if missing
            traits[1:2:end, 1] .= NaN
            traits[2:2:end, 2] .= NaN
        end
        frame = DataFrame(taxon = tree.tip_labels, first = traits[:, 1], second = traits[:, 2])
        fitter = getfield(EvoTraits, Symbol("fit_", lowercase(string(model))))
        fit = fitter(tree, frame, simmap; max_iterations = 8, optimization = :L_BFGS)
        @test fit.success
        @test fit.root_mean_mode === :regime
        @test fit.mean_diagnostics !== nothing
        @test fit.mean_diagnostics.ncoefficients == 4
        @test fit.theta0 === nothing
        bundle = EvoTraits._mvou_asr_bundle(fit)
        fixed = _regime_root_fixed(model, tree, traits, segments, bundle, fit.theta)
        @test isapprox(fixed.loglik, fit.loglik; atol = 1e-9)
        @test fit.nparams == fixed.nparams
        @test fit.aic == -2fit.loglik + 2fit.nparams
        asr = estim_node(tree, traits, fit; edge_segments = segments)
        @test asr.success
    end
    bundle = _regime_root_bundle(:mvOUMVA, [0.7 0.12; 0.12 0.9], [0.5 -0.08; -0.08 0.8])
    for theta0 in (nothing, [1.0, 2.0])
        unsupported = MVContinuousOUResult(model = :mvOUMVA, success = true, ntraits = 2,
            nregimes = 2, theta = zeros(2, 2), A = bundle.A_regimes, Sigma = bundle.Sigma_regimes,
            root_mean_mode = :free_theta0, root_cov_mode = :fixed, theta0 = theta0)
        @test_throws ArgumentError estim_node(tree, data, unsupported; edge_segments = segments)
        @test_throws ArgumentError estim_branch_for_simmap(tree, data, unsupported; edge_segments = segments)
    end
end
