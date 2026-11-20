using Test, EvoTraits, LinearAlgebra

function oumva_numerics_tree(newick)
    return mktempdir() do dir
        path = joinpath(dir, "numerics.nwk")
        write(path, newick)
        load_tree(path)
    end
end

function oumva_numerics_segments(tree)
    return [begin
        t = tree.edge_length[e]
        label = tree.node_labels[tree.child_of_edge[e]]
        if t == 0 || label == "A"
            [SimmapSegment(state = 1, length = t)]
        elseif label in ("B", "D", "F")
            [SimmapSegment(state = 1, length = t/4),
             SimmapSegment(state = 2, length = 3t/4),
             SimmapSegment(state = 2, length = 0)]
        else
            [SimmapSegment(state = 1, length = t/4),
             SimmapSegment(state = 2, length = t/2),
             SimmapSegment(state = 1, length = t/4)]
        end
    end for e in 1:tree.nedges]
end

# Sum each segment's contribution after decay by all later segments.
function oumva_numerics_transition(segments, alpha, sigma2, time)
    lengths = Float64[]
    remaining = time
    for segment in segments
        push!(lengths, min(segment.length, max(0.0, remaining)))
        remaining -= segment.length
    end
    decay = [alpha[s.state] * t for (s,t) in zip(segments, lengths)]
    W = zeros(length(alpha))
    v = 0.0
    for (i, segment) in enumerate(segments)
        r = segment.state
        phi = exp(-sum(decay[(i+1):end]))
        W[r] -= phi * expm1(-decay[i])
        v -= phi^2 * (sigma2[r] / (2alpha[r])) * expm1(-2decay[i])
    end
    return exp(-sum(decay)), W, v
end

function oumva_numerics_reference(tree, segments, alpha, sigma2, y;
    root_var = 0.0, shared_length = false)
    W = zeros(tree.nnodes, length(alpha))
    G = zeros(tree.nnodes, tree.nnodes)
    W[tree.root, 1] = 1
    G[tree.root, tree.root] = sqrt(root_var)
    for parent in tree.preorder
        tree.is_tip[parent] && continue
        for e in tree.first_child_edge[parent]:tree.last_child_edge[parent]
            child = tree.child_of_edge[e]
            duration = sum(s.length for s in segments[e])
            a, b, v = oumva_numerics_transition(segments[e], alpha, sigma2, duration)
            if shared_length
                a = exp(-alpha[1] * tree.edge_length[e])
                v = -sigma2[1] * expm1(-2alpha[1] * tree.edge_length[e]) / (2alpha[1])
            end
            W[child,:] = a * W[parent,:] + b
            G[child,:] = a * G[parent,:]
            G[child,child] = sqrt(v)
        end
    end
    V = G * G'
    obs = findall(!isnan, y)
    tips = tree.tip_ids[obs]
    F = cholesky(Symmetric(V[tips,tips]))
    Z = F.L \ W[tips,:]
    z = F.L \ y[obs]
    theta = pinv(Z; rtol = sqrt(eps())) * z
    return (; W, G, V, obs, tips, F, theta)
end

function oumva_numerics_ll(ref, y, theta)
    e = ref.F.L \ (y[ref.obs] - ref.W[ref.tips,:] * theta)
    return -(sum(abs2, e) + logdet(ref.F) + length(e) * log(2pi)) / 2
end

function oumva_numerics_profile(tree, segments, alpha, sigma2, y; root_cov_mode = :fixed)
    spec = EvoTraits.ou_spec(:OUMVA; root_cov_mode)
    cache = EvoTraits._prepare_oum_edge_cache(tree, segments)
    context = EvoTraits._ou_prepare_context(tree, y, spec, cache)
    return EvoTraits._ou_profile_parameters(context, vcat(alpha, sigma2),
        EvoTraits.OUMeanWorkspace(context); diagnostics = true)
end

function oumva_numerics_check_asr(tree, segments, alpha, sigma2, theta, y, ref, fit;
    mean_atol, variance_atol)
    node = estim_node(tree, y, fit; edge_segments = segments)
    branch = estim_branch_for_simmap(tree, y, fit; edge_segments = segments)
    @test node.success
    @test branch.success
    node.success && branch.success || return
    mean = ref.W * theta
    residual = y[ref.obs] - mean[ref.tips]
    cross = ref.V[:,ref.tips]
    @test isapprox(node.all_node_estimates, mean + cross * (ref.F \ residual);
        atol = mean_atol, rtol = 0)
    @test isapprox(node.all_node_variances, diag(ref.V - cross * (ref.F \ cross'));
        atol = variance_atol, rtol = 0)
    for (times, estimates, variances) in (
        (branch.branch_start, branch.start_estimates, branch.start_variances),
        (branch.branch_midpoint, branch.midpoint_estimates, branch.midpoint_variances),
        (branch.branch_end, branch.end_estimates, branch.end_variances),
    ), i in eachindex(times)
        e = branch.edge_ids[i]
        parent, child = tree.parent_of_edge[e], tree.child_of_edge[e]
        a, b, v = oumva_numerics_transition(segments[e], alpha, sigma2, times[i])
        _, _, full_v = oumva_numerics_transition(segments[e], alpha, sigma2, tree.edge_length[e])
        decay = sum(alpha[s.state] * s.length for s in segments[e])
        suffix = exp(-decay) / a
        c = a * ref.V[parent,ref.tips]
        if full_v > 0
            c += (suffix * v / sqrt(full_v)) * ref.G[ref.tips,child]
        end
        expected_mean = a * mean[parent] + dot(b, theta) + dot(c, ref.F \ residual)
        expected_var = a^2 * ref.V[parent,parent] + v - dot(c, ref.F \ c)
        @test isapprox(estimates[i], expected_mean; atol = mean_atol, rtol = 0)
        @test isapprox(variances[i], expected_var; atol = variance_atol, rtol = 0)
    end
end

@testset "OUMVA numerical regressions" begin
    @testset "Large additive offsets in fixed and profiled likelihood" begin
        tree = oumva_numerics_tree("(((A:0.5,B:0.5):0.5,C:1):1,((D:0.5,E:0.5):0.5,F:1):1);")
        segments = oumva_numerics_segments(tree)
        alpha, sigma2, theta = [0.125, 2.75], [0.25, 0.75], [0.25, 1.5]
        obs = [-0.5, 0.75, 2.0, 1.25, 0.125, 0.875]
        for nmissing in (0, 1, 5)
            y = copy(obs)
            y[1:nmissing] .= NaN
            ref = oumva_numerics_reference(tree, segments, alpha, sigma2, y)
            fixed = oumva_numerics_ll(ref, y, theta)
            profiled = oumva_numerics_ll(ref, y, ref.theta)
            for shift in (0.0, 1e6, 1e9)
                fit = oumva_LL(tree, y .+ shift, segments, alpha, sigma2, theta .+ shift)
                @test fit.success
                @test fit.root_state == theta[1] + shift
                @test isapprox(fit.loglik, fixed; atol = 2e-10, rtol = 0)
                profile = oumva_numerics_profile(tree, segments, alpha, sigma2, y .+ shift)
                @test profile.profile.success
                @test isapprox(profile.profile.loglik, profiled; atol = 2e-9, rtol = 0)
                @test profile.profile.mean_diagnostics.rank == min(2, length(ref.obs))
                @test isapprox(ref.W[ref.tips,:] * (profile.bundle.theta .- shift),
                    ref.W[ref.tips,:] * ref.theta; atol = 2e-6, rtol = 0)
            end
        end
    end

    @testset "Small positive diffusion in likelihood and ASR" begin
        tree = oumva_numerics_tree("((A:0.000001,B:0.000001):0.000001,(C:0.000001,D:0.000001):0.000001);")
        segments = oumva_numerics_segments(tree)
        alpha, sigma2, theta = [0.1, 2.8], [2e-8, 4e-8], [0.2e-7, -0.5e-7]
        for missing_tip in (false, true)
            y = [-0.7e-7, 1.5e-7, 0.25e-7, 1.75e-7]
            missing_tip && (y[2] = NaN)
            ref = oumva_numerics_reference(tree, segments, alpha, sigma2, y)
            fit = oumva_LL(tree, y, segments, alpha, sigma2, theta)
            @test fit.success
            @test isapprox(fit.loglik, oumva_numerics_ll(ref, y, theta); atol = 2e-10, rtol = 0)
            profile = oumva_numerics_profile(tree, segments, alpha, sigma2, y)
            @test profile.profile.success
            @test isapprox(profile.profile.loglik, oumva_numerics_ll(ref, y, ref.theta);
                atol = 2e-9, rtol = 0)
            oumva_numerics_check_asr(tree, segments, alpha, sigma2, theta, y, ref, fit;
                mean_atol = 1e-18, variance_atol = 1e-23)
        end
    end

    @testset "OUM centering preserves accepted SIMMAP length residuals" begin
        tree = oumva_numerics_tree("((A:1,B:1):1,(C:2,D:2):0);")
        segments = oumva_numerics_segments(tree)
        for e in findall(iszero, tree.edge_length)
            segments[e] = [SimmapSegment(state = 1, length = 1.4901161193847656e-8)]
        end
        alpha, sigma2, theta = 0.5, 0.5, [100001.0, 100003.0]
        y = [100000.5, 100001.0, 100002.0, 100003.0]
        ref = oumva_numerics_reference(tree, segments, fill(alpha, 2), fill(sigma2, 2), y;
            shared_length = true)
        fit = oum_LL(tree, y, segments, alpha, sigma2, theta)
        @test fit.success
        @test isapprox(fit.loglik, oumva_numerics_ll(ref, y, theta); atol = 3e-10, rtol = 0)
    end

    @testset "Deterministic edges and singular observations" begin
        tree = oumva_numerics_tree("((A:0,(B:0,C:0):0):1,((D:0.5,E:0.5):0,F:0.5):0.5);")
        segments = oumva_numerics_segments(tree)
        alpha, sigma2, theta = [0.125, 2.75], [0.25, 0.75], [0.25, 1.5]
        y = [0.75, NaN, NaN, 1.25, 0.125, 0.875]
        ref = oumva_numerics_reference(tree, segments, alpha, sigma2, y)
        fit = oumva_LL(tree, y, segments, alpha, sigma2, theta)
        @test fit.success
        @test isapprox(fit.loglik, oumva_numerics_ll(ref, y, theta); atol = 2e-10, rtol = 0)
        profile = oumva_numerics_profile(tree, segments, alpha, sigma2, y)
        @test profile.profile.success
        @test isapprox(profile.profile.loglik, oumva_numerics_ll(ref, y, ref.theta);
            atol = 2e-10, rtol = 0)
        oumva_numerics_check_asr(tree, segments, alpha, sigma2, theta, y, ref, fit;
            mean_atol = 2e-10, variance_atol = 2e-11)
        for b in (y[1], 2y[1])
            singular = copy(y)
            singular[2] = b
            @test !oumva_LL(tree, singular, segments, alpha, sigma2, theta).success
            @test !oumva_numerics_profile(tree, segments, alpha, sigma2, singular).profile.success
            @test !estim_node(tree, singular, fit; edge_segments = segments).success
            @test !estim_branch_for_simmap(tree, singular, fit; edge_segments = segments).success
        end
        singular = copy(y)
        singular[2] = y[1]
        fitted = fit_oumva(tree, singular, segments; max_iterations = 10)
        @test !fitted.success
        @test !fitted.converged
        @test fitted.loglik == -Inf
        @test fitted.aic == Inf
    end

    @testset "Fixed root and a deterministic observation" begin
        tree = oumva_numerics_tree("(A:1,(B:0.5,C:0.5):0.5);")
        segments = oumva_numerics_segments(tree)
        y = [0.25, 0.75, 1.25]
        alpha, sigma2, theta = [0.125, 2.75], [0.0, 0.75], [0.25, 1.5]
        # A is exactly the root optimum, but the joint density is still singular.
        @test !oumva_LL(tree, y, segments, alpha, sigma2, theta).success
        y[1] = NaN
        ref = oumva_numerics_reference(tree, segments, alpha, sigma2, y)
        fit = oumva_LL(tree, y, segments, alpha, sigma2, theta)
        @test fit.success
        @test isapprox(fit.loglik, oumva_numerics_ll(ref, y, theta); atol = 2e-10, rtol = 0)
        oumva_numerics_check_asr(tree, segments, alpha, sigma2, theta, y, ref, fit;
            mean_atol = 2e-10, variance_atol = 2e-11)
        @test !oumva_LL(tree, y, segments, alpha, fill(nextfloat(0.0), 2), theta).success
    end
end
