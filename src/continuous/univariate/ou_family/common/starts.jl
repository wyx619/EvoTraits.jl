@inline function _ou_multistart_count(model::Symbol)
    model === :OUM && return 10
    model === :OUMV && return 10
    model === :OUMA && return 10
    model === :OUMVA && return 20
    return 3
end

function _ou_scale_pattern(n::Integer, kind::Symbol)
    if kind === :flat_low
        return fill(0.5, n)
    elseif kind === :flat_high
        return fill(2.0, n)
    elseif kind === :ascending
        return collect(range(0.6, 1.8; length = n))
    elseif kind === :descending
        return collect(range(1.8, 0.6; length = n))
    elseif kind === :alternating_low
        return [isodd(i) ? 0.7 : 1.4 for i in 1:n]
    elseif kind === :alternating_high
        return [isodd(i) ? 1.4 : 0.7 for i in 1:n]
    elseif kind === :mild_ascending
        return collect(range(0.8, 1.3; length = n))
    elseif kind === :mild_descending
        return collect(range(1.3, 0.8; length = n))
    end
    return fill(1.0, n)
end

function _ou_terminal_regime_means(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache::OUMEdgeSegmentCache,
    nregimes::Integer,
)
    sums = zeros(Float64, nregimes)
    counts = zeros(Int, nregimes)
    edge_of_node = zeros(Int, tree.nnodes)
    @inbounds for edge in 1:tree.nedges
        edge_of_node[Int(tree.child_of_edge[edge])] = edge
    end
    @inbounds for (i, tip0) in enumerate(tree.tip_ids)
        y = Float64(trait[i])
        isnan(y) && continue
        tip = Int(tip0)
        edge = edge_of_node[tip]
        last_seg = Int(cache.edge_last_segment[edge])
        regime = Int(cache.segment_states[last_seg])
        sums[regime] += y
        counts[regime] += 1
    end
    observed = filter(!isnan, Float64.(trait))
    fallback = mean(observed)
    means = Vector{Float64}(undef, nregimes)
    @inbounds for r in 1:nregimes
        means[r] = counts[r] > 0 ? sums[r] / counts[r] : fallback
    end
    return means
end

function _ou_candidate_with_theta(
    init::AbstractVector{<:Real},
    nalphas::Integer,
    nsigmas::Integer,
    theta::AbstractVector{<:Real},
)
    cand = Vector{Float64}(init)
    idx = Int(nalphas + nsigmas)
    @inbounds for i in eachindex(theta)
        cand[idx + i] = Float64(theta[i])
    end
    return cand
end

function _ou_multistart_candidates(
    spec::OUSpec,
    init::AbstractVector{<:Real},
    nregimes::Integer,
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache,
)
    nalphas = spec.alpha_mode === :shared ? 1 : nregimes
    nsigmas = spec.sigma_mode === :shared ? 1 : nregimes
    candidates = Vector{Float64}[Vector{Float64}(init)]

    patterns =
        spec.model === :OUM ? (
            (:flat_low, :flat_low),
            (:flat_high, :flat_high),
            (:flat_low, :flat_high),
            (:flat_high, :flat_low),
        ) :
        spec.model === :OUMV ? (
            (:flat_low, :flat_low),
            (:flat_high, :flat_high),
            (:flat_low, :flat_high),
            (:flat_high, :flat_low),
            (:base, :ascending),
            (:base, :descending),
            (:base, :alternating_low),
            (:base, :alternating_high),
            (:base, :mild_ascending),
        ) :
        spec.model === :OUMA ? (
            (:flat_low, :flat_low),
            (:flat_high, :flat_high),
            (:flat_low, :flat_high),
            (:flat_high, :flat_low),
            (:ascending, :base),
            (:descending, :base),
            (:alternating_low, :base),
            (:alternating_high, :base),
            (:mild_ascending, :base),
        ) :
        spec.model === :OUMVA ? (
            (:flat_low, :flat_low),
            (:flat_high, :flat_high),
            (:flat_low, :flat_high),
            (:flat_high, :flat_low),
            (:ascending, :base),
            (:descending, :base),
            (:alternating_low, :base),
            (:alternating_high, :base),
            (:mild_ascending, :base),
            (:base, :ascending),
            (:base, :descending),
            (:base, :alternating_low),
            (:base, :alternating_high),
            (:base, :mild_ascending),
            (:ascending, :descending),
            (:descending, :ascending),
            (:alternating_low, :alternating_high),
            (:alternating_high, :alternating_low),
            (:mild_ascending, :mild_descending),
        ) :
        ((:flat_low, :flat_low), (:flat_high, :flat_high))

    for (alpha_kind, sigma_kind) in patterns
        cand = Vector{Float64}(init)
        alpha_scales = _ou_scale_pattern(nalphas, alpha_kind)
        sigma_scales = _ou_scale_pattern(nsigmas, sigma_kind)
        @inbounds for i in 1:nalphas
            cand[i] = max(cand[i] * alpha_scales[i], 1e-8)
        end
        @inbounds for i in 1:nsigmas
            idx = nalphas + i
            cand[idx] = max(cand[idx] * sigma_scales[i], 1e-8)
        end
        push!(candidates, cand)
    end

    if spec.theta_mode === :by_regime && cache isa OUMEdgeSegmentCache
        regime_theta = _ou_terminal_regime_means(tree, trait, cache, nregimes)
        push!(candidates, _ou_candidate_with_theta(init, nalphas, nsigmas, regime_theta))
        for theta_scale in (0.5, 2.0)
            cand = _ou_candidate_with_theta(init, nalphas, nsigmas, regime_theta)
            @inbounds for i in 1:(nalphas + nsigmas)
                cand[i] = max(cand[i] * theta_scale, 1e-8)
            end
            push!(candidates, cand)
        end
    end

    target = _ou_multistart_count(spec.model)
    if length(candidates) > target
        resize!(candidates, target)
    end
    return candidates
end

function _ou_optimize_from_initial(
    objective,
    init::AbstractVector{<:Real},
    spec::OUSpec,
    nregimes::Integer,
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache;
    max_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::AbstractVector{<:Real},
)
    if spec.model === :OU1
        return _continuous_optimize_objective(
            objective,
            init;
            method = :L_BFGS,
            max_iterations = max_iterations,
            rel_tol = rel_tol,
            lower_bounds = lower_bounds,
        )
    end
    candidates = _ou_multistart_candidates(spec, init, nregimes, tree, trait, cache)
    return _continuous_two_stage_multistart(
        objective,
        candidates;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )
end
