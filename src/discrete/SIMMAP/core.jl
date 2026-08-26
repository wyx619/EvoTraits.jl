"""
    simmap_endpoints(tree, tip_priors, Q; kwargs...)

Phase-1 simmap kernel: sample the root state and branch endpoint states using
the Mk pruning cache. This is the high-performance precursor to full within-branch
history sampling and already produces the branch-level regime endpoints needed
for later stochastic mapping steps.
"""
function endpoints(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    kwargs...,
)
    return sample_mk_endpoints(tree, tip_priors, Q; kwargs...)
end

"""
    SimmapSampler

Reusable sampler state for repeated SIMMAP draws from the same tree, tip priors,
and transition matrix. It caches the Mk pruning cache and eigendecomposition so
large `nsim` workflows do not rebuild them for every replicate.
"""
Base.@kwdef struct SimmapSampler
    tree::CompactTree
    cache::MkPruningCache
    transition_matrix::Matrix{Float64}
    evals::Vector{ComplexF64}
    V::Matrix{ComplexF64}
    Vinv::Matrix{ComplexF64}
    root_prior::Symbol = :likelihoods
    root_prior_probs::Union{Nothing, Vector{Float64}} = nothing
    nparams::Int = 0
    state_labels::Vector{String} = String[]
end

function _simmap_sample_from_endpoints(
    tree::CompactTree,
    Q::AbstractMatrix{<:Real},
    endpoints::MkEndpointSample;
    state_labels::AbstractVector = String[],
    branch_lengths::AbstractVector{<:Real} = tree.edge_length,
    rng::AbstractRNG = Random.default_rng(),
    max_attempt::Integer = 100000,
)
    endpoints.success || return SimmapSample(success = false, nstates = endpoints.nstates, state_labels = string.(state_labels), loglik = endpoints.loglik)
    nstates = endpoints.nstates
    labels = isempty(state_labels) ? string.(collect(1:nstates)) : string.(state_labels)
    edge_segments = [Vector{SimmapSegment}() for _ in 1:tree.nedges]
    mapped_edge = zeros(Float64, tree.nedges, nstates)

    for edge in 1:tree.nedges
        segments = _simulate_branch_subst_history_rejection(
            rng,
            Q,
            endpoints.edge_start_states[edge],
            endpoints.edge_end_states[edge],
            Float64(branch_lengths[edge]),
            max_attempt,
        )
        edge_segments[edge] = segments
        for seg in segments
            mapped_edge[edge, Int(seg.state)] += seg.length
        end
    end

    return SimmapSample(
        success = true,
        nstates = nstates,
        root_state = endpoints.root_state,
        state_labels = labels,
        node_states = endpoints.node_states,
        edge_start_states = endpoints.edge_start_states,
        edge_end_states = endpoints.edge_end_states,
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
        loglik = endpoints.loglik,
    )
end

function remap_simmap_states(sample::SimmapSample, state_map::AbstractVector{<:Integer}, target_labels::AbstractVector)
    sample.success || return SimmapSample(success = false, nstates = length(target_labels), state_labels = string.(target_labels), loglik = sample.loglik)
    ntarget = length(target_labels)
    edge_segments = Vector{Vector{SimmapSegment}}(undef, length(sample.edge_segments))
    mapped_edge = zeros(Float64, size(sample.mapped_edge, 1), ntarget)

    for edge in eachindex(sample.edge_segments)
        collapsed = SimmapSegment[]
        for seg in sample.edge_segments[edge]
            mapped_state = Int32(state_map[Int(seg.state)])
            if !isempty(collapsed) && collapsed[end].state == mapped_state
                collapsed[end] = SimmapSegment(state = mapped_state, length = collapsed[end].length + seg.length)
            else
                push!(collapsed, SimmapSegment(state = mapped_state, length = seg.length))
            end
        end
        edge_segments[edge] = collapsed
        for seg in collapsed
            mapped_edge[edge, Int(seg.state)] += seg.length
        end
    end

    node_states = Int32[state_map[Int(s)] for s in sample.node_states]
    edge_start_states = Int32[state_map[Int(s)] for s in sample.edge_start_states]
    edge_end_states = Int32[state_map[Int(s)] for s in sample.edge_end_states]

    return SimmapSample(
        success = true,
        nstates = ntarget,
        root_state = Int32(state_map[Int(sample.root_state)]),
        state_labels = string.(target_labels),
        node_states = node_states,
        edge_start_states = edge_start_states,
        edge_end_states = edge_end_states,
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
        loglik = sample.loglik,
    )
end

function sample_conditioned_endpoints(
    tree::CompactTree,
    conditional_likelihoods::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior_probs::AbstractVector{<:Real},
    branch_lengths::AbstractVector{<:Real} = tree.edge_length,
    loglik::Real = NaN,
    rng::AbstractRNG = Random.default_rng(),
    evals = nothing,
    V = nothing,
    Vinv = nothing,
)
    nstates = size(conditional_likelihoods, 2)
    size(conditional_likelihoods, 1) == tree.nnodes || throw(ArgumentError("conditional_likelihoods must have tree.nnodes rows"))
    length(root_prior_probs) == nstates || throw(ArgumentError("root_prior_probs length and conditional_likelihoods width disagree"))
    length(branch_lengths) == tree.nedges || throw(ArgumentError("branch_lengths must have tree.nedges entries"))

    evals_local, V_local, Vinv_local =
        evals === nothing || V === nothing || Vinv === nothing ? _mk_eigen_cache(_validate_rate_matrix(Q)) : (evals, V, Vinv)

    node_states = fill(Int32(0), tree.nnodes)
    edge_start_states = fill(Int32(0), tree.nedges)
    edge_end_states = fill(Int32(0), tree.nedges)
    probs = zeros(Float64, nstates)
    P = zeros(Float64, nstates, nstates)

    root = tree.root
    @views probs .= conditional_likelihoods[root, :]
    @inbounds for s in 1:nstates
        probs[s] *= root_prior_probs[s]
    end
    node_states[root] = _sample_categorical!(rng, probs)

    for node in tree.preorder
        tree.is_tip[node] && continue
        parent_state = Int(node_states[node])
        for edge in Int(tree.first_child_edge[node]):Int(tree.last_child_edge[node])
            child = Int(tree.child_of_edge[edge])
            _fill_transition_matrix!(P, Float64(branch_lengths[edge]), evals_local, V_local, Vinv_local)
            @inbounds for s in 1:nstates
                probs[s] = P[parent_state, s] * conditional_likelihoods[child, s]
            end
            child_state = _sample_categorical!(rng, probs)
            node_states[child] = child_state
            edge_start_states[edge] = Int32(parent_state)
            edge_end_states[edge] = child_state
        end
    end

    return MkEndpointSample(
        success = true,
        nstates = nstates,
        root_state = node_states[root],
        node_states = node_states,
        edge_start_states = edge_start_states,
        edge_end_states = edge_end_states,
        loglik = Float64(loglik),
    )
end

"""
    prepare_simmap_sampler(tree, tip_priors, Q; kwargs...)

Prepare reusable SIMMAP sampling state for repeated stochastic maps from fixed
tip priors and a fixed Mk transition matrix.
"""
function preparesampler(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior::Symbol = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    nparams::Union{Nothing, Integer} = nothing,
    state_labels::AbstractVector = String[],
)
    cache = mk_pruning_cache(tree, tip_priors, Q; root_prior = root_prior, root_prior_probs = root_prior_probs, nparams = nparams)
    Qf = cache.transition_matrix
    evals, V, Vinv = _mk_eigen_cache(Qf)
    labels = isempty(state_labels) ? string.(collect(1:cache.nstates)) : string.(state_labels)
    effective_root_prior = _mk_effective_root_prior(
        tree,
        cache;
        root_prior = root_prior,
        root_prior_probs = root_prior_probs,
    )
    return SimmapSampler(
        tree = tree,
        cache = cache,
        transition_matrix = Qf,
        evals = collect(ComplexF64.(evals)),
        V = V,
        Vinv = Vinv,
        root_prior = root_prior,
        root_prior_probs = effective_root_prior,
        nparams = Int(something(nparams, cache.nparams)),
        state_labels = labels,
    )
end

simmap_endpoints(args...; kwargs...) = endpoints(args...; kwargs...)
prepare_simmap_sampler(args...; kwargs...) = preparesampler(args...; kwargs...)

@inline function _sample_next_state!(rng::AbstractRNG, Q::AbstractMatrix{<:Real}, current_state::Int32)
    nstates = size(Q, 1)
    probs = zeros(Float64, nstates)
    total = 0.0
    @inbounds for s in 1:nstates
        if s != current_state
            rate = Float64(Q[current_state, s])
            probs[s] = max(rate, 0.0)
            total += probs[s]
        end
    end
    total > 0.0 || return current_state
    u = rand(rng) * total
    acc = 0.0
    @inbounds for s in 1:nstates
        acc += probs[s]
        if u <= acc
            return Int32(s)
        end
    end
    return Int32(argmax(probs))
end

@inline function _sample_next_state_by_rate!(rng::AbstractRNG, Q::AbstractMatrix{<:Real}, current_state::Int)
    nstates = size(Q, 1)
    probs = zeros(Float64, nstates)
    total = 0.0
    @inbounds for s in 1:nstates
        s == current_state && continue
        rate = Float64(Q[current_state, s])
        probs[s] = max(rate, 0.0)
        total += probs[s]
    end
    total > 0.0 || return current_state
    u = rand(rng) * total
    acc = 0.0
    @inbounds for s in 1:nstates
        acc += probs[s]
        if u <= acc
            return s
        end
    end
    return argmax(probs)
end

function _shortest_state_path(Q::AbstractMatrix{<:Real}, start::Int, target::Int)
    start == target && return [start]
    n = size(Q, 1)
    parent = fill(0, n)
    queue = [start]
    parent[start] = -1
    head = 1
    while head <= length(queue)
        cur = queue[head]
        head += 1
        for nxt in 1:n
            nxt == cur && continue
            if parent[nxt] == 0 && Q[cur, nxt] > 0.0
                parent[nxt] = cur
                nxt == target && break
                push!(queue, nxt)
            end
        end
        parent[target] != 0 && break
    end
    parent[target] == 0 && return [start, target]
    path = Int[target]
    cur = target
    while parent[cur] != -1
        cur = parent[cur]
        push!(path, cur)
    end
    reverse!(path)
    return path
end

function _simulate_branch_subst_history_rejection(
    rng::AbstractRNG,
    Q::AbstractMatrix{<:Real},
    init::Int32,
    final::Int32,
    total_bl::Float64,
    max_attempt::Integer,
)
    total_bl >= 0.0 || throw(ArgumentError("branch length must be non-negative"))
    if total_bl == 0.0
        return [SimmapSegment(state = init, length = 0.0)]
    end
    d_rates = [-Float64(Q[i, i]) for i in axes(Q, 1)]
    attempt = 0
    while true
        if attempt >= max_attempt
            path = _shortest_state_path(Q, Int(init), Int(final))
            len = total_bl / length(path)
            return [SimmapSegment(state = Int32(s), length = len) for s in path]
        end
        attempt += 1
        current_state = Int(init)
        segments = SimmapSegment[]
        elapsed = 0.0
        while true
            rate = d_rates[current_state]
            waiting = rate == 0.0 ? total_bl : randexp(rng) / rate
            push!(segments, SimmapSegment(state = Int32(current_state), length = waiting))
            elapsed += waiting
            if elapsed < total_bl
                current_state = _sample_next_state_by_rate!(rng, Q, current_state)
            else
                if current_state != Int(final)
                    break
                end
                if length(segments) == 1
                    segments[1] = SimmapSegment(state = segments[1].state, length = total_bl)
                else
                    used = sum(seg.length for seg in @view segments[1:(end - 1)])
                    segments[end] = SimmapSegment(state = segments[end].state, length = total_bl - used)
                end
                return segments
            end
        end
    end
end

@inline function _transition_probability(
    from_state::Int,
    to_state::Int,
    t::Float64,
    evals::AbstractVector{<:Number},
    V::Matrix{ComplexF64},
    Vinv::Matrix{ComplexF64},
    P::Matrix{Float64},
)
    _fill_transition_matrix!(P, t, evals, V, Vinv)
    return P[from_state, to_state]
end

function _sample_conditioned_jump!(
    rng::AbstractRNG,
    Q::AbstractMatrix{<:Real},
    current_state::Int,
    target_state::Int,
    remaining_time::Float64,
    evals::AbstractVector{<:Number},
    V::Matrix{ComplexF64},
    Vinv::Matrix{ComplexF64},
    Pwork::Matrix{Float64},
    gridP::Matrix{Float64},
    weights::Vector{Float64},
    cdf::Vector{Float64},
)
    denom = _transition_probability(current_state, target_state, remaining_time, evals, V, Vinv, Pwork)
    denom > EVOTRAITS_TINY || return nothing

    nojump =
        if current_state == target_state
            exp(Float64(Q[current_state, current_state]) * remaining_time) / denom
        else
            0.0
        end
    nojump = clamp(nojump, 0.0, 1.0)
    rand(rng) <= nojump && return nothing

    nstates = size(Q, 1)
    ngrid = length(cdf)
    step = remaining_time / ngrid
    total = 0.0

    @inbounds for bin in 1:ngrid
        mid = (bin - 0.5) * step
        stay = exp(Float64(Q[current_state, current_state]) * mid)
        rem = remaining_time - mid
        _fill_transition_matrix!(gridP, rem, evals, V, Vinv)
        bin_weight = 0.0
        for next_state in 1:nstates
            if next_state == current_state
                continue
            end
            rate = Float64(Q[current_state, next_state])
            if rate > 0.0
                weight = stay * rate * gridP[next_state, target_state]
                weights[(bin - 1) * nstates + next_state] = weight
                bin_weight += weight
            else
                weights[(bin - 1) * nstates + next_state] = 0.0
            end
        end
        total += bin_weight
        cdf[bin] = total
    end

    total *= step
    total > EVOTRAITS_TINY || return nothing

    u = rand(rng) * total
    prev_mass = 0.0
    chosen_bin = 1
    @inbounds for bin in 1:ngrid
        bin_mass = cdf[bin] * step
        if u <= bin_mass
            chosen_bin = bin
            break
        end
        prev_mass = bin_mass
    end

    local_u = (u - prev_mass) / step
    offset = (chosen_bin - 1) * nstates
    chosen_state = current_state
    @inbounds for next_state in 1:nstates
        if next_state == current_state
            continue
        end
        local_u -= weights[offset + next_state]
        if local_u <= 0.0
            chosen_state = next_state
            break
        end
    end
    chosen_state == current_state && return nothing

    jump_time = ((chosen_bin - 1) + rand(rng)) * step
    jump_time = clamp(jump_time, min(step, remaining_time), remaining_time)
    return (jump_time = jump_time, next_state = Int32(chosen_state))
end

function _simulate_conditioned_branch!(
    rng::AbstractRNG,
    Q::AbstractMatrix{<:Real},
    branch_length::Float64,
    start_state::Int32,
    end_state::Int32;
    quadrature_bins::Integer = 64,
)
    branch_length >= 0.0 || throw(ArgumentError("branch_length must be non-negative"))
    quadrature_bins >= 8 || throw(ArgumentError("quadrature_bins must be at least 8"))
    if branch_length == 0.0
        return start_state == end_state ? [SimmapSegment(state = start_state, length = 0.0)] : SimmapSegment[]
    end

    nstates = size(Q, 1)
    evals, V, Vinv = _mk_eigen_cache(_validate_rate_matrix(Q))
    Pwork = zeros(Float64, nstates, nstates)
    gridP = zeros(Float64, nstates, nstates)
    weights = zeros(Float64, quadrature_bins * nstates)
    cdf = zeros(Float64, quadrature_bins)
    segments = Vector{SimmapSegment}()

    t = 0.0
    current = Int(start_state)
    target = Int(end_state)
    while t < branch_length - 1e-12
        remaining = branch_length - t
        jump = _sample_conditioned_jump!(rng, Q, current, target, remaining, evals, V, Vinv, Pwork, gridP, weights, cdf)
        if jump === nothing
            push!(segments, SimmapSegment(state = Int32(current), length = remaining))
            break
        end
        jump_time = jump.jump_time
        next_state = Int(jump.next_state)
        jump_time > 0.0 && push!(segments, SimmapSegment(state = Int32(current), length = jump_time))
        t += jump_time
        current = next_state
    end

    if isempty(segments)
        return [SimmapSegment(state = end_state, length = branch_length)]
    end
    segments[end].state == end_state || throw(ErrorException("Conditioned branch sampler failed to match endpoint state"))

    merged = Vector{SimmapSegment}()
    for seg in segments
        if !isempty(merged) && merged[end].state == seg.state
            merged[end] = SimmapSegment(state = seg.state, length = merged[end].length + seg.length)
        else
            push!(merged, seg)
        end
    end
    return merged
end

"""
    simmap_sample(tree, tip_priors, Q; kwargs...)

Full phase-2 simmap kernel: sample root and edge endpoint states, then sample
within-branch histories conditioned on those endpoints. The result includes both
ordered per-edge segments and a dense `mapped_edge[edge, state]` summary matrix.
"""
function simmap_sample(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior::Symbol = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    nparams::Union{Nothing, Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    endpoints = sample_mk_endpoints(
        tree,
        tip_priors,
        Q;
        root_prior = root_prior,
        root_prior_probs = root_prior_probs,
        nparams = nparams,
        rng = rng,
    )
    endpoints.success || return SimmapSample(success = false, nstates = endpoints.nstates, loglik = endpoints.loglik)

    nstates = endpoints.nstates
    edge_segments = [Vector{SimmapSegment}() for _ in 1:tree.nedges]
    mapped_edge = zeros(Float64, tree.nedges, nstates)

    for edge in 1:tree.nedges
        segments = _simulate_conditioned_branch!(
            rng,
            Q,
            tree.edge_length[edge],
            endpoints.edge_start_states[edge],
            endpoints.edge_end_states[edge],
        )
        edge_segments[edge] = segments
        for seg in segments
            mapped_edge[edge, seg.state] += seg.length
        end
    end

    return SimmapSample(
        success = true,
        nstates = nstates,
        root_state = endpoints.root_state,
        state_labels = string.(collect(1:nstates)),
        node_states = endpoints.node_states,
        edge_start_states = endpoints.edge_start_states,
        edge_end_states = endpoints.edge_end_states,
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
        loglik = endpoints.loglik,
    )
end

function simmap_sample(sampler::SimmapSampler; rng::AbstractRNG = Random.default_rng())
    endpoints = _sample_mk_endpoints_from_cache(
        sampler.tree,
        sampler.cache,
        sampler.evals,
        sampler.V,
        sampler.Vinv;
        root_prior = sampler.root_prior,
        root_prior_probs = sampler.root_prior_probs,
        rng = rng,
    )
    endpoints.success || return SimmapSample(success = false, nstates = endpoints.nstates, state_labels = sampler.state_labels, loglik = endpoints.loglik)

    nstates = endpoints.nstates
    edge_segments = [Vector{SimmapSegment}() for _ in 1:sampler.tree.nedges]
    mapped_edge = zeros(Float64, sampler.tree.nedges, nstates)

    for edge in 1:sampler.tree.nedges
        segments = _simulate_conditioned_branch!(
            rng,
            sampler.transition_matrix,
            sampler.tree.edge_length[edge],
            endpoints.edge_start_states[edge],
            endpoints.edge_end_states[edge],
        )
        edge_segments[edge] = segments
        for seg in segments
            mapped_edge[edge, seg.state] += seg.length
        end
    end

    return SimmapSample(
        success = true,
        nstates = nstates,
        root_state = endpoints.root_state,
        state_labels = sampler.state_labels,
        node_states = endpoints.node_states,
        edge_start_states = endpoints.edge_start_states,
        edge_end_states = endpoints.edge_end_states,
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
        loglik = endpoints.loglik,
    )
end

function simmap_samples(sampler::SimmapSampler; nsim::Integer = 1, rng::AbstractRNG = Random.default_rng())
    nsim >= 1 || throw(ArgumentError("nsim must be at least 1"))
    samples = Vector{SimmapSample}(undef, nsim)
    for i in 1:nsim
        samples[i] = simmap_sample(sampler; rng = rng)
    end
    return samples
end

