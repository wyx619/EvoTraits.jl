"""
    simulate_mvou1(tree, A, Sigma, theta; root_state=nothing, rng=Random.GLOBAL_RNG)

Simulate an OU1 trait matrix with shared `A`, `Sigma`, and `theta`.
"""
function simulate_mvou1(
    tree::CompactTree,
    A::AbstractMatrix,
    Sigma::AbstractMatrix,
    theta::AbstractVector;
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    A_mat = _mvsim_validate_stable_matrix("A", A)
    Sigma_mat = _mvsim_validate_spd_matrix("Sigma", Sigma)
    p = size(A_mat, 1)
    size(Sigma_mat, 1) == p || throw(ArgumentError("A and Sigma dimensions must match"))
    theta_vec = collect(Float64, theta)
    length(theta_vec) == p || throw(ArgumentError("theta length must match matrix size"))
    root_vec = root_state === nothing ? copy(theta_vec) : collect(Float64, root_state)
    length(root_vec) == p || throw(ArgumentError("root_state length must match matrix size"))
    ws = _mvsim_ou_workspace(A_mat, Sigma_mat)

    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= root_vec

    for node in tree.preorder
        parent_state = @view node_states[node, :]
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            t = tree.edge_length[edge]
            child_state = @view node_states[tree.child_of_edge[edge], :]
            _mvsim_ou_step!(child_state, parent_state, A_mat, Sigma_mat, theta_vec, t, rng, ws)
        end
    end

    tip_data = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        @views tip_data[i, :] .= node_states[tip, :]
    end
    return tip_data
end

"""
    simulate_mvoum(tree, edge_segments, A, Sigma, thetas; root_state=nothing, rng=Random.GLOBAL_RNG)

Simulate an OUM trait matrix with regime-specific optima and shared `A` and
`Sigma`.
"""
function simulate_mvoum(
    tree::CompactTree,
    edge_segments::Vector{Vector{SimmapSegment}},
    A::AbstractMatrix,
    Sigma::AbstractMatrix,
    thetas::AbstractVector{<:AbstractVector};
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    _mvsim_validate_edge_segments(tree, edge_segments)
    nregimes = _mvsim_nregimes(edge_segments)
    length(thetas) == nregimes || throw(ArgumentError("thetas must have one optimum vector per regime"))
    A_mat = _mvsim_validate_stable_matrix("A", A)
    Sigma_mat = _mvsim_validate_spd_matrix("Sigma", Sigma)
    p = size(A_mat, 1)
    size(Sigma_mat, 1) == p || throw(ArgumentError("A and Sigma dimensions must match"))
    theta_vecs = [collect(Float64, th) for th in thetas]
    all(length(th) == p for th in theta_vecs) || throw(ArgumentError("All optimum vectors must match the matrix size"))

    root_regime = Int(edge_segments[tree.first_child_edge[tree.root]][1].state)
    root_vec = root_state === nothing ? copy(theta_vecs[root_regime]) : collect(Float64, root_state)
    length(root_vec) == p || throw(ArgumentError("root_state length must match matrix size"))
    ws = _mvsim_ou_workspace(A_mat, Sigma_mat)
    current = zeros(Float64, p)
    next = zeros(Float64, p)

    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= root_vec

    for node in tree.preorder
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            @views current .= node_states[node, :]
            for seg in edge_segments[edge]
                _mvsim_ou_step!(next, current, A_mat, Sigma_mat, theta_vecs[Int(seg.state)], seg.length, rng, ws)
                current, next = next, current
            end
            @views node_states[tree.child_of_edge[edge], :] .= current
        end
    end

    tip_data = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        @views tip_data[i, :] .= node_states[tip, :]
    end
    return tip_data
end

"""
    simulate_mvoumva(tree, edge_segments, As, Sigmas, thetas; root_state=nothing, rng=Random.GLOBAL_RNG)

Simulate an OUMVA trait matrix with regime-specific `A`, `Sigma`, and `theta`.
"""
function simulate_mvoumva(
    tree::CompactTree,
    edge_segments::Vector{Vector{SimmapSegment}},
    As::AbstractVector{<:AbstractMatrix},
    Sigmas::AbstractVector{<:AbstractMatrix},
    thetas::AbstractVector{<:AbstractVector};
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    _mvsim_validate_edge_segments(tree, edge_segments)
    nregimes = _mvsim_nregimes(edge_segments)
    length(As) == nregimes || throw(ArgumentError("As must have one matrix per regime (got $(length(As)) for $nregimes regimes)"))
    length(Sigmas) == nregimes || throw(ArgumentError("Sigmas must have one matrix per regime (got $(length(Sigmas)) for $nregimes regimes)"))
    length(thetas) == nregimes || throw(ArgumentError("thetas must have one optimum vector per regime (got $(length(thetas)) for $nregimes regimes)"))

    A_mats = [_mvsim_validate_stable_matrix("As[$i]", A) for (i, A) in enumerate(As)]
    Sigma_mats = [_mvsim_validate_spd_matrix("Sigmas[$i]", S) for (i, S) in enumerate(Sigmas)]

    p = size(A_mats[1], 1)
    all(A -> size(A) == (p, p), A_mats) || throw(ArgumentError("All A matrices must be $(p)x$(p)"))
    all(S -> size(S) == (p, p), Sigma_mats) || throw(ArgumentError("All Sigma matrices must be $(p)x$(p)"))

    theta_vecs = [collect(Float64, th) for th in thetas]
    all(th -> length(th) == p, theta_vecs) || throw(ArgumentError("All optimum vectors must match the matrix size ($p)"))

    root_regime = Int(edge_segments[tree.first_child_edge[tree.root]][1].state)
    root_vec = root_state === nothing ? copy(theta_vecs[root_regime]) : collect(Float64, root_state)
    length(root_vec) == p || throw(ArgumentError("root_state length must match the matrix size ($p)"))

    workspaces = [_mvsim_ou_workspace(A_mats[i], Sigma_mats[i]) for i in 1:nregimes]
    current = zeros(Float64, p)
    next = zeros(Float64, p)

    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= root_vec

    for node in tree.preorder
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            @views current .= node_states[node, :]
            for seg in edge_segments[edge]
                regime = Int(seg.state)
                _mvsim_ou_step!(next, current, A_mats[regime], Sigma_mats[regime], theta_vecs[regime], seg.length, rng, workspaces[regime])
                current, next = next, current
            end
            @views node_states[tree.child_of_edge[edge], :] .= current
        end
    end

    tip_data = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        @views tip_data[i, :] .= node_states[tip, :]
    end
    return tip_data
end
