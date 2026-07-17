"""
    simulate_mveb(tree, Sigma, beta; root_state=nothing, rng=Random.GLOBAL_RNG)

Simulate an early-burst trait matrix with shared `Sigma` and scalar `beta`.
"""
function simulate_mveb(
    tree::CompactTree,
    Sigma::AbstractMatrix,
    beta::Real;
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    Sigma_mat = _mvsim_validate_spd_matrix("Sigma", Sigma)
    p = size(Sigma_mat, 1)
    L = Matrix{Float64}(cholesky(Symmetric(Sigma_mat)).L)
    ws = _mvsim_gaussian_workspace(p)
    root_vec = root_state === nothing ? zeros(Float64, p) : collect(Float64, root_state)
    length(root_vec) == p || throw(ArgumentError("root_state length must match covariance size"))
    beta_val = Float64(beta)

    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= root_vec

    for node in tree.preorder
        parent_state = @view node_states[node, :]
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            parent_time = tree.dist_from_root[node]
            child_time = tree.dist_from_root[child]
            scale =
                if abs(beta_val) < 1e-12
                    child_time - parent_time
                else
                    (exp(beta_val * child_time) - exp(beta_val * parent_time)) / beta_val
                end
            scale >= 0.0 || throw(ArgumentError("EB branch variance increment must be non-negative"))
            child_state = @view node_states[child, :]
            child_state .= parent_state
            _mvsim_add_chol_noise!(child_state, L, scale, rng, ws)
        end
    end

    tip_data = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        @views tip_data[i, :] .= node_states[tip, :]
    end
    return tip_data
end
