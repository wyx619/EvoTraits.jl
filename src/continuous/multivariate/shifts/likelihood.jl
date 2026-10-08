function _shift_oubm_profile(tree, data, mapping, spec, factor, primary, secondary; theta = nothing, beta = 0.0)
    try
        edges = _shift_oubm_edges(tree, mapping, spec, factor, primary, secondary; beta = beta)
        return _shift_augmented_profile(tree, data, edges; theta = theta)
    catch err
        err isa PosDefException || err isa OverflowError || rethrow()
        return (success = false, loglik = -Inf,
            theta = theta === nothing ? fill(NaN, size(data, 2)) : theta)
    end
end

function _shift_propagate_message(P, h, c, F, L)
    isempty(L) && return (F' * P * F, F' * h, c)
    PL = P * L
    B = cholesky(Symmetric(Matrix{Float64}(I, size(L, 2), size(L, 2)) + L' * PL))
    noise_linear = L' * h
    precision = F' * (P - PL * (B \ PL')) * F
    linear = F' * (h - PL * (B \ noise_linear))
    constant = c - 0.5 * logdet(B) + 0.5 * dot(noise_linear, B \ noise_linear)
    return ((precision + precision') / 2, linear, constant)
end

function _shift_checked_cholesky(matrix)
    values = eigvals(Symmetric(matrix))
    all(isfinite, values) && minimum(values) > maximum(values) * 1e-12 ||
        throw(PosDefException(0))
    return cholesky(Symmetric(matrix))
end

function _shift_augmented_profile(tree::CompactTree, data::Matrix{Float64}, edges; theta = nothing)
    p = size(data, 2)
    d = 3p
    P = [zeros(d, d) for _ in 1:tree.nnodes]
    h = [zeros(d) for _ in 1:tree.nnodes]
    constants = zeros(tree.nnodes)
    H = hcat(Matrix{Float64}(I, p, p), Matrix{Float64}(I, p, p), Matrix{Float64}(I, p, p))
    tip_index = zeros(Int, tree.nnodes)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_index[tip] = i
    end
    for node in tree.postorder_internal
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = tree.child_of_edge[edge]
            if tree.is_tip[child]
                i = tip_index[child]
                observed = findall(!isnan, @view data[i, :])
                isempty(observed) && continue
                observation = H[observed, :]
                F = observation * edges.Phi[edge]
                R = _shift_checked_cholesky(observation * edges.Q[edge] * observation')
                y = data[i, observed]
                msgP = F' * (R \ F)
                msgh = F' * (R \ y)
                msgc = -0.5 * (dot(y, R \ y) + logdet(R) + length(observed) * log(2pi))
            else
                msgP, msgh, msgc = _shift_propagate_message(P[child], h[child], constants[child],
                    edges.Phi[edge], edges.L[edge])
            end
            P[node] .+= msgP
            h[node] .+= msgh
            constants[node] += msgc
        end
    end
    # The two random components start at zero; only the constant theta block is profiled.
    indices = 2p + 1:3p
    rootP = Symmetric(P[tree.root][indices, indices])
    rooth = h[tree.root][indices]
    mean = theta === nothing ? _shift_checked_cholesky(rootP) \ rooth : theta
    ll = constants[tree.root] - 0.5 * dot(mean, rootP * mean) + dot(rooth, mean)
    return (success = isfinite(ll), loglik = ll, theta = mean)
end
