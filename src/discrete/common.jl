const EVOTRAITS_TINY = 1.0e-300

@inline function logaddexp2(x::Float64, y::Float64)
    if x == -Inf
        return y
    elseif y == -Inf
        return x
    elseif x > y
        return x + log1p(exp(y - x))
    else
        return y + log1p(exp(x - y))
    end
end

function normalize_probability_vector(prob::AbstractVector{<:Real})
    p = Float64.(prob)
    any(x -> x < 0.0, p) && throw(ArgumentError("Probability vector contains negative entries"))
    total = sum(p)
    total > 0.0 || throw(ArgumentError("Probability vector must have positive sum"))
    return p ./ total
end

function stationary_distribution(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    n = size(Q, 1)
    A = Matrix{Float64}(transpose(Q))
    A[end, :] .= 1.0
    b = zeros(Float64, n)
    b[end] = 1.0
    pi_vec = A \ b
    return normalize_probability_vector(pi_vec)
end

@inline function foreach_child_edge(tree::CompactTree, node::Integer, f::F) where {F}
    first_edge = Int(tree.first_child_edge[node])
    last_edge = Int(tree.last_child_edge[node])
    for edge in first_edge:last_edge
        f(edge, Int(tree.child_of_edge[edge]))
    end
    return nothing
end

@inline function foreach_child_edge(f::F, tree::CompactTree, node::Integer) where {F}
    return foreach_child_edge(tree, node, f)
end

@inline function foreach_postorder_internal(tree::CompactTree, f::F) where {F}
    for node in tree.postorder_internal
        f(Int(node))
    end
    return nothing
end

@inline function foreach_postorder_internal(f::F, tree::CompactTree) where {F}
    return foreach_postorder_internal(tree, f)
end
