"""
    _validate_multivariate_trait(tree, trait)

Shared input validation for EvoTraits multivariate workflows.

Current hard boundary:
- matrix input with at least 1 trait
- at most 4000 tips
- finite numeric data, with NaN accepted as missing
"""
@inline function _validate_multivariate_trait(tree::CompactTree, trait::AbstractMatrix{<:Real})
    size(trait, 1) == tree.ntips || throw(ArgumentError("trait matrix must have $(tree.ntips) rows"))
    size(trait, 2) >= 1 || throw(ArgumentError("trait matrix must have at least 1 column"))
    tree.ntips <= 4000 || throw(ArgumentError("multivariate EvoTraits currently supports at most 4000 tips"))
    data = trait isa Matrix{Float64} ? trait : Matrix{Float64}(trait)
    any(isinf, data) && throw(ArgumentError("trait matrix contains infinite values"))
    all(isnan, data) && throw(ArgumentError("trait matrix contains no observed values"))
    for j in axes(data, 2)
        all(isnan, @view data[:, j]) && throw(ArgumentError("trait column $j contains no observed values"))
    end
    return data
end

mutable struct MVProfileWorkspace
    precision::Vector{Matrix{Float64}}
    linear::Vector{Vector{Float64}}
    logconst::Vector{Float64}
    tip_index::Vector{Int}
    msg_precision::Matrix{Float64}
    msg_linear::Vector{Float64}
    Qobs::Matrix{Float64}
    yobs::Vector{Float64}
    obs_index::Vector{Int}
    Qinv_y::Vector{Float64}
    solve_matrix::Matrix{Float64}
    solve_vector::Vector{Float64}
    edge_Q::Array{Float64, 3}
    edge_Qinv::Array{Float64, 3}
    edge_logdet_Q::Vector{Float64}
    edge_chol_work::Matrix{Float64}
end

function _mv_profile_workspace(tree::CompactTree, p::Integer)
    tip_index = zeros(Int, tree.nnodes)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_index[tip] = i
    end
    return MVProfileWorkspace(
        [zeros(Float64, p, p) for _ in 1:tree.nnodes],
        [zeros(Float64, p) for _ in 1:tree.nnodes],
        zeros(Float64, tree.nnodes),
        tip_index,
        zeros(Float64, p, p),
        zeros(Float64, p),
        zeros(Float64, p, p),
        zeros(Float64, p),
        zeros(Int, p),
        zeros(Float64, p),
        zeros(Float64, p, p),
        zeros(Float64, p),
        Array{Float64, 3}(undef, p, p, tree.nedges),
        Array{Float64, 3}(undef, p, p, tree.nedges),
        zeros(Float64, tree.nedges),
        zeros(Float64, p, p),
    )
end

function _mv_complete_rows_cov(data::AbstractMatrix{<:Real})
    mat = Matrix{Float64}(data)
    p = size(mat, 2)
    observed_all = filter(!isnan, vec(mat))
    isempty(observed_all) && throw(ArgumentError("trait matrix contains no observed values"))
    global_var = length(observed_all) >= 2 ? max(var(observed_all), 1e-8) : 1.0
    empirical = zeros(Float64, p, p)
    for j in 1:p
        xj = filter(!isnan, @view mat[:, j])
        isempty(xj) && throw(ArgumentError("trait column $j contains no observed values"))
        empirical[j, j] = length(xj) >= 2 ? max(var(xj), 1e-8) : global_var
    end
    for i in 1:p, j in (i + 1):p
        keep = [!isnan(mat[row, i]) && !isnan(mat[row, j]) for row in axes(mat, 1)]
        if count(keep) >= 2
            xi = mat[keep, i]
            xj = mat[keep, j]
            cij = dot(xi .- mean(xi), xj .- mean(xj)) / (length(xi) - 1)
            empirical[i, j] = cij
            empirical[j, i] = cij
        end
    end
    return (empirical + empirical') / 2
end

@inline function _mvou_is_zero_cov(S::AbstractMatrix{<:Real}; atol::Float64 = 1e-12)
    return maximum(abs, S) <= atol
end

"""
    _mvou_cholesky_psd(S; jitter=1e-10, max_tries=6)

Prefer `Cholesky(Symmetric(...))` for multivariate covariance operators and
apply a small diagonal jitter ladder when necessary.
"""
function _mvou_cholesky_psd(S::AbstractMatrix{<:Real}; jitter::Float64 = 1e-10, max_tries::Int = 6)
    p = size(S, 1)
    size(S, 2) == p || throw(ArgumentError("matrix must be square"))
    M = Matrix{Float64}(undef, p, p)
    @inbounds for j in 1:p
        M[j, j] = Float64(S[j, j])
        for i in (j + 1):p
            v = 0.5 * (Float64(S[i, j]) + Float64(S[j, i]))
            M[i, j] = v
            M[j, i] = v
        end
    end
    for k in 0:max_tries
        shift = k == 0 ? 0.0 : jitter * 10.0^(k - 1)
        T = copy(M)
        if shift != 0.0
            @inbounds for i in 1:p
                T[i, i] += shift
            end
        end
        try
            return cholesky!(Symmetric(T))
        catch err
            err isa PosDefException || rethrow()
        end
    end
    throw(PosDefException(p))
end

function _mvou_context_with_info(
    mean::AbstractVector{<:Real},
    cov::AbstractMatrix{<:Real},
    precision::AbstractMatrix{<:Real},
    linear::AbstractVector{<:Real},
)
    m = Vector{Float64}(mean)
    C = Matrix{Float64}(cov)
    P = Matrix{Float64}(precision)
    h = Vector{Float64}(linear)
    p = length(m)
    if _mvou_is_zero_cov(C)
        return (success = true, mean = copy(m), cov = copy(C))
    end
    try
        M = Matrix{Float64}(I, p, p) + C * P
        r = h - P * m
        Cpost = M \ C
        mpost = m + Cpost * r
        return (success = true, mean = mpost, cov = (Cpost + Cpost') / 2)
    catch
        return (success = false, mean = zeros(Float64, p), cov = zeros(Float64, p, p))
    end
end

