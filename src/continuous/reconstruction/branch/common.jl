@inline function _mvou_empty_kernel(p::Integer)
    return (Phi = Matrix{Float64}(I, p, p), Q = zeros(Float64, p, p))
end

function _mvbranch_push_point!(
    means::Vector{Vector{Float64}},
    covs::Vector{Matrix{Float64}},
    mean::AbstractVector{<:Real},
    cov::AbstractMatrix{<:Real},
)
    push!(means, Vector{Float64}(mean))
    push!(covs, Matrix{Float64}(cov))
    return nothing
end

function _mvbranch_pack_means(means::Vector{Vector{Float64}}, p::Integer)
    out = Matrix{Float64}(undef, length(means), p)
    for (i, mean) in enumerate(means)
        out[i, :] .= mean
    end
    return out
end

function _mvbranch_pack_covariances(covs::Vector{Matrix{Float64}}, p::Integer)
    out = Array{Float64, 3}(undef, length(covs), p, p)
    for (i, cov) in enumerate(covs)
        out[i, :, :] .= cov
    end
    return out
end

function _mv_kernel_compose(first, second)
    Phi = second.Phi * first.Phi
    Q = second.Phi * first.Q * second.Phi' + second.Q
    return (Phi = Phi, Q = (Q + Q') / 2)
end

@inline function _mv_identity_predict_to_child(
    parent_mean::AbstractVector{<:Real},
    parent_cov::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
)
    cov = Matrix{Float64}(parent_cov) + Matrix{Float64}(Q)
    return (mean = Vector{Float64}(parent_mean), cov = (cov + cov') / 2)
end
