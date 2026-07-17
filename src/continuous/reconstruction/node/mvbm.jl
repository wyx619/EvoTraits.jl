function estim_node(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousBMResult;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    if fit.model === :mvBM1
        return _mvbm1_asr(tree, trait, fit)
    elseif fit.model === :mvEB
        return _mveb_asr(tree, trait, fit)
    end
    throw(ArgumentError("estim_node currently supports MVContinuousBMResult models mvBM1 and mvEB"))
end

function estim_node(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousMultiBMResult;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    if fit.model === :mvBMM
        mapped = _resolve_mapped_edge_context(tree, mapped_edge, edge_segments)
        return _mvbmm_asr(tree, trait, fit, mapped)
    end
    throw(ArgumentError("estim_node currently supports MVContinuousMultiBMResult for mvBMM"))
end
