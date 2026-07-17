function estim_node(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    if fit.model === :mvOU1 || fit.model === :mvOUM || fit.model === :mvOUMV || fit.model === :mvOUMA || fit.model === :mvOUMVA
        return _mvou_asr_dispatch(tree, trait, fit, edge_segments)
    end
    throw(ArgumentError("estim_node currently supports MVContinuousOUResult only for mvOU1, mvOUM, mvOUMV, mvOUMA, and mvOUMVA"))
end
