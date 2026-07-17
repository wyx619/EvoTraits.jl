@inline function _estim_node_from_simmap(
    tree::CompactTree,
    trait,
    fit,
    simmap::SimmapSample,
)
    return estim_node(tree, trait, fit; mapped_edge = simmap.mapped_edge, edge_segments = simmap.edge_segments)
end

function estim_node(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    fit::ContinuousMultiRegimeResult,
    simmap::SimmapSample,
)
    return _estim_node_from_simmap(tree, trait, fit, simmap)
end

function estim_node(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousOUResult,
    simmap::SimmapSample,
)
    return _estim_node_from_simmap(tree, trait, fit, simmap)
end

function estim_node(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousMultiBMResult,
    simmap::SimmapSample,
)
    return _estim_node_from_simmap(tree, trait, fit, simmap)
end
