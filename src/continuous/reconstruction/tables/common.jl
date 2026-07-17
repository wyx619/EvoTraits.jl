function _attach_simmap_node_regime!(
    df::DataFrame,
    tree::CompactTree,
    node_ids,
    simmap::SimmapSample,
)
    length(simmap.node_states) == tree.nnodes || throw(ArgumentError("simmap.node_states must match tree.nnodes"))
    regime_id = Int32[simmap.node_states[Int(node)] for node in node_ids]
    regime = Vector{String}(undef, length(regime_id))
    for i in eachindex(regime_id)
        state = Int(regime_id[i])
        regime[i] =
            1 <= state <= length(simmap.state_labels) ? String(simmap.state_labels[state]) :
            string(state)
    end
    df[!, :regime_id] = Int.(regime_id)
    df[!, :regime] = regime
    return df
end

@inline function _result_trait_names(fit::AbstractMVContinuousFitResult, p::Integer)
    names = getproperty(fit, :trait_names)
    return isempty(names) ? _mv_default_trait_names(p) : String.(names)
end

@inline function _mv_table_safe_name(name::AbstractString)
    raw = strip(String(name))
    isempty(raw) && return "trait"
    safe = replace(raw, r"[^A-Za-z0-9_]+" => "_")
    safe = replace(safe, r"_+" => "_")
    safe = strip(safe, '_')
    isempty(safe) && return "trait"
    return safe
end
