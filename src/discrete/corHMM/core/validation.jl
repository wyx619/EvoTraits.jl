function validateratecat(rate_cat::Integer)
    rate_cat >= 1 || throw(ArgumentError("rate_cat must be >= 1"))
    return Int(rate_cat)
end

function validatenodestates(node_states::Symbol)
    node_states in (:marginal, :joint, :scaled, :none) || throw(ArgumentError("node_states must be one of :marginal, :joint, :scaled, :none"))
    return node_states
end

function resolverootprior(root_prior)
    if root_prior === :yang
        return :yang
    elseif root_prior === :flat
        return :flat
    elseif root_prior === :maddfitz
        return :maddfitz
    elseif root_prior isa AbstractVector
        return root_prior
    end
    throw(ArgumentError("Unsupported corHMM root_prior=$root_prior; use :yang, :flat, :maddfitz, or a numeric probability vector"))
end

_validate_corhmm_rate_cat(args...; kwargs...) = validateratecat(args...; kwargs...)
_validate_corhmm_node_states(args...; kwargs...) = validatenodestates(args...; kwargs...)
_corhmm_root_prior(args...; kwargs...) = resolverootprior(args...; kwargs...)
