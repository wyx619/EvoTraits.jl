const CORHMM_MISSING_STRINGS = Set(["?", "missing", "NA", "NaN", "nan", ""])

_corhmm_state_string(x) = x === missing ? "missing" : strip(string(x))

function _is_corhmm_missing_state(x)
    x === missing && return true
    return _corhmm_state_string(x) in CORHMM_MISSING_STRINGS
end

function _split_corhmm_polymorphic_state(x)
    label = _corhmm_state_string(x)
    parts = String[strip(part) for part in split(label, '&')]
    any(isempty, parts) && throw(ArgumentError("Invalid polymorphic state '$label'"))
    return unique(parts)
end

function observed_tip_priors(raw_states::AbstractVector, observed_labels::Vector{String})
    nstates = length(observed_labels)
    nstates >= 2 || throw(ArgumentError("Need at least two observed states"))
    index = Dict(label => i for (i, label) in enumerate(observed_labels))
    priors = zeros(Float64, length(raw_states), nstates)
    tip_labels = Vector{String}(undef, length(raw_states))
    poly_mask = falses(length(raw_states))
    miss_mask = falses(length(raw_states))

    for (i, raw) in enumerate(raw_states)
        if _is_corhmm_missing_state(raw)
            priors[i, :] .= 1.0
            tip_labels[i] = "?"
            miss_mask[i] = true
            continue
        end

        parts = _split_corhmm_polymorphic_state(raw)
        tip_labels[i] = join(parts, "&")
        poly_mask[i] = length(parts) > 1
        for part in parts
            haskey(index, part) || throw(ArgumentError("State '$part' is not in state_order"))
            priors[i, index[part]] = 1.0
        end
    end
    return priors, tip_labels, poly_mask, miss_mask
end

function _corhmm_states_from_dataframe(tree::CompactTree, states)
    ncol(states) >= 2 || throw(ArgumentError("corHMM DataFrame input must have at least two columns: taxon, state"))
    taxon_col = states[!, 1]
    state_col = states[!, 2]
    mapping = Dict{String, Any}()
    for i in eachindex(taxon_col)
        mapping[string(taxon_col[i])] = state_col[i]
    end
    return [haskey(mapping, string(label)) ? mapping[string(label)] : throw(ArgumentError("Missing tip label '$label' in corHMM states")) for label in tree.tip_labels]
end

function _corhmm_states_from_mapping(tree::CompactTree, states::AbstractDict)
    return [haskey(states, string(label)) ? states[string(label)] :
            haskey(states, label) ? states[label] :
            throw(ArgumentError("Missing tip label '$label' in corHMM states")) for label in tree.tip_labels]
end

function _corhmm_states_from_vector(tree::CompactTree, states::AbstractVector)
    length(states) == tree.ntips || throw(ArgumentError("Expected $(tree.ntips) states, got $(length(states))"))
    return collect(states)
end

function _collect_corhmm_observed_labels(raw_states::AbstractVector, state_order)
    if state_order !== nothing
        labels = String[string(label) for label in state_order]
        length(labels) == length(unique(labels)) || throw(ArgumentError("state_order contains duplicate states"))
        length(labels) >= 2 || throw(ArgumentError("state_order must contain at least two states"))
        return labels
    end

    labels = String[]
    for raw in raw_states
        _is_corhmm_missing_state(raw) && continue
        for part in _split_corhmm_polymorphic_state(raw)
            part in labels || push!(labels, part)
        end
    end
    length(labels) >= 2 || throw(ArgumentError("Need at least two observed non-missing states"))
    sort!(labels)
    return labels
end

function parsestates(tree::CompactTree, states; state_order = nothing, rate_cat::Integer = 1)
    rc = _validate_corhmm_rate_cat(rate_cat)
    raw_states =
        if states isa AbstractDict
            _corhmm_states_from_mapping(tree, states)
        elseif states isa AbstractVector
            _corhmm_states_from_vector(tree, states)
        else
            _corhmm_states_from_dataframe(tree, states)
        end

    observed_labels = _collect_corhmm_observed_labels(raw_states, state_order)
    observed_priors, tip_labels, poly_mask, miss_mask = observed_tip_priors(raw_states, observed_labels)
    hidden_labels, hidden_to_observed = hidden_state_labels(observed_labels, rc)
    hidden_priors = expand_tip_priors(observed_priors, rc)

    return CorHMMStateData(
        observed_labels = observed_labels,
        hidden_labels = hidden_labels,
        hidden_to_observed = hidden_to_observed,
        tip_priors_observed = observed_priors,
        tip_priors_hidden = hidden_priors,
        tip_state_labels = tip_labels,
        polymorphic_tip_mask = poly_mask,
        missing_tip_mask = miss_mask,
        rate_cat = rc,
    )
end

parse_corhmm_states(args...; kwargs...) = parsestates(args...; kwargs...)
