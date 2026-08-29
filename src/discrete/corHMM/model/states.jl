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
        ismissing(taxon_col[i]) && continue
        mapping[string(taxon_col[i])] = state_col[i]
    end
    return [get(mapping, string(label), missing) for label in tree.tip_labels]
end

function _corhmm_tip_fog_spec(tip_fog, nstates::Integer)
    tip_fog === nothing && return (estimated = false, groups = Int[], fixed = nothing)
    if tip_fog === :estimate
        return (estimated = true, groups = ones(Int, nstates), fixed = nothing)
    elseif tip_fog isa Real
        value = Float64(tip_fog)
        0.0 <= value < 0.5 || throw(ArgumentError("tip_fog must be in [0, 0.5)"))
        return (estimated = false, groups = Int[], fixed = fill(value, nstates))
    elseif tip_fog isa AbstractVector
        values = collect(tip_fog)
        length(values) == 1 && return _corhmm_tip_fog_spec(values[1], nstates)
        length(values) == nstates || throw(ArgumentError("tip_fog must have length 1 or the number of observed states"))
        numeric = Float64.(values)
        if all(0.0 <= value < 0.5 for value in numeric)
            return (estimated = false, groups = Int[], fixed = numeric)
        end
        all(x -> x isa Integer && Int(x) >= 1, values) || throw(ArgumentError("Estimated tip_fog groups must be positive integers"))
        return (estimated = true, groups = Int.(values), fixed = nothing)
    end
    throw(ArgumentError("Unsupported tip_fog; use nothing, a probability, a probability vector, :estimate, or integer groups"))
end

function _corhmm_apply_tip_fog(priors::AbstractMatrix{<:Real}, fog_probs::AbstractVector{<:Real})
    nstates = size(priors, 2)
    length(fog_probs) == nstates || throw(ArgumentError("tip_fog probabilities and state count disagree"))
    all(value -> 0.0 <= value < 0.5, fog_probs) || throw(ArgumentError("tip_fog probabilities must be in [0, 0.5)"))
    out = Matrix{Float64}(priors)
    for i in axes(out, 1)
        known = findall(>(0.0), @view out[i, :])
        length(known) == nstates && continue
        unknown = findall(==(0.0), @view out[i, :])
        error_mass = sum(fog_probs[unknown])
        error_mass < 1.0 || throw(ArgumentError("tip_fog probabilities leave no mass for observed states"))
        @views out[i, known] .= 1.0 - error_mass
        @views out[i, unknown] .= fog_probs[unknown]
    end
    return out
end

function _corhmm_state_data_with_tip_fog(state_data::CorHMMStateData, fog_probs::AbstractVector{<:Real})
    observed_priors = _corhmm_apply_tip_fog(state_data.tip_priors_observed, fog_probs)
    return CorHMMStateData(
        observed_labels = state_data.observed_labels,
        possible_labels = state_data.possible_labels,
        character_labels = state_data.character_labels,
        character_state_labels = state_data.character_state_labels,
        multi_character = state_data.multi_character,
        hidden_labels = state_data.hidden_labels,
        hidden_to_observed = state_data.hidden_to_observed,
        tip_priors_observed = observed_priors,
        tip_priors_hidden = expand_tip_priors(observed_priors, state_data.rate_cat),
        tip_state_labels = state_data.tip_state_labels,
        polymorphic_tip_mask = state_data.polymorphic_tip_mask,
        missing_tip_mask = state_data.missing_tip_mask,
        rate_cat = state_data.rate_cat,
    )
end

function _corhmm_tip_priors_with_fog(
    hidden_priors::AbstractMatrix{<:Real},
    fog_values::AbstractVector{<:Real},
    fog_groups::AbstractVector{<:Integer},
    hidden_to_observed::AbstractVector{<:Integer},
)
    isempty(fog_groups) && return Matrix{Float64}(hidden_priors)
    nobs = maximum(hidden_to_observed)
    length(fog_groups) == nobs || throw(ArgumentError("tip_fog groups must have one entry per observed state"))
    length(fog_values) == length(unique(fog_groups)) || throw(ArgumentError("tip_fog values and groups disagree"))
    group_to_value = Dict{Int, Float64}(Int(group) => Float64(value) for (group, value) in zip(sort!(unique(Int.(fog_groups))), fog_values))
    fog_probs = Float64[group_to_value[Int(group)] for group in fog_groups]
    observed_priors = zeros(Float64, size(hidden_priors, 1), nobs)
    filled = falses(nobs)
    for hidden_state in axes(hidden_priors, 2)
        observed_state = Int(hidden_to_observed[hidden_state])
        if !filled[observed_state]
            @views observed_priors[:, observed_state] .= hidden_priors[:, hidden_state]
            filled[observed_state] = true
        end
    end
    return expand_tip_priors(_corhmm_apply_tip_fog(observed_priors, fog_probs), length(hidden_to_observed) ÷ nobs)
end

function _corhmm_fog_values_by_observed_state(
    fog_values::AbstractVector{<:Real},
    fog_group_levels::AbstractVector{<:Integer},
    fog_groups::AbstractVector{<:Integer},
)
    lookup = Dict{Int, Float64}(Int(group) => Float64(value) for (group, value) in zip(fog_group_levels, fog_values))
    return Float64[lookup[Int(group)] for group in fog_groups]
end

function _corhmm_dataframe_rows(tree::CompactTree, states)
    nchar = ncol(states) - 1
    nchar >= 2 || throw(ArgumentError("Expected at least two character columns"))
    taxon_col = states[!, 1]
    columns = [states[!, j] for j in 2:ncol(states)]
    mapping = Dict{String, Int}()
    for i in eachindex(taxon_col)
        ismissing(taxon_col[i]) && continue
        mapping[string(taxon_col[i])] = i
    end
    rows = Vector{Vector{Any}}(undef, tree.ntips)
    for (tip_i, label) in enumerate(tree.tip_labels)
        row_i = get(mapping, string(label), 0)
        rows[tip_i] = row_i == 0 ? Any[missing for _ in 1:nchar] : Any[columns[j][row_i] for j in 1:nchar]
    end
    return rows
end

function _corhmm_character_levels(rows::Vector{Vector{Any}}, state_order, nchar::Integer)
    if state_order === nothing
        levels = Vector{Vector{String}}(undef, nchar)
        for j in 1:nchar
            labels = String[]
            for row in rows
                value = row[j]
                _is_corhmm_missing_state(value) && continue
                for part in _split_corhmm_polymorphic_state(value)
                    part in labels || push!(labels, part)
                end
            end
            length(labels) >= 2 || throw(ArgumentError("Each corHMM character must have at least two observed states"))
            sort!(labels)
            levels[j] = labels
        end
        return levels
    end

    state_order isa AbstractVector || throw(ArgumentError("For multiple corHMM characters, state_order must be a vector of state-order vectors"))
    length(state_order) == nchar || throw(ArgumentError("state_order must contain one vector for each character"))
    levels = Vector{Vector{String}}(undef, nchar)
    for j in 1:nchar
        order = state_order[j]
        order isa AbstractVector || throw(ArgumentError("Each multiple-character state_order entry must be a vector"))
        labels = String[string(x) for x in order]
        length(labels) >= 2 || throw(ArgumentError("Each character state_order must contain at least two states"))
        length(unique(labels)) == length(labels) || throw(ArgumentError("Character state_order contains duplicate states"))
        levels[j] = labels
    end
    return levels
end

function _corhmm_joint_labels(levels::Vector{Vector{String}})
    products = collect(Iterators.product(levels...))
    return String[join(combo, "_") for combo in vec(products)]
end

function _corhmm_matching_joint_labels(row::Vector{Any}, levels::Vector{Vector{String}}, possible_labels::Vector{String})
    allowed = Vector{Set{String}}(undef, length(levels))
    for j in eachindex(levels)
        value = row[j]
        if _is_corhmm_missing_state(value)
            allowed[j] = Set(levels[j])
        else
            parts = _split_corhmm_polymorphic_state(value)
            all(part -> part in levels[j], parts) || throw(ArgumentError("State '$value' is not in the supplied state_order"))
            allowed[j] = Set(parts)
        end
    end
    matches = Int[]
    for (i, label) in enumerate(possible_labels)
        parts = split(label, '_')
        all(parts[j] in allowed[j] for j in eachindex(levels)) && push!(matches, i)
    end
    isempty(matches) && throw(ArgumentError("No valid joint state matches a tip character combination"))
    return matches
end

function _parse_corhmm_multichar_states(tree::CompactTree, states; collapse::Bool, state_order)
    rows = _corhmm_dataframe_rows(tree, states)
    nchar = length(first(rows))
    levels = _corhmm_character_levels(rows, state_order, nchar)
    possible_labels = _corhmm_joint_labels(levels)
    matching = [_corhmm_matching_joint_labels(row, levels, possible_labels) for row in rows]
    observed_indices = sort!(unique(vcat(matching...)))
    observed_labels = collapse ? possible_labels[observed_indices] : possible_labels
    observed_lookup = Dict(label => i for (i, label) in enumerate(observed_labels))
    priors = zeros(Float64, tree.ntips, length(observed_labels))
    tip_labels = Vector{String}(undef, tree.ntips)
    poly_mask = falses(tree.ntips)
    miss_mask = falses(tree.ntips)
    for i in eachindex(rows)
        labels = possible_labels[matching[i]]
        for label in labels
            haskey(observed_lookup, label) && (priors[i, observed_lookup[label]] = 1.0)
        end
        tip_labels[i] = any(_is_corhmm_missing_state(value) for value in rows[i]) ? "?" : join(labels, "&")
        poly_mask[i] = any(!_is_corhmm_missing_state(value) && length(_split_corhmm_polymorphic_state(value)) > 1 for value in rows[i])
        miss_mask[i] = any(_is_corhmm_missing_state, rows[i])
    end
    return (
        observed_labels = observed_labels,
        possible_labels = possible_labels,
        character_labels = String[string(names(states)[j]) for j in 2:ncol(states)],
        character_state_labels = levels,
        tip_priors = priors,
        tip_labels = tip_labels,
        poly_mask = poly_mask,
        miss_mask = miss_mask,
    )
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

function parsestates(tree::CompactTree, states; state_order = nothing, rate_cat::Integer = 1, collapse::Bool = true)
    rc = _validate_corhmm_rate_cat(rate_cat)
    if !(states isa AbstractDict || states isa AbstractVector) && ncol(states) > 2
        joint = _parse_corhmm_multichar_states(tree, states; collapse = collapse, state_order = state_order)
        hidden_labels, hidden_to_observed = hidden_state_labels(joint.observed_labels, rc)
        hidden_priors = expand_tip_priors(joint.tip_priors, rc)
        return CorHMMStateData(
            observed_labels = joint.observed_labels,
            possible_labels = joint.possible_labels,
            character_labels = joint.character_labels,
            character_state_labels = joint.character_state_labels,
            multi_character = true,
            hidden_labels = hidden_labels,
            hidden_to_observed = hidden_to_observed,
            tip_priors_observed = joint.tip_priors,
            tip_priors_hidden = hidden_priors,
            tip_state_labels = joint.tip_labels,
            polymorphic_tip_mask = joint.poly_mask,
            missing_tip_mask = joint.miss_mask,
            rate_cat = rc,
        )
    end
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
        possible_labels = observed_labels,
        character_labels = ["character_1"],
        character_state_labels = [observed_labels],
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
