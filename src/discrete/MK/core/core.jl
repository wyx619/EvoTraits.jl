"""
    tip_priors_from_states(tree, states, nstates)

Convert integer-encoded tip states into a dense `ntips x nstates` prior matrix.
Each tip gets a one-hot probability vector. State codes are expected in `1:nstates`.
"""
function tip_priors_from_states(tree::CompactTree, states::AbstractVector{<:Integer}, nstates::Integer)
    length(states) == tree.ntips || throw(ArgumentError("Expected $(tree.ntips) tip states, got $(length(states))"))
    nstates >= 2 || throw(ArgumentError("nstates must be at least 2"))
    priors = zeros(Float64, tree.ntips, nstates)
    for i in eachindex(states)
        state = Int(states[i])
        1 <= state <= nstates || throw(ArgumentError("Tip state $state is outside 1:$nstates"))
        priors[i, state] = 1.0
    end
    return priors
end

function _validate_tip_priors(tree::CompactTree, tip_priors::AbstractMatrix{<:Real})
    size(tip_priors, 1) == tree.ntips || throw(ArgumentError("tip_priors must have $(tree.ntips) rows"))
    size(tip_priors, 2) >= 2 || throw(ArgumentError("tip_priors must have at least 2 states"))
    priors = Matrix{Float64}(tip_priors)
    for i in axes(priors, 1)
        row_sum = sum(@view priors[i, :])
        row_sum > 0.0 || throw(ArgumentError("tip_priors row $i sums to zero"))
        @views priors[i, :] ./= row_sum
    end
    return priors
end

function _validate_rate_matrix(Q::AbstractMatrix{<:Real})
    size(Q, 1) == size(Q, 2) || throw(ArgumentError("Q must be square"))
    nstates = size(Q, 1)
    Qf = Matrix{Float64}(Q)
    for i in 1:nstates
        for j in 1:nstates
            if i != j && Qf[i, j] < 0.0
                throw(ArgumentError("Off-diagonal entries of Q must be non-negative"))
            end
        end
        rowsum = sum(@view Qf[i, :])
        abs(rowsum) <= 1e-8 || throw(ArgumentError("Each row of Q must sum to zero; row $i sums to $rowsum"))
    end
    return Qf
end

function _root_prior_vector(Q::Matrix{Float64}, root_prior::Symbol, root_prior_probs::Union{Nothing, AbstractVector{<:Real}})
    if root_prior === :custom
        root_prior_probs === nothing && throw(ArgumentError("root_prior_probs must be provided when root_prior=:custom"))
        return normalize_probability_vector(root_prior_probs)
    elseif root_prior === :stationary
        return stationary_distribution(Q)
    elseif root_prior === :flat
        return fill(1.0 / size(Q, 1), size(Q, 1))
    elseif root_prior === :likelihoods || root_prior === :max_likelihood
        return nothing
    else
        throw(ArgumentError("Unsupported root_prior=$root_prior"))
    end
end

function _mk_eigen_cache(Q::Matrix{Float64})
    F = eigen(ComplexF64.(Q))
    Icomplex = Matrix{ComplexF64}(I, size(F.vectors, 1), size(F.vectors, 2))
    return F.values, F.vectors, F.vectors \ Icomplex
end

function _fill_transition_matrix!(
    P::Matrix{Float64},
    t::Float64,
    evals::AbstractVector{<:Number},
    V::Matrix{ComplexF64},
    Vinv::Matrix{ComplexF64},
)
    nstates = size(P, 1)
    @inbounds for r in 1:nstates
        for c in 1:nstates
            acc = 0.0 + 0.0im
            for k in 1:nstates
                acc += V[r, k] * exp(evals[k] * t) * Vinv[k, c]
            end
            P[r, c] = max(real(acc), 0.0)
        end
    end
    for r in 1:nstates
        rowsum = 0.0
        @inbounds for c in 1:nstates
            rowsum += P[r, c]
        end
        rowsum > 0.0 || continue
        @inbounds for c in 1:nstates
            P[r, c] /= rowsum
        end
    end
end

function _default_mk_nparams(Q::AbstractMatrix{<:Real})
    n = size(Q, 1)
    total = 0
    for i in 1:n, j in 1:n
        if i != j && Q[i, j] > 0.0
            total += 1
        end
    end
    return total
end

function _ordered_state_labels(observed_states::AbstractVector, state_order)
    if state_order === nothing
        return Any[observed_states...]
    end

    ordered = Any[state for state in state_order]
    observed_set = Set(observed_states)
    ordered_set = Set(ordered)
    missing_states = setdiff(observed_set, ordered_set)
    extra_states = setdiff(ordered_set, observed_set)
    isempty(missing_states) || throw(ArgumentError("state_order is missing observed states: $(collect(missing_states))"))
    isempty(extra_states) || throw(ArgumentError("state_order contains unobserved states: $(collect(extra_states))"))
    length(ordered) == length(ordered_set) || throw(ArgumentError("state_order contains duplicate states"))
    return ordered
end

function _require_order_sensitive_state_order(rate_model::Symbol, state_order)
    if (rate_model === :SRD || rate_model === :SUEDE) && state_order === nothing
        throw(ArgumentError("rate_model=$rate_model is order-sensitive; pass complete state_order to define adjacent states"))
    end
    return nothing
end

function _encode_tip_states_from_mapping(tree, tip_states::AbstractDict; state_order = nothing)
    tip_labels = tree.tip_labels
    observed_states = Any[]

    for label in tip_labels
        haskey(tip_states, label) || throw(ArgumentError("Missing tip label '$label' in tip_states mapping"))
        state = tip_states[label]
        if !(state in observed_states)
            push!(observed_states, state)
        end
    end

    state_labels = _ordered_state_labels(observed_states, state_order)
    length(state_labels) >= 2 || throw(ArgumentError("Need at least two observed states for Mk ASR"))
    state_to_int = Dict{Any, Int}(state => i for (i, state) in enumerate(state_labels))

    encoded = Vector{Int}(undef, length(tip_labels))
    for (i, label) in enumerate(tip_labels)
        encoded[i] = state_to_int[tip_states[label]]
    end
    return encoded, state_labels
end

function _encode_tip_states_from_pairs(tree, tip_states_pairs::AbstractVector{<:Pair}; state_order = nothing)
    mapping = Dict{Any, Any}()
    for pair in tip_states_pairs
        mapping[pair.first] = pair.second
    end
    return _encode_tip_states_from_mapping(tree, mapping; state_order = state_order)
end
