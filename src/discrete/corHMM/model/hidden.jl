function hidden_state_labels(observed_labels::AbstractVector{<:AbstractString}, rate_cat::Integer)
    rc = _validate_corhmm_rate_cat(rate_cat)
    hidden_labels = String[]
    hidden_to_observed = Int[]
    for h in 1:rc
        for (i, label) in enumerate(observed_labels)
            push!(hidden_labels, rc == 1 ? string(label) : string(label, "_R", h))
            push!(hidden_to_observed, i)
        end
    end
    return hidden_labels, hidden_to_observed
end

function expand_tip_priors(observed_priors::AbstractMatrix{<:Real}, rate_cat::Integer)
    rc = _validate_corhmm_rate_cat(rate_cat)
    ntips, nobs = size(observed_priors)
    out = zeros(Float64, ntips, nobs * rc)
    for h in 1:rc
        offset = (h - 1) * nobs
        @views out[:, (offset + 1):(offset + nobs)] .= observed_priors
    end
    return out
end

function collapse_hidden_likelihoods(hidden_likelihoods::AbstractMatrix{<:Real}, hidden_to_observed::AbstractVector{<:Integer}, nobs::Integer)
    out = zeros(Float64, size(hidden_likelihoods, 1), nobs)
    for h in axes(hidden_likelihoods, 2)
        obs = Int(hidden_to_observed[h])
        @views out[:, obs] .+= hidden_likelihoods[:, h]
    end
    for i in axes(out, 1)
        row_sum = sum(@view out[i, :])
        row_sum > 0.0 || continue
        @views out[i, :] ./= row_sum
    end
    return out
end

corhmm_hidden_state_labels(args...; kwargs...) = hidden_state_labels(args...; kwargs...)
expand_corhmm_tip_priors(args...; kwargs...) = expand_tip_priors(args...; kwargs...)
collapse_corhmm_hidden_likelihoods(args...; kwargs...) = collapse_hidden_likelihoods(args...; kwargs...)
