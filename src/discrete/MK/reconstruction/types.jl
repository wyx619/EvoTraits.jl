"""
    MkASRResult

Result type returned by Mk ancestral state reconstruction. In addition to the
fitted log-likelihood and AIC, it stores per-node ancestral likelihoods,
the most likely ancestral states, and the underlying fitted Mk model.
"""
Base.@kwdef struct MkASRResult
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    nstates::Int = 0
    root_prior::Symbol = :likelihoods
    rate_model::Symbol = :ER
    rates::Vector{Float64} = Float64[]
    transition_matrix::Matrix{Float64} = zeros(0, 0)
    node_ids::Vector{Int} = Int[]
    ancestral_likelihoods::Matrix{Float64} = zeros(0, 0)
    ancestral_states::Vector{Int32} = Int32[]
    reroot::Bool = true
    fit::MkFitResult = MkFitResult()
    state_labels::Vector{Any} = Vector{Any}()
    state_encoding::Dict{Any, Int} = Dict{Any, Int}()
    ancestral_state_labels::Vector{Any} = Vector{Any}()
end

function Base.show(io::IO, res::MkASRResult)
    labels = isempty(res.state_labels) ? "" : ", state_labels=$(res.state_labels)"
    print(io, "MkASRResult(success=$(res.success), states=$(res.nstates), nodes=$(length(res.node_ids)), loglik=$(res.loglik), aic=$(res.aic), reroot=$(res.reroot)$(labels))")
end

function Base.show(io::IO, ::MIME"text/plain", res::MkASRResult)
    show(io, res)
end
