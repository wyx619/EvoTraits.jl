Base.@kwdef struct CorHMMStateData
    observed_labels::Vector{String} = String[]
    possible_labels::Vector{String} = String[]
    character_labels::Vector{String} = String[]
    character_state_labels::Vector{Vector{String}} = Vector{Vector{String}}()
    multi_character::Bool = false
    hidden_labels::Vector{String} = String[]
    hidden_to_observed::Vector{Int} = Int[]
    tip_priors_observed::Matrix{Float64} = zeros(0, 0)
    tip_priors_hidden::Matrix{Float64} = zeros(0, 0)
    tip_state_labels::Vector{String} = String[]
    polymorphic_tip_mask::BitVector = falses(0)
    missing_tip_mask::BitVector = falses(0)
    rate_cat::Int = 1
end

Base.@kwdef struct CorHMMFitResult
    tree::Union{Nothing, CompactTree} = nothing
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    aicc::Float64 = NaN
    nparams::Int = 0
    model::Symbol = :ARD
    rate_cat::Int = 1
    node_states::Symbol = :marginal
    root_prior::Any = :yang
    collapse::Bool = true
    tip_fog::Any = nothing
    observed_labels::Vector{String} = String[]
    hidden_labels::Vector{String} = String[]
    hidden_to_observed::Vector{Int} = Int[]
    tip_priors_observed::Matrix{Float64} = zeros(0, 0)
    tip_priors_hidden::Matrix{Float64} = zeros(0, 0)
    rates::Vector{Float64} = Float64[]
    index_matrix::Matrix{Int} = zeros(Int, 0, 0)
    solution::Matrix{Float64} = zeros(0, 0)
    transition_matrix::Matrix{Float64} = zeros(0, 0)
    root_prior_probs::Vector{Float64} = Float64[]
    branch_lengths::Vector{Float64} = Float64[]
    states::Matrix{Float64} = zeros(0, 0)
    tip_states::Matrix{Float64} = zeros(0, 0)
    fit::MkFitResult = MkFitResult()
    asr::Any = nothing
    diagnostics::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

Base.@kwdef struct _CorHMMOptimizationResult
    fit::MkFitResult = MkFitResult()
    tip_fog::Union{Nothing, Vector{Float64}} = nothing
end

Base.@kwdef struct CorHMMPruningCache
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    aicc::Float64 = NaN
    nstates::Int = 0
    nparams::Int = 0
    root_prior::Symbol = :yang
    node_liks::Matrix{Float64} = zeros(0, 0)
    transition_matrix::Matrix{Float64} = zeros(0, 0)
    comp::Vector{Float64} = Float64[]
    root_prior_probs::Vector{Float64} = Float64[]
end

Base.@kwdef mutable struct CorHMMPruningWorkspace
    node_liks::Matrix{Float64} = zeros(0, 0)
    P::Matrix{Float64} = zeros(0, 0)
    tmp::Vector{Float64} = Float64[]
    comp::Vector{Float64} = Float64[]
    root_prior_probs::Vector{Float64} = Float64[]
    exp_evals::Vector{ComplexF64} = ComplexF64[]
    stationary_A::Matrix{Float64} = zeros(0, 0)
    stationary_b::Vector{Float64} = Float64[]
end

Base.@kwdef struct CorHMMASRResult
    success::Bool = false
    mode::Symbol = :marginal
    node_ids::Vector{Int} = Int[]
    hidden_likelihoods::Matrix{Float64} = zeros(0, 0)
    observed_likelihoods::Matrix{Float64} = zeros(0, 0)
    tip_hidden_likelihoods::Matrix{Float64} = zeros(0, 0)
    tip_observed_likelihoods::Matrix{Float64} = zeros(0, 0)
    hidden_states::Vector{Int32} = Int32[]
    observed_states::Vector{Int32} = Int32[]
    tip_hidden_states::Vector{Int32} = Int32[]
    tip_observed_states::Vector{Int32} = Int32[]
    joint_loglik::Float64 = NaN
    hidden_state_labels::Vector{String} = String[]
    observed_state_labels::Vector{String} = String[]
    fit::Any = nothing
end

Base.@kwdef struct CorHMMSimmapResult
    success::Bool = false
    samples::Vector{SimmapSample} = SimmapSample[]
    collapsed_samples::Vector{SimmapSample} = SimmapSample[]
    observed_labels::Vector{String} = String[]
    hidden_labels::Vector{String} = String[]
    fit::Any = nothing
end

function Base.show(io::IO, res::CorHMMFitResult)
    print(
        io,
        "CorHMMFitResult(success=$(res.success), model=$(res.model), rate_cat=$(res.rate_cat), ",
        "observed_states=$(length(res.observed_labels)), hidden_states=$(length(res.hidden_labels)), ",
        "loglik=$(res.loglik), aic=$(res.aic))",
    )
end

Base.show(io::IO, ::MIME"text/plain", res::CorHMMFitResult) = show(io, res)

function Base.show(io::IO, res::CorHMMASRResult)
    print(
        io,
        "CorHMMASRResult(success=$(res.success), mode=$(res.mode), nodes=$(length(res.node_ids)), ",
        "observed_states=$(length(res.observed_state_labels)), hidden_states=$(length(res.hidden_state_labels)))",
    )
end

Base.show(io::IO, ::MIME"text/plain", res::CorHMMASRResult) = show(io, res)

function Base.show(io::IO, res::CorHMMSimmapResult)
    print(
        io,
        "CorHMMSimmapResult(success=$(res.success), nsim=$(length(res.samples)), ",
        "observed_states=$(length(res.observed_labels)), hidden_states=$(length(res.hidden_labels)))",
    )
end

Base.show(io::IO, ::MIME"text/plain", res::CorHMMSimmapResult) = show(io, res)

