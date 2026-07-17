"""
    ContinuousASRResult

Result type for continuous ancestral state reconstruction.
"""
Base.@kwdef struct ContinuousASRResult
    model::Symbol = :unknown
    success::Bool = false
    node_ids::Vector{Int32} = Int32[]
    node_labels::Vector{String} = String[]
    time_from_root::Vector{Float64} = Float64[]
    time_before_present::Vector{Float64} = Float64[]
    estimates::Vector{Float64} = Float64[]
    variances::Vector{Float64} = Float64[]
    se::Vector{Float64} = Float64[]
    all_node_estimates::Vector{Float64} = Float64[]
    all_node_variances::Vector{Float64} = Float64[]
    tip_ids::Vector{Int32} = Int32[]
    tip_labels::Vector{String} = String[]
    tip_estimates::Vector{Float64} = Float64[]
    tip_variances::Vector{Float64} = Float64[]
    tip_se::Vector{Float64} = Float64[]
    tip_observed::Vector{Bool} = Bool[]
end

"""
    ContinuousBranchPosteriorResult

Result type for simmap-aware branch posterior reconstruction.
"""
Base.@kwdef struct ContinuousBranchPosteriorResult
    model::Symbol = :unknown
    success::Bool = false
    edge_ids::Vector{Int32} = Int32[]
    parent_nodes::Vector{Int32} = Int32[]
    child_nodes::Vector{Int32} = Int32[]
    segment_indices::Vector{Int32} = Int32[]
    segment_states::Vector{Int32} = Int32[]
    branch_start::Vector{Float64} = Float64[]
    branch_end::Vector{Float64} = Float64[]
    branch_midpoint::Vector{Float64} = Float64[]
    time_from_root_start::Vector{Float64} = Float64[]
    time_from_root_end::Vector{Float64} = Float64[]
    time_from_root_midpoint::Vector{Float64} = Float64[]
    time_before_present_start::Vector{Float64} = Float64[]
    time_before_present_end::Vector{Float64} = Float64[]
    time_before_present_midpoint::Vector{Float64} = Float64[]
    absolute_start::Vector{Float64} = Float64[]
    absolute_end::Vector{Float64} = Float64[]
    absolute_midpoint::Vector{Float64} = Float64[]
    start_estimates::Vector{Float64} = Float64[]
    midpoint_estimates::Vector{Float64} = Float64[]
    end_estimates::Vector{Float64} = Float64[]
    start_variances::Vector{Float64} = Float64[]
    midpoint_variances::Vector{Float64} = Float64[]
    end_variances::Vector{Float64} = Float64[]
    start_se::Vector{Float64} = Float64[]
    midpoint_se::Vector{Float64} = Float64[]
    end_se::Vector{Float64} = Float64[]
end

"""
    MVContinuousASRResult

Result type for multivariate continuous ancestral state reconstruction.
"""
Base.@kwdef struct MVContinuousASRResult
    model::Symbol = :unknown
    success::Bool = false
    node_ids::Vector{Int32} = Int32[]
    node_labels::Vector{String} = String[]
    time_from_root::Vector{Float64} = Float64[]
    time_before_present::Vector{Float64} = Float64[]
    estimates::Matrix{Float64} = zeros(0, 0)
    node_covariances::Array{Float64, 3} = zeros(0, 0, 0)
    all_node_estimates::Matrix{Float64} = zeros(0, 0)
    all_node_covariances::Array{Float64, 3} = zeros(0, 0, 0)
    tip_ids::Vector{Int32} = Int32[]
    tip_labels::Vector{String} = String[]
    tip_estimates::Matrix{Float64} = zeros(0, 0)
    tip_covariances::Array{Float64, 3} = zeros(0, 0, 0)
    tip_observed_mask::Matrix{Bool} = zeros(Bool, 0, 0)
end

"""
    MVContinuousBranchPosteriorResult

Result type for multivariate simmap-aware branch posterior reconstruction.
"""
Base.@kwdef struct MVContinuousBranchPosteriorResult
    model::Symbol = :unknown
    success::Bool = false
    edge_ids::Vector{Int32} = Int32[]
    parent_nodes::Vector{Int32} = Int32[]
    child_nodes::Vector{Int32} = Int32[]
    segment_indices::Vector{Int32} = Int32[]
    segment_states::Vector{Int32} = Int32[]
    branch_start::Vector{Float64} = Float64[]
    branch_end::Vector{Float64} = Float64[]
    branch_midpoint::Vector{Float64} = Float64[]
    time_from_root_start::Vector{Float64} = Float64[]
    time_from_root_end::Vector{Float64} = Float64[]
    time_from_root_midpoint::Vector{Float64} = Float64[]
    time_before_present_start::Vector{Float64} = Float64[]
    time_before_present_end::Vector{Float64} = Float64[]
    time_before_present_midpoint::Vector{Float64} = Float64[]
    absolute_start::Vector{Float64} = Float64[]
    absolute_end::Vector{Float64} = Float64[]
    absolute_midpoint::Vector{Float64} = Float64[]
    start_estimates::Matrix{Float64} = zeros(0, 0)
    midpoint_estimates::Matrix{Float64} = zeros(0, 0)
    end_estimates::Matrix{Float64} = zeros(0, 0)
    start_covariances::Array{Float64, 3} = zeros(0, 0, 0)
    midpoint_covariances::Array{Float64, 3} = zeros(0, 0, 0)
    end_covariances::Array{Float64, 3} = zeros(0, 0, 0)
end
