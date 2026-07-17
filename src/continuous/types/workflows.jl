"""
    ContinuousWorkflowResult

Bundled result for a model-comparison workflow.
"""
Base.@kwdef struct ContinuousWorkflowResult <: AbstractContinuousWorkflowResult
    fits::Dict{Symbol, AbstractContinuousFitResult} = Dict{Symbol, AbstractContinuousFitResult}()
    aic_table::Vector{NamedTuple} = NamedTuple[]
    best_name::Symbol = :unknown
    best_fit::Union{Nothing, AbstractContinuousFitResult} = nothing
    asr::Union{Nothing, ContinuousASRResult} = nothing
end

"""
    MVContinuousWorkflowResult

Bundled result for a multivariate model-comparison workflow.
"""
Base.@kwdef struct MVContinuousWorkflowResult <: AbstractMVContinuousWorkflowResult
    fits::Dict{Symbol, AbstractMVContinuousFitResult} = Dict{Symbol, AbstractMVContinuousFitResult}()
    aic_table::Vector{NamedTuple} = NamedTuple[]
    best_name::Symbol = :unknown
    best_fit::Union{Nothing, AbstractMVContinuousFitResult} = nothing
    asr::Union{Nothing, MVContinuousASRResult} = nothing
end
