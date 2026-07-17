Base.@kwdef struct WorkflowModelSpec
    fit_fn::Function
    context_kind::Symbol = :none
end

@inline _workflow_fit_kwargs(max_iterations::Integer, rel_tol::Float64) = (; max_iterations = max_iterations, rel_tol = rel_tol)
@inline _workflow_error_prefix(multivariate::Bool) = multivariate ? "multivariate continuous model" : "continuous model"

@inline _workflow_basic_spec(f::Function) = WorkflowModelSpec(fit_fn = f, context_kind = :none)
@inline _workflow_edge_segments_spec(f::Function) = WorkflowModelSpec(fit_fn = f, context_kind = :edge_segments)
@inline _workflow_regime_map_spec(f::Function) = WorkflowModelSpec(fit_fn = f, context_kind = :regime_map)

function _edge_segments_to_mapped_edge(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}})
    length(edge_segments) == tree.nedges || throw(ArgumentError("edge_segments must have $(tree.nedges) entries"))
    nregimes = maximum(seg.state for segments in edge_segments for seg in segments)
    mapped = zeros(Float64, tree.nedges, nregimes)
    for edge in 1:tree.nedges
        segments = edge_segments[edge]
        isempty(segments) && throw(ArgumentError("edge_segments[$edge] is empty"))
        total = 0.0
        for seg in segments
            seg.state >= 1 || throw(ArgumentError("edge_segments[$edge] contains invalid regime state $(seg.state)"))
            seg.length >= 0.0 || throw(ArgumentError("edge_segments[$edge] contains a negative segment length"))
            mapped[edge, Int(seg.state)] += seg.length
            total += seg.length
        end
        isapprox(total, tree.edge_length[edge]; atol = 1e-8) || throw(ArgumentError("edge_segments[$edge] lengths do not sum to branch length"))
    end
    return mapped
end

function _resolve_mapped_edge_context(tree::CompactTree, mapped_edge, edge_segments)
    edge_segments !== nothing && return _edge_segments_to_mapped_edge(tree, edge_segments)
    mapped_edge !== nothing && return mapped_edge
    throw(ArgumentError("regime-aware BM fitting requires edge_segments"))
end

function _workflow_build_registry(multivariate::Bool)
    if multivariate
        return Dict{Symbol, WorkflowModelSpec}(
            :mvBM1 => _workflow_basic_spec((tree, trait; kwargs...) -> fit_mvbm1(tree, trait; kwargs...)),
            :mvBMM => _workflow_regime_map_spec((tree, trait; mapped_edge = nothing, edge_segments = nothing, kwargs...) ->
                fit_mvbmm(tree, trait, _resolve_mapped_edge_context(tree, mapped_edge, edge_segments); kwargs...)),
            :mvEB => _workflow_basic_spec((tree, trait; kwargs...) -> fit_mveb(tree, trait; kwargs...)),
            :mvOU1 => _workflow_basic_spec((tree, trait; kwargs...) -> fit_mvou1(tree, trait; kwargs...)),
            :mvOUM => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_mvoum(tree, trait, edge_segments; kwargs...)),
            :mvOUMV => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_mvoumv(tree, trait, edge_segments; kwargs...)),
            :mvOUMA => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_mvouma(tree, trait, edge_segments; kwargs...)),
            :mvOUMVA => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_mvoumva(tree, trait, edge_segments; kwargs...)),
        )
    end

    return Dict{Symbol, WorkflowModelSpec}(
        :BM1 => _workflow_basic_spec((tree, trait; kwargs...) -> fit_bm1(tree, trait; kwargs...)),
        :OU1 => _workflow_basic_spec((tree, trait; kwargs...) -> fit_ou1(tree, trait; kwargs...)),
        :EB => _workflow_basic_spec((tree, trait; kwargs...) -> fit_eb(tree, trait; kwargs...)),
        :BMM => _workflow_regime_map_spec((tree, trait; mapped_edge = nothing, edge_segments = nothing, kwargs...) ->
            fit_bmm(tree, trait, _resolve_mapped_edge_context(tree, mapped_edge, edge_segments); kwargs...)),
        :EBM => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_ebm(tree, trait, edge_segments; kwargs...)),
        :OUM => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_oum(tree, trait, edge_segments; kwargs...)),
        :OUMV => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_oumv(tree, trait, edge_segments; kwargs...)),
        :OUMA => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_ouma(tree, trait, edge_segments; kwargs...)),
        :OUMVA => _workflow_edge_segments_spec((tree, trait; edge_segments, kwargs...) -> fit_oumva(tree, trait, edge_segments; kwargs...)),
    )
end

const CONTINUOUS_WORKFLOW_REGISTRY = _workflow_build_registry(false)
const MV_CONTINUOUS_WORKFLOW_REGISTRY = _workflow_build_registry(true)

_continuous_model_registry() = CONTINUOUS_WORKFLOW_REGISTRY
_multivariate_model_registry() = MV_CONTINUOUS_WORKFLOW_REGISTRY
@inline _workflow_registry(multivariate::Bool) = multivariate ? MV_CONTINUOUS_WORKFLOW_REGISTRY : CONTINUOUS_WORKFLOW_REGISTRY

function _lookup_workflow_spec(model_name::Symbol, multivariate::Bool)
    registry = _workflow_registry(multivariate)
    haskey(registry, model_name) || throw(ArgumentError("Unsupported $(_workflow_error_prefix(multivariate)) $model_name"))
    return registry[model_name]
end

function _validate_workflow_context(model_name::Symbol, context_kind::Symbol, mapped_edge, edge_segments)
    if context_kind === :mapped_edge
        mapped_edge === nothing && throw(ArgumentError("$model_name fitting requires mapped_edge"))
    elseif context_kind === :edge_segments
        edge_segments === nothing && throw(ArgumentError("$model_name fitting requires edge_segments"))
    elseif context_kind === :regime_map
        edge_segments === nothing && mapped_edge === nothing && throw(ArgumentError("$model_name fitting requires edge_segments"))
    end
end

function _workflow_dispatch_fit(
    spec::WorkflowModelSpec,
    tree::CompactTree,
    trait,
    mapped_edge,
    edge_segments,
    max_iterations::Integer,
    rel_tol::Float64,
)
    kwargs = _workflow_fit_kwargs(max_iterations, rel_tol)
    if spec.context_kind === :mapped_edge
        return spec.fit_fn(tree, trait; mapped_edge = mapped_edge, kwargs...)
    elseif spec.context_kind === :edge_segments
        return spec.fit_fn(tree, trait; edge_segments = edge_segments, kwargs...)
    elseif spec.context_kind === :regime_map
        return spec.fit_fn(tree, trait; mapped_edge = mapped_edge, edge_segments = edge_segments, kwargs...)
    end
    return spec.fit_fn(tree, trait; kwargs...)
end

function _workflow_estim(tree::CompactTree, trait, fit, context_kind::Symbol, mapped_edge, edge_segments)
    if context_kind === :mapped_edge
        return estim_node(tree, trait, fit; mapped_edge = mapped_edge)
    elseif context_kind === :edge_segments
        return estim_node(tree, trait, fit; edge_segments = edge_segments)
    elseif context_kind === :regime_map
        return estim_node(tree, trait, fit; mapped_edge = mapped_edge, edge_segments = edge_segments)
    end
    return estim_node(tree, trait, fit)
end

function _fit_workflow_models(
    tree::CompactTree,
    trait,
    models::AbstractVector{<:Symbol},
    multivariate::Bool;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    isempty(models) && throw(ArgumentError("models must not be empty"))
    length(unique(models)) == length(models) || throw(ArgumentError("models contains duplicates"))

    fits =
        multivariate ?
        Dict{Symbol, AbstractMVContinuousFitResult}() :
        Dict{Symbol, AbstractContinuousFitResult}()

    for model_name in models
        spec = _lookup_workflow_spec(model_name, multivariate)
        _validate_workflow_context(model_name, spec.context_kind, mapped_edge, edge_segments)
        fits[model_name] = _workflow_dispatch_fit(spec, tree, trait, mapped_edge, edge_segments, max_iterations, rel_tol)
    end
    return fits
end

function _fit_compare_workflow(
    tree::CompactTree,
    trait,
    models::AbstractVector{<:Symbol},
    multivariate::Bool;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    fits = _fit_workflow_models(
        tree,
        trait,
        models,
        multivariate;
        mapped_edge = mapped_edge,
        edge_segments = edge_segments,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )
    rows = aic_table((name => fits[name] for name in models)...)
    best = best_model(rows)
    best_fit = fits[best.name]
    best_spec = _lookup_workflow_spec(best.name, multivariate)
    asr = _workflow_estim(tree, trait, best_fit, best_spec.context_kind, mapped_edge, edge_segments)

    if multivariate
        return MVContinuousWorkflowResult(
            fits = fits,
            aic_table = rows,
            best_name = best.name,
            best_fit = best_fit,
            asr = asr,
        )
    end

    return ContinuousWorkflowResult(
        fits = fits,
        aic_table = rows,
        best_name = best.name,
        best_fit = best_fit,
        asr = asr,
    )
end

"""
    fit_compare_estim(tree, trait, models; kwargs...)

Run the full continuous-model comparison workflow: fit all requested models,
assemble the ranked AIC table, choose the best successful model, and perform
continuous ancestral state reconstruction for that fit.
"""
function fit_compare_estim(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    models::AbstractVector{<:Symbol};
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    return _fit_compare_workflow(
        tree,
        trait,
        models,
        false;
        mapped_edge = mapped_edge,
        edge_segments = edge_segments,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )
end

"""
    fit_compare_multivariate(tree, trait, models; kwargs...)

Run the full multivariate model-comparison workflow: fit all requested models,
assemble the ranked AIC table, choose the best successful model, and perform
multivariate ancestral state reconstruction for that fit.
"""
function fit_compare_multivariate(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    models::AbstractVector{<:Symbol};
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
)
    return _fit_compare_workflow(
        tree,
        trait,
        models,
        true;
        mapped_edge = mapped_edge,
        edge_segments = edge_segments,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )
end
