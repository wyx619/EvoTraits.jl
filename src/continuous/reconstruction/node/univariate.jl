function estim_node(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    fit::ContinuousFitResult;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    fit.success || throw(ArgumentError("estim_node requires a successful fitted model"))
    if fit.model === :BM1
        edge_a = ones(Float64, tree.nedges)
        edge_b = zeros(Float64, tree.nedges)
        edge_v = fit.sigma2 .* tree.edge_length
        return _linear_gaussian_asr(tree, trait, edge_a, edge_b, edge_v; model = :BM1)
    elseif fit.model === :OU1
        spec = ou_spec(:OU1)
        _validate_ultrametric_tree(tree)
        bundle = OUParameterBundle(theta = [fit.theta], alpha = [fit.alpha], sigma2 = [fit.sigma2])
        edges = _build_ou_edges(tree, spec, bundle)
        root = _ou_root_prior(spec, bundle)
        return _linear_gaussian_asr(
            tree,
            trait,
            edges.edge_a,
            edges.edge_b,
            edges.edge_v;
            root_prior_mean = root.mean,
            root_prior_var = root.var,
            model = :OU1,
        )
    elseif fit.model === :EB
        _validate_ultrametric_tree(tree)
        edge_a = ones(Float64, tree.nedges)
        edge_b = zeros(Float64, tree.nedges)
        edge_v = similar(tree.edge_length)
        for edge in 1:tree.nedges
            parent = tree.parent_of_edge[edge]
            child = tree.child_of_edge[edge]
            edge_v[edge] = _eb_branch_variance_increment(tree.dist_from_root[parent], tree.dist_from_root[child], fit.sigma2, fit.beta)
        end
        return _linear_gaussian_asr(tree, trait, edge_a, edge_b, edge_v; model = :EB)
    end

    throw(ArgumentError("estim_node currently supports ContinuousFitResult models BM1, OU1, and EB; got $(fit.model)"))
end

function estim_node(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    fit::ContinuousMultiRegimeResult;
    mapped_edge::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
)
    fit.success || throw(ArgumentError("estim_node requires a successful fitted model"))
    if fit.model === :BMM
        mapped = _validate_mapped_edge(tree, _resolve_mapped_edge_context(tree, mapped_edge, edge_segments))
        edge_a = ones(Float64, tree.nedges)
        edge_b = zeros(Float64, tree.nedges)
        edge_v = _bmm_edge_variances(mapped, fit.sigma2)
        return _linear_gaussian_asr(tree, trait, edge_a, edge_b, edge_v; model = :BMM)
    elseif fit.model === :EBM
        edge_segments === nothing && throw(ArgumentError("EBM ancestral reconstruction requires edge_segments"))
        _validate_ultrametric_tree(tree)
        edge_a = ones(Float64, tree.nedges)
        edge_b = zeros(Float64, tree.nedges)
        edge_v = _ebm_edge_variances(tree, edge_segments, fit.sigma2, fit.beta_regimes)
        return _linear_gaussian_asr(tree, trait, edge_a, edge_b, edge_v; model = :EBM)
    elseif fit.model === :OUM || fit.model === :OUMV || fit.model === :OUMA || fit.model === :OUMVA
        spec = ou_spec(fit.model)
        edge_segments === nothing && throw(ArgumentError("$(fit.model) ancestral reconstruction requires edge_segments"))
        _validate_ultrametric_tree(tree)
        cache = _prepare_oum_edge_cache(tree, edge_segments)
        alpha_values = isempty(fit.alpha_regimes) ? [fit.alpha] : fit.alpha_regimes
        bundle = OUParameterBundle(theta = fit.theta_regimes, alpha = alpha_values, sigma2 = fit.sigma2)
        edges = _build_ou_edges(tree, spec, bundle; cache = cache)
        root = _ou_root_prior(spec, bundle; cache = cache)
        primary = _linear_gaussian_asr(
            tree,
            trait,
            edges.edge_a,
            edges.edge_b,
            edges.edge_v;
            root_prior_mean = root.mean,
            root_prior_var = root.var,
            model = fit.model,
        )
        primary.success && return primary
        return _linear_gaussian_asr(tree, trait, edges.edge_a, edges.edge_b, edges.edge_v; model = fit.model)
    end

    throw(ArgumentError("estim_node currently supports ContinuousMultiRegimeResult models BMM, EBM, OUM, OUMV, OUMA, and OUMVA; got $(fit.model)"))
end
