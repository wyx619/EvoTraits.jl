function _ou_root_prior(spec::OUSpec, bundle::OUParameterBundle; cache = nothing)
    mean =
        if spec.root_mean_mode === :theta
            bundle.theta[1]
        elseif spec.root_mean_mode === :root_regime_theta
            cache === nothing && throw(ArgumentError("cache is required for root_regime_theta"))
            bundle.theta[cache.root_regime]
        elseif spec.root_mean_mode === :free_theta0
            bundle.theta0 === nothing && throw(ArgumentError("theta0 is required for free_theta0"))
            bundle.theta0
        elseif spec.root_mean_mode === :stationary_design
            0.0
        else
            throw(ArgumentError("Unsupported OU root_mean_mode=$(spec.root_mean_mode)"))
        end

    if spec.root_cov_mode === :stationary
        root_state = cache === nothing ? 1 : cache.root_regime
        alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, root_state)
        sigma2 = _ou_regime_value(spec.sigma_mode, bundle.sigma2, root_state)
        return (mean = mean, var = sigma2 / (2.0 * alpha), profile_root = false)
    elseif spec.root_cov_mode === :fixed
        return (mean = mean, var = 0.0, profile_root = false)
    elseif spec.root_cov_mode === :nonstationary
        return (mean = mean, var = Inf, profile_root = true)
    elseif spec.root_cov_mode === :free
        return (mean = mean, var = Inf, profile_root = true)
    end
    throw(ArgumentError("Unsupported OU root_cov_mode=$(spec.root_cov_mode)"))
end

function _ou_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec,
    bundle::OUParameterBundle;
    cache = nothing,
)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    edges = _build_ou_edges(tree, spec, bundle; cache = cache)
    root = _ou_root_prior(spec, bundle; cache = cache)
    data = trait
    if spec.root_mean_mode === :stationary_design
        cache === nothing && throw(ArgumentError("cache is required for stationary_design"))
        node_means = _ou_stationary_design_node_means(tree, spec, bundle, cache)
        data = copy(Float64.(trait))
        for (i, tip) in enumerate(tree.tip_ids)
            data[i] -= node_means[Int(tip)]
        end
        fill!(edges.edge_b, 0.0)
    end
    prof = _linear_gaussian_loglik(
        tree,
        data,
        edges.edge_a,
        edges.edge_b,
        edges.edge_v;
        root_prior_mean = root.mean,
        root_prior_var = root.var,
        profile_root = root.profile_root,
    )
    return (
        success = prof.success,
        loglik = prof.loglik,
        root_state = prof.root_state,
        edges = edges,
        root = root,
    )
end

function _ou_prepare_context(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec,
    cache,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    tip_index = zeros(Int, tree.nnodes)
    @inbounds for (i, node) in enumerate(tree.tip_ids)
        tip_index[node] = i
    end
    return OULikelihoodContext(tree, tr, spec, cache, tip_index)
end

function _ou_context_loglikelihood(
    context::OULikelihoodContext,
    bundle::OUParameterBundle,
    workspace::OULikelihoodWorkspace,
)
    tree = context.tree
    edges = _build_ou_edges(
        tree,
        context.spec,
        bundle;
        cache = context.cache,
        workspace = workspace,
    )
    root = _ou_root_prior(context.spec, bundle; cache = context.cache)
    data = context.trait
    if context.spec.root_mean_mode === :stationary_design
        context.cache === nothing && throw(ArgumentError("cache is required for stationary_design"))
        node_means = _ou_stationary_design_node_means(tree, context.spec, bundle, context.cache)
        data = copy(context.trait)
        for (i, tip) in enumerate(tree.tip_ids)
            data[i] -= node_means[Int(tip)]
        end
        fill!(edges.edge_b, 0.0)
    end
    prof = _linear_gaussian_loglik(
        tree,
        data,
        edges.edge_a,
        edges.edge_b,
        edges.edge_v;
        root_prior_mean = root.mean,
        root_prior_var = root.var,
        profile_root = root.profile_root,
        workspace = workspace,
        tip_index = context.tip_index,
        validate = false,
    )
    return (
        success = prof.success,
        loglik = prof.loglik,
        root_state = prof.root_state,
        edges = edges,
        root = root,
    )
end

function _ou_context_objective(
    context::OULikelihoodContext,
    workspace::OULikelihoodWorkspace,
)
    nregimes = context.cache === nothing ? 1 : context.cache.nregimes
    return function (par)
        bundle = _ou_unpack_params(context.spec, par, nregimes)
        prof = _ou_context_loglikelihood(context, bundle, workspace)
        return prof.success ? -prof.loglik : Inf
    end
end
