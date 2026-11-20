function _ou_root_prior(spec::OUSpec, bundle::OUParameterBundle; cache = nothing)
    _validate_ou_theta0(spec.root_mean_mode, bundle.theta0)
    mean =
        if spec.root_mean_mode === :theta
            bundle.theta[1]
        elseif spec.root_mean_mode === :regime
            bundle.theta[cache === nothing ? 1 : cache.root_regime]
        elseif spec.root_mean_mode === :stationary_design
            0.0
        elseif spec.root_mean_mode === :stationary
            cache === nothing ? bundle.theta[1] : 0.0
        else
            throw(ArgumentError("Unsupported OU root_mean_mode=$(spec.root_mean_mode)"))
        end

    if spec.root_cov_mode === :random
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

function _ou_prepare_context(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec,
    cache,
)
    _validate_ou_root_mean_mode(spec.root_mean_mode)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    tip_index = zeros(Int, tree.nnodes)
    @inbounds for (i, node) in enumerate(tree.tip_ids)
        tip_index[node] = i
    end
    return OULikelihoodContext(tree, tr, spec, cache, tip_index)
end

function _ou_LL(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    spec::OUSpec,
    bundle::OUParameterBundle;
    cache = nothing,
)
    context = _ou_prepare_context(tree, trait, spec, cache)
    return _ou_context_LL(context, bundle, OULikelihoodWorkspace(tree))
end

# A whitened row is z - w*x. Inf marks an exact constraint x = z instead.
@inline function _ou_whitening_step(w, a, v)
    if w == Inf
        if v > 0.0
            sd = sqrt(v)
            scale = inv(sd)
            return (weight = a * scale, scale = scale, drift = scale, log_scale = log(sd))
        elseif a > 0.0
            scale = inv(a)
            return (weight = Inf, scale = scale, drift = scale, log_scale = log(a))
        end
        return (weight = NaN, scale = NaN, drift = NaN, log_scale = NaN)
    end
    d = hypot(1.0, sqrt(v) * w)
    drift = w / d
    return (weight = a * drift, scale = inv(d), drift = drift, log_scale = log(d))
end

# Collect squared orthogonal contrasts of the centered observations, without
# forming a covariance matrix or subtracting quadratic information constants.
function _ou_residual_loglik(tree, data, edges, means, center, root, work, tip_index)
    weights, rows = work.precision, work.linear
    rss, log_scale, nobs = 0.0, 0.0, 0
    failed = (loglik = -Inf, root_residual = NaN)
    @inbounds for node in tree.postorder_internal
        left_w, left_z = -1.0, 0.0
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = tree.child_of_edge[edge]
            a, v = edges.edge_a[edge], edges.edge_v[edge]
            isfinite(a) && a >= 0 && isfinite(v) && v >= 0 || return failed
            if tree.is_tip[child]
                y = data[tip_index[child]]
                isnan(y) && continue
                nobs += 1
                w0, z0 = Inf, (y - center) - means[child]
            else
                w0, z0 = weights[child], rows[child]
                w0 < 0 && continue
            end
            step = _ou_whitening_step(w0, a, v)
            w, z = step.weight, z0 * step.scale
            isnan(w) && return failed
            log_scale += step.log_scale
            if left_w < 0
                left_w, left_z = w, z
                continue
            end
            if left_w == Inf
                w == Inf && return failed # Redundant deterministic observations.
                rss += abs2(muladd(-w, left_z, z))
                continue
            elseif w == Inf
                rss += abs2(muladd(-left_w, z, left_z))
                left_w, left_z = w, z
                continue
            end
            weight = hypot(left_w, w)
            if weight == 0.0
                rss += abs2(z)
            else
                l, r = left_w / weight, w / weight
                rss += abs2(muladd(r, left_z, -l * z))
                left_w, left_z = weight, muladd(l, left_z, r * z)
            end
        end
        weights[node], rows[node] = left_w, left_z
    end
    w, z = weights[tree.root], rows[tree.root]
    nobs > 0 && w >= 0 || return failed
    if root.profile_root
        0 < w < Inf || return failed
        root_residual = z / w
    elseif isfinite(root.var) && root.var >= 0
        step = _ou_whitening_step(w, 0.0, root.var)
        isnan(step.weight) && return failed
        residual = z * step.scale
        rss += abs2(residual)
        log_scale += step.log_scale
        root_residual = w == Inf ? z : (root.var * step.drift) * residual
    else
        return failed
    end
    loglik = -(rss + nobs * log(2pi)) / 2 - log_scale
    return (; loglik, root_residual)
end

function _ou_mapped_loglik(context, bundle, work, root)
    tree, spec, cache = context.tree, context.spec, context.cache
    data = context.trait
    special_mean = spec.root_mean_mode in (:stationary_design, :stationary)
    center = special_mean ? 0.0 : data[findfirst(!isnan, data)]
    edges = _build_ou_edges(tree, spec, bundle; cache, workspace = work, theta_offset = center)
    if special_mean
        means = _ou_stationary_design_node_means(tree, spec, bundle, cache)
    else
        means = work.logconst
        means[tree.root] = root.mean - center
        @inbounds for node in tree.preorder
            tree.is_tip[node] && continue
            for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
                means[tree.child_of_edge[edge]] = muladd(edges.edge_a[edge], means[node], edges.edge_b[edge])
            end
        end
    end
    score = _ou_residual_loglik(tree, data, edges, means, center, root, work, context.tip_index)
    success = isfinite(score.loglik) && isfinite(score.root_residual)
    root_state = success ? (special_mean ? means[tree.root] : root.mean) + score.root_residual : NaN
    return (; success, loglik = success ? score.loglik : -Inf, root_state)
end

function _ou_context_LL(
    context::OULikelihoodContext,
    bundle::OUParameterBundle,
    workspace::OULikelihoodWorkspace,
)
    tree = context.tree
    root = _ou_root_prior(context.spec, bundle; cache = context.cache)
    if context.cache !== nothing
        return _ou_mapped_loglik(context, bundle, workspace, root)
    end
    edges = _build_ou_edges(
        tree,
        context.spec,
        bundle;
        cache = context.cache,
        workspace = workspace,
    )
    return _linear_gaussian_loglik(
        tree,
        context.trait,
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
end

function _ou_context_objective(
    context::OULikelihoodContext,
    workspace::OULikelihoodWorkspace,
)
    nregimes = context.cache === nothing ? 1 : context.cache.nregimes
    return function (par)
        bundle = _ou_unpack_params(context.spec, par, nregimes)
        prof = _ou_context_LL(context, bundle, workspace)
        return prof.success ? -prof.loglik : Inf
    end
end
