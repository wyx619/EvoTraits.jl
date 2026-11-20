@inline function _ou_posterior_product(left, right)
    if isfinite(left.var) && isfinite(right.var) && left.var > 0.0 && right.var > 0.0
        small, large = left.var <= right.var ? (left, right) : (right, left)
        ratio = small.var / large.var
        fraction = ratio / (1.0 + ratio)
        return (mean = muladd(fraction, large.mean, (1.0 - fraction) * small.mean),
            var = small.var / (1.0 + ratio))
    end
    return _gaussian_product(left.mean, left.var, right.mean, right.var)
end

@inline function _ou_posterior_to_parent(child, a, b, v)
    isfinite(v) && v >= 0.0 || return (mean = NaN, var = NaN)
    isinf(child.var) && return (mean = 0.0, var = Inf)
    if a == 0.0
        return child.var == 0.0 && v == 0.0 ? (mean = NaN, var = NaN) : (mean = 0.0, var = Inf)
    end
    # Divide the standard deviation first, so very small a^2 need not be represented.
    var = abs2(hypot(sqrt(child.var), sqrt(v)) / a)
    isinf(var) && return (mean = 0.0, var = Inf)
    return (mean = (child.mean - b) / a, var = var)
end

function _ou_posterior_cache(tree, trait, a, b, v;
    root_prior_mean::Float64, root_prior_var::Float64)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    failed = (success = false,)
    isfinite(root_prior_mean) && root_prior_var >= 0.0 || return failed
    no_data = (mean = 0.0, var = Inf)
    desc = fill(no_data, tree.nnodes)
    up = fill(no_data, tree.nedges)
    for (i, tip) in enumerate(tree.tip_ids)
        isnan(tr[i]) || (desc[tip] = (mean = tr[i], var = 0.0))
    end
    @inbounds for node in tree.postorder_internal
        lo, hi = tree.first_child_edge[node], tree.last_child_edge[node]
        for e in lo:hi
            isfinite(a[e]) && a[e] >= 0 && isfinite(b[e]) && isfinite(v[e]) && v[e] >= 0 ||
                return failed
            up[e] = _ou_posterior_to_parent(desc[tree.child_of_edge[e]], a[e], b[e], v[e])
            isfinite(up[e].mean) || return failed
        end
        # Two exact observations of the same latent scalar give a singular joint density.
        up[lo].var == 0.0 && up[hi].var == 0.0 && return failed
        desc[node] = _ou_posterior_product(up[lo], up[hi])
    end

    root = Int(tree.root)
    (root_prior_var == 0.0 || isinf(root_prior_var)) && desc[root].var == 0.0 && return failed
    outside = fill(no_data, tree.nnodes)
    outside[root] = (mean = root_prior_mean, var = root_prior_var)
    full_mean, full_var = zeros(tree.nnodes), zeros(tree.nnodes)
    edge_context_mean, edge_context_var = zeros(tree.nedges), zeros(tree.nedges)
    @inbounds for node in tree.preorder
        posterior = _ou_posterior_product(desc[node], outside[node])
        isfinite(posterior.mean) && isfinite(posterior.var) || return failed
        full_mean[node], full_var[node] = posterior.mean, posterior.var
        tree.is_tip[node] && continue
        lo, hi = tree.first_child_edge[node], tree.last_child_edge[node]
        for (e, sibling) in ((lo, hi), (hi, lo))
            parent = _ou_posterior_product(outside[node], up[sibling])
            edge_context_mean[e], edge_context_var[e] = parent.mean, parent.var
            outside[tree.child_of_edge[e]] = _edge_predict_to_child(parent.mean, parent.var, a[e], b[e], v[e])
        end
    end
    return (; success = true, desc, full_mean, full_var, edge_context_mean, edge_context_var)
end
