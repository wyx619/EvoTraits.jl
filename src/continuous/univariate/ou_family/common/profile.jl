struct OUMeanWorkspace
    likelihood::OULikelihoodWorkspace
    edge_design::Matrix{Float64}
    node_rows::Matrix{Float64}
    weights::Vector{Float64}
    active::BitVector
    design::Matrix{Float64}
    response::Vector{Float64}
end

function OUMeanWorkspace(context::OULikelihoodContext)
    k = context.spec.theta_mode === :shared ? 1 : context.cache.nregimes
    tree = context.tree
    return OUMeanWorkspace(
        OULikelihoodWorkspace(tree), zeros(k, tree.nedges), zeros(k + 1, tree.nnodes),
        zeros(tree.nnodes), falses(tree.nnodes), zeros(tree.ntips, k), zeros(tree.ntips),
    )
end

function _ou_edge_mean_design!(context::OULikelihoodContext, bundle::OUParameterBundle, design::Matrix{Float64})
    cache, spec = context.cache, context.spec
    fill!(design, 0.0)
    @inbounds for edge in 1:context.tree.nedges
        for seg in cache.edge_first_segment[edge]:cache.edge_last_segment[edge]
            state = Int(cache.segment_states[seg])
            alpha = _ou_regime_value(spec.alpha_mode, bundle.alpha, state)
            logphi = -alpha * cache.segment_lengths[seg]
            phi = exp(logphi)
            for j in axes(design, 1)
                design[j, edge] *= phi
            end
            j = spec.theta_mode === :shared ? 1 : state
            design[j, edge] -= expm1(logphi)
        end
    end
    if spec.root_mean_mode === :stationary
        tree = context.tree
        rows = zeros(size(design, 1), tree.nnodes)
        for parent in tree.preorder
            tree.is_tip[parent] && continue
            for edge in tree.first_child_edge[parent]:tree.last_child_edge[parent]
                child = tree.child_of_edge[edge]
                phi = exp(-bundle.alpha[1] * tree.edge_length[edge])
                @views rows[:, child] .= phi .* rows[:, parent] .+ design[:, edge]
            end
        end
        for node in 1:tree.nnodes
            duration = tree.dist_from_root[node]
            if duration == 0.0
                rows[cache.root_regime, node] = 1.0
            else
                @views rows[:, node] ./= -expm1(-bundle.alpha[1] * duration)
            end
        end
        for edge in 1:tree.nedges
            phi = exp(-bundle.alpha[1] * tree.edge_length[edge])
            @views design[:, edge] .= rows[:, tree.child_of_edge[edge]] .-
                phi .* rows[:, tree.parent_of_edge[edge]]
        end
    end
    return design
end

# Square-root elimination of scalar node states whitens y and its mean design
# together. No division by the root-to-tip attenuation or n-by-n covariance.
function _ou_whiten_mean_design!(context::OULikelihoodContext, bundle::OUParameterBundle, work::OUMeanWorkspace)
    tree, spec = context.tree, context.spec
    edges = _build_ou_edges(tree, spec, bundle; cache = context.cache, workspace = work.likelihood)
    _ou_edge_mean_design!(context, bundle, work.edge_design)
    rows, weights, active = work.node_rows, work.weights, work.active
    fill!(active, false)
    nrows, k = 0, size(work.design, 2)
    @inbounds for node in tree.postorder_internal
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = Int(tree.child_of_edge[edge])
            v = max(edges.edge_v[edge], 1e-12)
            if tree.is_tip[child]
                y = context.trait[context.tip_index[child]]
                isnan(y) && continue
                scale = inv(sqrt(v))
                weights[child] = edges.edge_a[edge] * scale
                rows[1, child] = y * scale
                for j in 1:k
                    rows[j + 1, child] = work.edge_design[j, edge] * scale
                end
            else
                active[child] || continue
                denominator = hypot(1.0, sqrt(v) * weights[child])
                scale = weights[child] / denominator
                rows[1, child] /= denominator
                for j in 1:k
                    rows[j + 1, child] = rows[j + 1, child] / denominator +
                        scale * work.edge_design[j, edge]
                end
                weights[child] = edges.edge_a[edge] * scale
            end
            if !active[node]
                for j in 1:(k + 1)
                    rows[j, node] = rows[j, child]
                end
                weights[node] = weights[child]
                active[node] = true
                continue
            end
            nrows += 1
            weight = hypot(weights[node], weights[child])
            if weight == 0.0
                work.response[nrows] = rows[1, child]
                for j in 1:k
                    work.design[nrows, j] = rows[j + 1, child]
                end
                continue
            end
            left, right = weights[node] / weight, weights[child] / weight
            work.response[nrows] = right * rows[1, node] - left * rows[1, child]
            rows[1, node] = left * rows[1, node] + right * rows[1, child]
            for j in 1:k
                work.design[nrows, j] = right * rows[j + 1, node] - left * rows[j + 1, child]
                rows[j + 1, node] = left * rows[j + 1, node] + right * rows[j + 1, child]
            end
            weights[node] = weight
        end
    end
    root = Int(tree.root)
    prior = _ou_root_prior(spec, bundle; cache = context.cache)
    if active[root] && !prior.profile_root
        nrows += 1
        variance = prior.var <= 1e-12 ? 0.0 : prior.var
        denominator = hypot(1.0, sqrt(variance) * weights[root])
        work.response[nrows] = rows[1, root] / denominator
        for j in 1:k
            work.design[nrows, j] = rows[j + 1, root] / denominator
        end
        root_column = spec.root_mean_mode in (:regime, :stationary) ? context.cache.root_regime :
                      spec.root_mean_mode === :theta ? 1 : 0
        if root_column > 0
            work.design[nrows, root_column] += weights[root] / denominator
        end
    end
    return nrows
end

function _ou_profile_parameters(context::OULikelihoodContext, parameters::AbstractVector{<:Real}, work::OUMeanWorkspace;
    diagnostics::Bool = false)
    spec, nregimes = context.spec, context.cache.nregimes
    na = spec.alpha_mode === :shared ? 1 : nregimes
    ns = spec.sigma_mode === :shared ? 1 : nregimes
    nt = spec.theta_mode === :shared ? 1 : nregimes
    length(parameters) == na + ns || throw(ArgumentError("OU covariance parameter count mismatch"))
    alpha, sigma2 = Float64.(parameters[1:na]), Float64.(parameters[(na + 1):end])
    bundle = OUParameterBundle(theta = zeros(nt), alpha = alpha, sigma2 = sigma2)
    failed = (bundle = bundle, profile = (success = false, loglik = -Inf, root_state = NaN))
    all(x -> isfinite(x) && x >= 1e-8, parameters) || return failed
    nrows = _ou_whiten_mean_design!(context, bundle, work)
    design, response = view(work.design, 1:nrows, :), view(work.response, 1:nrows)
    all(isfinite, design) && all(isfinite, response) || return failed
    # Reduce the tall whitened system in place; only the small R factor needs
    # an SVD. This preserves singular values without forming normal equations.
    if nrows == 0
        decomposition = svd(zeros(0, size(design, 2)); full = false)
    else
        factor = qr!(design)
        lmul!(adjoint(factor.Q), response)
        decomposition = svd(Matrix(factor.R); full = false)
    end
    spectrum = decomposition.S
    threshold = isempty(spectrum) ? 0.0 : sqrt(eps(Float64)) * spectrum[1]
    rank = count(>(threshold), spectrum)
    coefficients = zeros(size(design, 2))
    for j in 1:rank
        projection = dot(view(decomposition.U, :, j), view(response, 1:size(decomposition.U, 1))) / spectrum[j]
        @inbounds for i in eachindex(coefficients)
            coefficients[i] += decomposition.Vt[j, i] * projection
        end
    end
    bundle = OUParameterBundle(theta = coefficients, alpha = alpha, sigma2 = sigma2)
    profile = _ou_context_LL(context, bundle, work.likelihood)
    if diagnostics
        identifiable = [sum(abs2, view(decomposition.Vt, 1:rank, j)) >= 1.0 - 1e-8
            for j in eachindex(coefficients)]
        info = OUMeanDiagnostics(rank = rank, ncoefficients = length(coefficients),
            singular_values = copy(spectrum), identifiable = identifiable,
            condition_number = rank == length(coefficients) ? spectrum[1] / spectrum[end] : Inf)
        profile = merge(profile, (mean_diagnostics = info,))
    end
    return (bundle = bundle, profile = profile)
end

function _ou_profile_objective(context::OULikelihoodContext, work::OUMeanWorkspace)
    return function (parameters)
        evaluation = _ou_profile_parameters(context, parameters, work)
        return evaluation.profile.success ? -evaluation.profile.loglik : Inf
    end
end

function _ou_profiled_fit(context::OULikelihoodContext, init::Vector{Float64};
    max_iterations::Integer, rel_tol::Float64, method::Symbol = :TWO_STAGE)
    spec, tree, cache = context.spec, context.tree, context.cache
    ncovariance = (spec.alpha_mode === :shared ? 1 : cache.nregimes) +
                  (spec.sigma_mode === :shared ? 1 : cache.nregimes)
    objective = _ou_profile_objective(context, OUMeanWorkspace(context))
    lower = fill(1e-8, ncovariance)
    if method in (:SBPLX_L_BFGS, :LN_SBPLX_L_BFGS, :TWO_STAGE)
        full_candidates = _ou_multistart_candidates(spec, init, cache.nregimes, tree, context.trait, cache)
        candidates = unique([p[1:ncovariance] for p in full_candidates])
        factory = _ -> _ou_profile_objective(context, OUMeanWorkspace(context))
        result = _continuous_two_stage_multistart(objective, candidates;
            max_iterations = max_iterations, rel_tol = rel_tol,
            lower_bounds = lower, objective_factory = factory)
    else
        result = _continuous_optimize_objective(objective, init[1:ncovariance];
            method = method, max_iterations = max_iterations, rel_tol = rel_tol, lower_bounds = lower)
    end
    evaluation = _ou_profile_parameters(context, _continuous_result_minimizer(result), OUMeanWorkspace(context);
        diagnostics = true)
    info = hasproperty(evaluation.profile, :mean_diagnostics) ? evaluation.profile.mean_diagnostics : nothing
    if info !== nothing && info.rank < info.ncoefficients
        @warn "OU mean design is rank deficient; minimum-norm GLS coefficients are not all individually identifiable" model=spec.model rank=info.rank ncoefficients=info.ncoefficients identifiable=info.identifiable
    end
    return (bundle = evaluation.bundle, profile = evaluation.profile, result = result, nregimes = cache.nregimes)
end
