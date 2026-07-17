function _ebm_edge_variances(
    tree::CompactTree,
    cache::OUMEdgeSegmentCache,
    sigma2::AbstractVector{<:Real},
    beta_regimes::AbstractVector{<:Real},
)
    length(sigma2) == cache.nregimes || throw(ArgumentError("Number of sigma2 parameters must match number of regimes"))
    length(beta_regimes) == cache.nregimes || throw(ArgumentError("Number of beta parameters must match number of regimes"))
    any(x -> !isfinite(x) || x <= 0.0, sigma2) && return fill(NaN, tree.nedges)
    any(x -> !isfinite(x), beta_regimes) && return fill(NaN, tree.nedges)

    edge_v = zeros(Float64, tree.nedges)
    for edge in 1:tree.nedges
        parent = Int(tree.parent_of_edge[edge])
        seg_time = tree.dist_from_root[parent]
        first_seg = Int(cache.edge_first_segment[edge])
        last_seg = Int(cache.edge_last_segment[edge])
        @inbounds for seg_idx in first_seg:last_seg
            seg_length = cache.segment_lengths[seg_idx]
            seg_stop = seg_time + seg_length
            regime = Int(cache.segment_states[seg_idx])
            edge_v[edge] += _eb_branch_variance_increment(
                seg_time,
                seg_stop,
                Float64(sigma2[regime]),
                Float64(beta_regimes[regime]),
            )
            seg_time = seg_stop
        end
    end
    return edge_v
end

function _ebm_edge_variances(
    tree::CompactTree,
    edge_segments::Vector{Vector{SimmapSegment}},
    sigma2::AbstractVector{<:Real},
    beta_regimes::AbstractVector{<:Real},
)
    return _ebm_edge_variances(tree, _prepare_oum_edge_cache(tree, edge_segments), sigma2, beta_regimes)
end

function _ebm_profile(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache::OUMEdgeSegmentCache,
    sigma2::AbstractVector{<:Real},
    beta_regimes::AbstractVector{<:Real},
)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    edge_variances = _ebm_edge_variances(tree, cache, sigma2, beta_regimes)
    return _bm_pruning_profile_from_edge_variances(tree, tr, edge_variances)
end

"""
    ebm_loglikelihood(tree, trait, edge_segments, sigma2, beta_regimes)

Evaluate a multi-regime early-burst model with regime-specific diffusion rates
`sigma2` and regime-specific early-burst parameters `beta_regimes`.
"""
function ebm_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    sigma2::AbstractVector{<:Real},
    beta_regimes::AbstractVector{<:Real},
    ;
    trait_name = nothing,
    regime_names = nothing,
)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    prof = _ebm_profile(tree, trait, cache, sigma2, beta_regimes)
    nregimes = cache.nregimes
    return ContinuousMultiRegimeResult(
        model = :EBM,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, 2 * nregimes + 1) : Inf,
        nparams = 2 * nregimes + 1,
        trait_name = _continuous_checked_trait_name(trait_name),
        regime_names = _continuous_checked_regime_names(regime_names, nregimes),
        sigma2 = Float64.(sigma2),
        beta_regimes = Float64.(beta_regimes),
        theta = prof.root_state,
        root_state = prof.root_state,
        nregimes = nregimes,
    )
end

function ebm_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    sigma2::AbstractVector{<:Real},
    beta::Real,
    ;
    trait_name = nothing,
    regime_names = nothing,
)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    beta_regimes = fill(Float64(beta), cache.nregimes)
    prof = _ebm_profile(tree, trait, cache, sigma2, beta_regimes)
    return ContinuousMultiRegimeResult(
        model = :EBM,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, 2 * cache.nregimes + 1) : Inf,
        nparams = 2 * cache.nregimes + 1,
        trait_name = _continuous_checked_trait_name(trait_name),
        regime_names = _continuous_checked_regime_names(regime_names, cache.nregimes),
        sigma2 = Float64.(sigma2),
        beta_regimes = beta_regimes,
        theta = prof.root_state,
        root_state = prof.root_state,
        nregimes = cache.nregimes,
    )
end

"""
    fit_ebm(tree, trait, edge_segments; ...)

Fit the multi-regime early-burst model with regime-specific `sigma2` and
regime-specific `beta`.
"""
function fit_ebm(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}};
    max_iterations::Integer = 500,
    rel_tol::Float64 = 1e-5,
    low::Union{Nothing, Real} = nothing,
    up::Real = 0.0,
    trait_name = nothing,
    regime_names = nothing,
)
    tr = _validate_univariate_trait_allow_missing(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    cache = _prepare_oum_edge_cache(tree, edge_segments)
    nregimes = cache.nregimes
    tree_height = maximum(tree.dist_from_root)
    low_beta = low === nothing ? log(1e-5) / max(tree_height, 1e-8) : Float64(low)
    up_beta = Float64(up)
    low_beta < up_beta || throw(ArgumentError("EBM requires low < up"))

    sigma0 = fill(max(var(filter(!isnan, tr)), 1e-8), nregimes)
    beta_base = fill(clamp(-1.0 / max(tree_height, 1e-8), low_beta + 1e-8, up_beta - 1e-8), nregimes)
    candidates = Vector{Float64}[]
    for (sigma_scale, beta_scale) in ((1.0, 1.0), (0.25, 1.0), (4.0, 1.0), (1.0, 0.5), (1.0, 2.0))
        beta0 = clamp.(beta_base .* beta_scale, low_beta + 1e-8, up_beta - 1e-8)
        p0 = (beta0 .- low_beta) ./ (up_beta - low_beta)
        push!(candidates, vcat(log.(sigma0 .* sigma_scale), log.(p0 ./ (1.0 .- p0))))
    end

    objective = function (par)
        sigma2 = exp.(Float64.(par[1:nregimes]))
        beta_regimes = _eb_beta_from_raw.(Float64.(par[(nregimes + 1):(2 * nregimes)]), low_beta, up_beta)
        prof = _ebm_profile(tree, tr, cache, sigma2, beta_regimes)
        return prof.success ? -prof.loglik : Inf
    end

    result = _continuous_two_stage_multistart_serial(
        objective,
        candidates;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )

    fitted = _continuous_result_minimizer(result)
    sigma2 = exp.(fitted[1:nregimes])
    beta_regimes = _eb_beta_from_raw.(Float64.(fitted[(nregimes + 1):(2 * nregimes)]), low_beta, up_beta)
    prof = _ebm_profile(tree, tr, cache, sigma2, beta_regimes)
    return ContinuousMultiRegimeResult(
        model = :EBM,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, 2 * nregimes + 1) : Inf,
        nparams = 2 * nregimes + 1,
        trait_name = _continuous_checked_trait_name(trait_name),
        regime_names = _continuous_checked_regime_names(regime_names, nregimes),
        sigma2 = collect(sigma2),
        beta_regimes = collect(beta_regimes),
        theta = prof.root_state,
        root_state = prof.root_state,
        nregimes = nregimes,
        converged = _continuous_result_converged(result),
        iterations = _continuous_result_iterations(result),
        f_calls = _continuous_result_f_calls(result),
    )
end

function fit_ebm(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return fit_ebm(
        tree,
        trait,
        simmap.edge_segments;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function fit_ebm(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample;
    trait_name = nothing,
    kwargs...,
)
    return fit_ebm(
        tree,
        trait,
        simmap.edge_segments;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function ebm_loglikelihood(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample,
    sigma2::AbstractVector{<:Real},
    beta_regimes::AbstractVector{<:Real};
    trait_col = nothing,
    kwargs...,
)
    trait, trait_name = _continuous_align_dataframe_trait(tree, data; trait_col = trait_col)
    return ebm_loglikelihood(
        tree,
        trait,
        simmap.edge_segments,
        sigma2,
        beta_regimes;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end

function ebm_loglikelihood(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    simmap::SimmapSample,
    sigma2::AbstractVector{<:Real},
    beta_regimes::AbstractVector{<:Real};
    trait_name = nothing,
    kwargs...,
)
    return ebm_loglikelihood(
        tree,
        trait,
        simmap.edge_segments,
        sigma2,
        beta_regimes;
        trait_name = trait_name,
        regime_names = _continuous_regime_names_from_simmap(simmap, simmap.nstates),
        kwargs...,
    )
end
