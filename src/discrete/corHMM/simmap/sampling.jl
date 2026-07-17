function _corhmm_conditional_node_likelihoods(fit::CorHMMFitResult)
    cache = corhmm_pruning_cache(
        fit.tree,
        fit.tip_priors_hidden,
        fit.transition_matrix;
        root_prior = _corhmm_root_prior(fit.root_prior),
        nparams = fit.nparams,
        rate_cat = fit.rate_cat,
        order_test = get(fit.diagnostics, :order_test, false),
        branch_lengths = isempty(fit.branch_lengths) ? _corhmm_branch_lengths(fit.tree) : fit.branch_lengths,
    )
    cache.success || return cache
    root = fit.tree.root
    @views cache.node_liks[root, :] .*= cache.root_prior_probs
    return cache
end

function _corhmm_simmap_sample_from_conditionals(
    fit::CorHMMFitResult,
    cache::CorHMMPruningCache,
    evals,
    V,
    Vinv;
    rng::AbstractRNG,
    max_attempt::Integer,
)
    tree = fit.tree
    branch_lengths = isempty(fit.branch_lengths) ? _corhmm_branch_lengths(tree) : fit.branch_lengths
    endpoints = sample_conditioned_endpoints(
        tree,
        cache.node_liks,
        fit.transition_matrix;
        root_prior_probs = cache.root_prior_probs,
        branch_lengths = branch_lengths,
        loglik = cache.loglik,
        rng = rng,
        evals,
        V,
        Vinv,
    )
    endpoints.success || return SimmapSample(success = false, nstates = endpoints.nstates, state_labels = fit.hidden_labels, loglik = endpoints.loglik)
    return _simmap_sample_from_endpoints(
        tree,
        fit.transition_matrix,
        endpoints;
        state_labels = fit.hidden_labels,
        branch_lengths = branch_lengths,
        rng = rng,
        max_attempt = max_attempt,
    )
end

function simmap_corhmm(
    fit::CorHMMFitResult;
    nsim::Integer = 100,
    rng::AbstractRNG = Random.default_rng(),
    fix_node = nothing,
    fix_state = nothing,
    parsimony::Bool = false,
    max_attempt::Integer = 100000,
)
    fix_node === nothing || throw(ArgumentError("corHMM SIMMAP fix_node is not supported in EvoTraits"))
    fix_state === nothing || throw(ArgumentError("corHMM SIMMAP fix_state is not supported in EvoTraits"))
    parsimony == false || throw(ArgumentError("corHMM SIMMAP parsimony mode is not supported in EvoTraits"))
    max_attempt >= 1 || throw(ArgumentError("max_attempt must be positive"))
    fit.success || return CorHMMSimmapResult(success = false, fit = fit)
    fit.tree === nothing && throw(ArgumentError("CorHMM fit does not store its tree"))

    cache = _corhmm_conditional_node_likelihoods(fit)
    cache.success || return CorHMMSimmapResult(success = false, fit = fit)
    evals, V, Vinv = _mk_eigen_cache(cache.transition_matrix)
    samples = Vector{SimmapSample}(undef, nsim)
    for i in 1:nsim
        samples[i] = _corhmm_simmap_sample_from_conditionals(fit, cache, evals, V, Vinv; rng = rng, max_attempt = max_attempt)
    end
    collapsed = [remap_simmap_states(sample, fit.hidden_to_observed, fit.observed_labels) for sample in samples]
    return CorHMMSimmapResult(
        success = all(sample.success for sample in samples),
        samples = samples,
        collapsed_samples = collapsed,
        observed_labels = fit.observed_labels,
        hidden_labels = fit.hidden_labels,
        fit = fit,
    )
end

