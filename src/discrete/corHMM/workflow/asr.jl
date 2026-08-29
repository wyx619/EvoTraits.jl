function asr_corhmm(fit::CorHMMFitResult; mode::Symbol = :marginal)
    validatenodestates(mode)
    mode === :none && return CorHMMASRResult(success = fit.success, mode = mode, fit = fit)
    fit.success || return CorHMMASRResult(success = false, mode = mode, fit = fit)
    fit.tree === nothing && throw(ArgumentError("CorHMM fit does not store its tree"))

    tree = fit.tree
    cache = corhmm_pruning_cache(
        tree,
        fit.tip_priors_hidden,
        fit.transition_matrix;
        root_prior = resolverootprior(fit.root_prior),
        nparams = fit.nparams,
        rate_cat = fit.rate_cat,
        order_test = get(fit.diagnostics, :order_test, false),
        branch_lengths = fitbranchlengths(fit),
        fixed_node_states = get(fit.diagnostics, :fixed_node_states, nothing),
        hidden_to_observed = fit.hidden_to_observed,
    )
    cache.success || return CorHMMASRResult(success = false, mode = mode, fit = fit)

    node_ids = internalnodes(tree)
    tip_hidden_likelihoods = tipliks(tree, cache)
    tip_observed_likelihoods = collapse_hidden_likelihoods(tip_hidden_likelihoods, fit.hidden_to_observed, length(fit.observed_labels))
    if mode === :joint
        joint = jointreconstruction(
            tree,
            fit.tip_priors_hidden,
            fit.transition_matrix;
            root_prior = cache.root_prior,
            fixed_node_states = get(fit.diagnostics, :fixed_node_states, nothing),
            hidden_to_observed = fit.hidden_to_observed,
        )
        hidden_likelihoods = zeros(Float64, length(joint.node_ids), length(fit.hidden_labels))
        for (i, state) in enumerate(joint.internal_states)
            hidden_likelihoods[i, Int(state)] = 1.0
        end
        observed_likelihoods = collapse_hidden_likelihoods(hidden_likelihoods, fit.hidden_to_observed, length(fit.observed_labels))
        observed_states = Int32[fit.hidden_to_observed[Int(state)] for state in joint.internal_states]
        tip_observed_states = Int32[fit.hidden_to_observed[Int(state)] for state in joint.tip_states]
        return CorHMMASRResult(
            success = true,
            mode = mode,
            node_ids = joint.node_ids,
            hidden_likelihoods = hidden_likelihoods,
            observed_likelihoods = observed_likelihoods,
            tip_hidden_likelihoods = tip_hidden_likelihoods,
            tip_observed_likelihoods = tip_observed_likelihoods,
            hidden_states = joint.internal_states,
            observed_states = observed_states,
            tip_hidden_states = joint.tip_states,
            tip_observed_states = tip_observed_states,
            joint_loglik = joint.joint_loglik,
            hidden_state_labels = fit.hidden_labels,
            observed_state_labels = fit.observed_labels,
            fit = fit,
        )
    end

    hidden_likelihoods =
        if mode === :scaled
            scaledliks(tree, cache, node_ids)
        else
            marginalliks(tree, cache, fitbranchlengths(fit))
        end
    hidden_states = jointstates(hidden_likelihoods)
    observed_likelihoods = collapse_hidden_likelihoods(hidden_likelihoods, fit.hidden_to_observed, length(fit.observed_labels))
    observed_states = jointstates(observed_likelihoods)

    return CorHMMASRResult(
        success = true,
        mode = mode,
        node_ids = node_ids,
        hidden_likelihoods = hidden_likelihoods,
        observed_likelihoods = observed_likelihoods,
        tip_hidden_likelihoods = tip_hidden_likelihoods,
        tip_observed_likelihoods = tip_observed_likelihoods,
        hidden_states = hidden_states,
        observed_states = observed_states,
        hidden_state_labels = fit.hidden_labels,
        observed_state_labels = fit.observed_labels,
        fit = fit,
    )
end

