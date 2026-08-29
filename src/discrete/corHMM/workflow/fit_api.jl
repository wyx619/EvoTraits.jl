function fitdiag(
    state_data::CorHMMStateData,
    order_test::Bool,
    collapse::Bool;
    fixed_node_states = nothing,
    tip_fog = nothing,
    tip_fog_groups::AbstractVector{<:Integer} = Int[],
    lewis_asc_bias::Bool = false,
)
    return Dict{Symbol, Any}(
        :polymorphic_tips => count(state_data.polymorphic_tip_mask),
        :missing_tips => count(state_data.missing_tip_mask),
        :tip_state_labels => state_data.tip_state_labels,
        :order_test => order_test,
        :collapse => collapse,
        :multi_character => state_data.multi_character,
        :character_labels => state_data.character_labels,
        :character_state_labels => state_data.character_state_labels,
        :possible_labels => state_data.possible_labels,
        :fixed_node_states => fixed_node_states,
        :tip_fog => tip_fog,
        :tip_fog_groups => Int.(collect(tip_fog_groups)),
        :lewis_asc_bias => lewis_asc_bias,
    )
end

function fitresult(
    tree::CompactTree,
    state_data::CorHMMStateData,
    fit::MkFitResult,
    index_matrix::AbstractMatrix{<:Integer},
    solution::AbstractMatrix{<:Real},
    branch_lengths::AbstractVector{<:Real},
    root_prior,
    root_prior_probs::AbstractVector{<:Real},
    model::Symbol,
    node_mode::Symbol,
    collapse::Bool,
    diagnostics::Dict{Symbol, Any},
    tip_fog = nothing,
)
    return CorHMMFitResult(
        tree = tree,
        success = fit.success,
        loglik = fit.loglik,
        aic = fit.aic,
        aicc = _corhmm_aicc(fit.loglik, fit.nparams, tree.ntips),
        nparams = fit.nparams,
        model = model,
        rate_cat = state_data.rate_cat,
        node_states = node_mode,
        root_prior = root_prior,
        collapse = collapse,
        tip_fog = tip_fog,
        observed_labels = state_data.observed_labels,
        hidden_labels = state_data.hidden_labels,
        hidden_to_observed = state_data.hidden_to_observed,
        tip_priors_observed = state_data.tip_priors_observed,
        tip_priors_hidden = state_data.tip_priors_hidden,
        rates = fit.rates,
        index_matrix = Int.(index_matrix),
        solution = Matrix{Float64}(solution),
        transition_matrix = fit.transition_matrix,
        root_prior_probs = Float64.(root_prior_probs),
        branch_lengths = Float64.(branch_lengths),
        fit = fit,
        diagnostics = diagnostics,
    )
end

function _corhmm_fixed_node_states(tree::CompactTree, state_data::CorHMMStateData, fixed_nodes)
    fixed_nodes === false && return nothing
    out = zeros(Int, tree.nnodes)
    nstates = length(state_data.observed_labels)
    resolve_state(value) = begin
        if value isa Integer
            1 <= value <= nstates || throw(ArgumentError("Fixed corHMM node state index $value is outside 1:$nstates"))
            return Int(value)
        end
        label = strip(string(value))
        idx = findfirst(==(label), state_data.observed_labels)
        idx === nothing && throw(ArgumentError("Fixed corHMM node state '$label' is not in observed_labels"))
        return Int(idx)
    end
    if fixed_nodes === true
        for node in tree.postorder_internal
            label = strip(tree.node_labels[Int(node)])
            isempty(label) && continue
            try
                out[Int(node)] = resolve_state(label)
            catch
                continue
            end
        end
        return out
    end
    fixed_nodes isa AbstractDict || throw(ArgumentError("fixed_nodes must be false, true, or a node-to-state dictionary"))
    for (key, value) in pairs(fixed_nodes)
        node = key isa Integer ? Int(key) : findfirst(==(string(key)), tree.node_labels)
        node === nothing && throw(ArgumentError("Fixed node '$key' was not found in the CompactTree"))
        1 <= node <= tree.nnodes || throw(ArgumentError("Fixed node id $node is outside the tree"))
        tree.is_tip[node] && throw(ArgumentError("fixed_nodes may only specify internal nodes"))
        out[node] = resolve_state(value)
    end
    return out
end

function attachasr(result::CorHMMFitResult, asr::CorHMMASRResult, node_mode::Symbol)
    return CorHMMFitResult(
        tree = result.tree,
        success = result.success,
        loglik = result.loglik,
        aic = result.aic,
        aicc = result.aicc,
        nparams = result.nparams,
        model = result.model,
        rate_cat = result.rate_cat,
        node_states = result.node_states,
        root_prior = result.root_prior,
        collapse = result.collapse,
        observed_labels = result.observed_labels,
        hidden_labels = result.hidden_labels,
        hidden_to_observed = result.hidden_to_observed,
        tip_priors_observed = result.tip_priors_observed,
        tip_priors_hidden = result.tip_priors_hidden,
        rates = result.rates,
        index_matrix = result.index_matrix,
        solution = result.solution,
        transition_matrix = result.transition_matrix,
        root_prior_probs = result.root_prior_probs,
        branch_lengths = result.branch_lengths,
        tip_fog = result.tip_fog,
        states = asr.observed_likelihoods,
        tip_states = _corhmm_r_tip_states(result, node_mode),
        fit = result.fit,
        asr = asr,
        diagnostics = result.diagnostics,
    )
end

function fitspec(
    tree::CompactTree,
    states,
    model::Symbol,
    rate_cat::Integer,
    rate_mat,
    node_states::Symbol,
    root_prior,
    collapse::Bool,
    state_order,
    fixed_nodes,
    tip_fog,
)
    node_mode = _validate_corhmm_node_states(node_states)
    state_data = parsestates(tree, states; state_order = state_order, rate_cat = rate_cat, collapse = collapse)
    if collapse == false && state_order === nothing && !state_data.multi_character
        throw(ArgumentError("corHMM collapse=false for single-trait data requires state_order to define possible states"))
    end
    branch_lengths = _corhmm_branch_lengths(tree)
    resolved_root = _corhmm_root_prior(root_prior)
    fixed_node_states = _corhmm_fixed_node_states(tree, state_data, fixed_nodes)
    fog_spec = _corhmm_tip_fog_spec(tip_fog, length(state_data.observed_labels))
    effective_state_data = fog_spec.fixed === nothing ? state_data : _corhmm_state_data_with_tip_fog(state_data, fog_spec.fixed)
    index_matrix =
        rate_mat === nothing ?
        (state_data.multi_character ?
         _corhmm_joint_rateindex(effective_state_data; model = model) :
         rateindex(length(effective_state_data.observed_labels); model = model, rate_cat = effective_state_data.rate_cat)) :
        normalize_ratematrix(rate_mat)
    size(index_matrix, 1) == length(effective_state_data.hidden_labels) || throw(ArgumentError("rate_mat size must match observed states * rate_cat"))
    order_test = rate_mat === nothing && effective_state_data.rate_cat > 1
    return (
        node_mode = node_mode,
        state_data = effective_state_data,
        branch_lengths = branch_lengths,
        resolved_root = resolved_root,
        index_matrix = index_matrix,
        order_test = order_test,
        fixed_node_states = fixed_node_states,
        tip_fog = fog_spec.fixed,
        tip_fog_groups = fog_spec.groups,
        tip_fog_estimated = fog_spec.estimated,
    )
end

function fitcore(
    tree::CompactTree,
    spec;
    model::Symbol,
    root_prior,
    collapse::Bool,
    Ntrials::Integer,
    Nthreads::Integer,
    nstarts::Integer,
    ip::Union{Nothing, AbstractVector{<:Real}, Real},
    lower_bound::Float64,
    upper_bound::Float64,
    max_iterations::Integer,
    rel_tol::Float64,
    lewis_asc_bias::Bool,
    fog_ip::Float64,
    rng::AbstractRNG,
)
    ntrials = ip === nothing ? max(Ntrials, nstarts, 1) : 1
    optimization = fitindexmodel(
        tree,
        spec.state_data.tip_priors_hidden,
        spec.index_matrix;
        root_prior = spec.resolved_root,
        rate_cat = spec.state_data.rate_cat,
        hidden_to_observed = spec.state_data.hidden_to_observed,
        fixed_node_states = spec.fixed_node_states,
        lewis_asc_bias = lewis_asc_bias,
        order_test = spec.order_test,
        branch_lengths = spec.branch_lengths,
        Ntrials = ntrials,
        Nthreads = Nthreads,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        ip = ip,
        rng = rng,
        tip_fog_groups = spec.tip_fog_groups,
        fog_ip = fog_ip,
    )
    fit = _with_mk_fit_state_metadata(optimization.fit, Any[state for state in spec.state_data.hidden_labels])
    final_tip_fog = optimization.tip_fog === nothing ? spec.tip_fog : optimization.tip_fog
    fit_state_data =
        optimization.tip_fog === nothing || !spec.tip_fog_estimated ? spec.state_data :
        _corhmm_state_data_with_tip_fog(spec.state_data, optimization.tip_fog)
    root_prior_probs =
        fit.success ?
        corhmm_pruning_cache(
            tree,
            fit_state_data.tip_priors_hidden,
            fit.transition_matrix;
            root_prior = spec.resolved_root,
            nparams = fit.nparams,
            rate_cat = fit_state_data.rate_cat,
            order_test = spec.order_test,
            branch_lengths = spec.branch_lengths,
            fixed_node_states = spec.fixed_node_states,
            hidden_to_observed = fit_state_data.hidden_to_observed,
            lewis_asc_bias = lewis_asc_bias,
        ).root_prior_probs :
        Float64[]
    solution = fit.success ? solutionmatrix(fit.rates, spec.index_matrix) : zeros(0, 0)
    diagnostics = fitdiag(
        fit_state_data,
        spec.order_test,
        collapse;
        fixed_node_states = spec.fixed_node_states,
        tip_fog = final_tip_fog,
        tip_fog_groups = spec.tip_fog_groups,
        lewis_asc_bias = lewis_asc_bias,
    )
    return fitresult(
        tree,
        fit_state_data,
        fit,
        spec.index_matrix,
        solution,
        spec.branch_lengths,
        root_prior,
        root_prior_probs,
        model,
        spec.node_mode,
        collapse,
        diagnostics,
        final_tip_fog,
    )
end

function fit_corhmm(
    tree::CompactTree,
    states;
    model::Symbol = :ARD,
    rate_cat::Integer = 1,
    rate_mat = nothing,
    node_states::Symbol = :marginal,
    root_prior = :yang,
    collapse::Bool = true,
    state_order = nothing,
    Ntrials::Integer = 1,
    Nthreads::Integer = 1,
    nstarts::Integer = 0,
    ip::Union{Nothing, AbstractVector{<:Real}, Real} = nothing,
    lower_bound::Float64 = 1e-9,
    upper_bound::Float64 = 100.0,
    max_iterations::Integer = 1_000_000,
    rel_tol::Float64 = sqrt(eps(Float64)),
    rng::AbstractRNG = Random.default_rng(),
    tip_fog = nothing,
    fixed_nodes = false,
    lewis_asc_bias::Bool = false,
    fog_ip::Float64 = 0.01,
)
    spec = fitspec(tree, states, model, rate_cat, rate_mat, node_states, root_prior, collapse, state_order, fixed_nodes, tip_fog)
    result = fitcore(
        tree,
        spec;
        model = model,
        root_prior = root_prior,
        collapse = collapse,
        Ntrials = Ntrials,
        Nthreads = Nthreads,
        nstarts = nstarts,
        ip = ip,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lewis_asc_bias = lewis_asc_bias,
        fog_ip = fog_ip,
        rng = rng,
    )
    spec.node_mode === :none && return result
    asr = asr_corhmm(result; mode = spec.node_mode)
    return attachasr(result, asr, spec.node_mode)
end

_corhmm_fit_diagnostics(args...; kwargs...) = fitdiag(args...; kwargs...)
_corhmm_base_fit_result(args...; kwargs...) = fitresult(args...; kwargs...)
_corhmm_attach_asr(args...; kwargs...) = attachasr(args...; kwargs...)

