function fitdiag(state_data::CorHMMStateData, order_test::Bool, collapse::Bool)
    return Dict{Symbol, Any}(
        :polymorphic_tips => count(state_data.polymorphic_tip_mask),
        :missing_tips => count(state_data.missing_tip_mask),
        :tip_state_labels => state_data.tip_state_labels,
        :order_test => order_test,
        :collapse => collapse,
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
)
    node_mode = _validate_corhmm_node_states(node_states)
    if collapse == false && state_order === nothing
        throw(ArgumentError("corHMM collapse=false for single-trait data requires state_order to define possible states"))
    end
    state_data = parsestates(tree, states; state_order = state_order, rate_cat = rate_cat)
    branch_lengths = _corhmm_branch_lengths(tree)
    resolved_root = _corhmm_root_prior(root_prior)
    index_matrix =
        rate_mat === nothing ?
        rateindex(length(state_data.observed_labels); model = model, rate_cat = state_data.rate_cat) :
        normalize_ratematrix(rate_mat)
    size(index_matrix, 1) == length(state_data.hidden_labels) || throw(ArgumentError("rate_mat size must match observed states * rate_cat"))
    order_test = rate_mat === nothing && state_data.rate_cat > 1
    return (
        node_mode = node_mode,
        state_data = state_data,
        branch_lengths = branch_lengths,
        resolved_root = resolved_root,
        index_matrix = index_matrix,
        order_test = order_test,
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
    rng::AbstractRNG,
)
    ntrials = ip === nothing ? max(Ntrials, nstarts, 1) : 1
    fit = fitindexmodel(
        tree,
        spec.state_data.tip_priors_hidden,
        spec.index_matrix;
        root_prior = spec.resolved_root,
        rate_cat = spec.state_data.rate_cat,
        hidden_to_observed = spec.state_data.hidden_to_observed,
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
    )
    fit = _with_mk_fit_state_metadata(fit, Any[state for state in spec.state_data.hidden_labels])
    root_prior_probs =
        fit.success ?
        corhmm_pruning_cache(
            tree,
            spec.state_data.tip_priors_hidden,
            fit.transition_matrix;
            root_prior = spec.resolved_root,
            nparams = fit.nparams,
            rate_cat = spec.state_data.rate_cat,
            order_test = spec.order_test,
            branch_lengths = spec.branch_lengths,
        ).root_prior_probs :
        Float64[]
    solution = fit.success ? solutionmatrix(fit.rates, spec.index_matrix) : zeros(0, 0)
    diagnostics = fitdiag(spec.state_data, spec.order_test, collapse)
    return fitresult(
        tree,
        spec.state_data,
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
    fixed_nodes::Bool = false,
    lewis_asc_bias::Bool = false,
)
    tip_fog === nothing || throw(ArgumentError("corHMM tip_fog is not supported in EvoTraits"))
    fixed_nodes == false || throw(ArgumentError("corHMM fixed_nodes is not supported in EvoTraits"))
    lewis_asc_bias == false || throw(ArgumentError("corHMM Lewis ascertainment bias is not supported in EvoTraits"))

    spec = fitspec(tree, states, model, rate_cat, rate_mat, node_states, root_prior, collapse, state_order)
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
        rng = rng,
    )
    spec.node_mode === :none && return result
    asr = asr_corhmm(result; mode = spec.node_mode)
    return attachasr(result, asr, spec.node_mode)
end

_corhmm_fit_diagnostics(args...; kwargs...) = fitdiag(args...; kwargs...)
_corhmm_base_fit_result(args...; kwargs...) = fitresult(args...; kwargs...)
_corhmm_attach_asr(args...; kwargs...) = attachasr(args...; kwargs...)

