@inline _rates_to_logrates(rates::AbstractVector{<:Real}, lower_rate::Float64) = log.(max.(Float64.(rates), lower_rate))
@inline _logrates_to_rates(logrates::AbstractVector{<:Real}) = exp.(Float64.(logrates))

function _validate_parallel_controls(Ntrials::Integer, Nscouts::Union{Nothing, Integer}, Nthreads::Integer)
    Ntrials >= 1 || throw(ArgumentError("Ntrials must be >= 1"))
    Nthreads >= 1 || throw(ArgumentError("Nthreads must be >= 1"))
    Nthreads <= Threads.nthreads() || throw(ArgumentError("Requested Nthreads=$Nthreads, but Julia session only has $(Threads.nthreads()) thread(s)"))
    Nthreads <= Ntrials || throw(ArgumentError("For trial-level parallelism, Nthreads must be <= Ntrials"))
    if Nscouts !== nothing
        Nscouts >= 0 || throw(ArgumentError("Nscouts must be >= 0"))
        Ntrials > 1 && Nscouts < (Ntrials - 1) && throw(ArgumentError("When Ntrials > 1, Nscouts must be at least Ntrials - 1"))
    end
    return true
end

function _mk_fit_algorithm(optim_method::Symbol)
    if optim_method === :LBFGS
        return Optim.LBFGS()
    elseif optim_method === :NelderMead
        return Optim.NelderMead()
    end
    throw(ArgumentError("Unsupported optim_method=$optim_method"))
end

function _mk_failed_fit(
    start_rates::AbstractVector{<:Real},
    nstates::Integer,
    root_prior,
    rate_model::Symbol,
    trial_logliks::Vector{Float64},
)
    start = Float64.(start_rates)
    return MkFitResult(
        success = false,
        loglik = -Inf,
        aic = Inf,
        nparams = length(start),
        nstates = nstates,
        root_prior = (root_prior isa Symbol ? root_prior : :custom),
        nrates = length(start),
        rates = collect(start),
        transition_matrix = mk_rates_to_Q(start, nstates; rate_model = rate_model),
        start_rates = collect(start),
        trial_logliks = copy(trial_logliks),
        converged = false,
        iterations = 0,
        f_calls = 0,
    )
end

function _mk_run_trial!(
    candidates::Vector{Union{Nothing, MkFitResult}},
    trial_logliks::Vector{Float64},
    trial::Int,
    start_rates_in,
    tree::CompactTree,
    priors::AbstractMatrix{<:Real},
    nstates::Integer,
    rate_model::Symbol,
    root_prior,
    root_prior_probs,
    algorithm,
    max_iterations::Integer,
    rel_tol::Float64,
    lower_rate::Float64,
)
    start_rates = Float64.(start_rates_in)
    start_logrates = _rates_to_logrates(start_rates, lower_rate)

    objective = function (logrates)
        any(!isfinite, logrates) && return Inf
        rates = _logrates_to_rates(logrates)
        any(!isfinite, rates) && return Inf
        try
            Q = mk_rates_to_Q(rates, nstates; rate_model = rate_model)
            root_mode, root_vec = _prepare_root_prior(root_prior, root_prior_probs, Q, nstates)
            fit = mk_loglikelihood(tree, priors, Q; root_prior=root_mode, root_prior_probs=root_vec, nparams=length(rates))
            return isfinite(fit.loglik) ? -fit.loglik : Inf
        catch
            return Inf
        end
    end

    try
        result = Optim.optimize(
            objective,
            start_logrates,
            algorithm,
            Optim.Options(
                iterations = max_iterations,
                f_reltol = rel_tol,
                g_tol = rel_tol,
                allow_f_increases = false,
            ),
        )

        fitted_rates = _logrates_to_rates(Optim.minimizer(result))
        fitted_Q = mk_rates_to_Q(fitted_rates, nstates; rate_model = rate_model)
        root_mode, root_vec = _prepare_root_prior(root_prior, root_prior_probs, fitted_Q, nstates)
        likelihood = mk_loglikelihood(tree, priors, fitted_Q; root_prior=root_mode, root_prior_probs=root_vec, nparams=length(fitted_rates))
        trial_logliks[trial] = likelihood.loglik

        candidates[trial] = MkFitResult(
            success = likelihood.success,
            loglik = likelihood.loglik,
            aic = likelihood.aic,
            nparams = likelihood.nparams,
            nstates = nstates,
            root_prior = root_mode,
            nrates = length(fitted_rates),
            rates = collect(fitted_rates),
            transition_matrix = fitted_Q,
            start_rates = collect(start_rates),
            trial_logliks = copy(trial_logliks),
            converged = Optim.converged(result),
            iterations = Optim.iterations(result),
            f_calls = Optim.f_calls(result),
        )
    catch
        trial_logliks[trial] = -Inf
        candidates[trial] = _mk_failed_fit(start_rates, nstates, root_prior, rate_model, trial_logliks)
    end

    return nothing
end

function _mk_select_best_result(candidates::Vector{Union{Nothing, MkFitResult}})
    best_result = nothing
    for candidate in candidates
        candidate === nothing && continue
        if best_result === nothing || candidate.loglik > best_result.loglik
            best_result = candidate
        end
    end
    return best_result
end

function _mk_fit_with_nstates(
    tree::CompactTree,
    nstates::Integer;
    tip_states=nothing,
    tip_priors=nothing,
    rate_model::Symbol = :ARD,
    root_prior::Union{Symbol, AbstractVector{<:Real}} = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_rates::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_transition_matrix::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    Ntrials::Integer = 1,
    Nscouts::Union{Nothing, Integer} = nothing,
    Nthreads::Integer = 1,
    optim_method::Symbol = :LBFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    lower_rate::Float64 = 1e-8,
    rng::AbstractRNG = Random.default_rng(),
)
    mk_nrates(rate_model, nstates)
    _validate_parallel_controls(Ntrials, Nscouts, Nthreads)
    priors = _tip_priors_input(tree, nstates; tip_states=tip_states, tip_priors=tip_priors)

    if guess_transition_matrix !== nothing
        guess_rates = mk_Q_to_rates(guess_transition_matrix; rate_model = rate_model)
    end
    starts = _generate_ard_starts(
        tree,
        priors,
        nstates;
        rate_model = rate_model,
        root_prior = root_prior isa Symbol ? root_prior : :custom,
        root_prior_probs = root_prior isa Symbol ? root_prior_probs : root_prior,
        guess_rates = guess_rates,
        Ntrials = Ntrials,
        Nscouts = Nscouts,
        rng = rng,
        threaded = false,
    )

    algorithm = _mk_fit_algorithm(optim_method)
    trial_logliks = fill(-Inf, length(starts))
    candidates = Vector{Union{Nothing, MkFitResult}}(undef, length(starts))
    fill!(candidates, nothing)

    if Nthreads > 1 && length(starts) > 1
        Threads.@threads for trial in eachindex(starts)
            _mk_run_trial!(
                candidates,
                trial_logliks,
                trial,
                starts[trial],
                tree,
                priors,
                nstates,
                rate_model,
                root_prior,
                root_prior_probs,
                algorithm,
                max_iterations,
                rel_tol,
                lower_rate,
            )
        end
    else
        for trial in eachindex(starts)
            _mk_run_trial!(
                candidates,
                trial_logliks,
                trial,
                starts[trial],
                tree,
                priors,
                nstates,
                rate_model,
                root_prior,
                root_prior_probs,
                algorithm,
                max_iterations,
                rel_tol,
                lower_rate,
            )
        end
    end

    best_result = _mk_select_best_result(candidates)
    best_result === nothing && return MkFitResult(success=false, nstates=nstates, root_prior=(root_prior isa Symbol ? root_prior : :custom))

    return MkFitResult(
        success = best_result.success,
        loglik = best_result.loglik,
        aic = best_result.aic,
        nparams = best_result.nparams,
        nstates = best_result.nstates,
        root_prior = best_result.root_prior,
        nrates = best_result.nrates,
        rates = best_result.rates,
        transition_matrix = best_result.transition_matrix,
        start_rates = best_result.start_rates,
        trial_logliks = trial_logliks,
        converged = best_result.converged,
        iterations = best_result.iterations,
        f_calls = best_result.f_calls,
    )
end
