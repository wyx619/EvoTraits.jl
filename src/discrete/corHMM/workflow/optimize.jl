function fitchscore(tree::CompactTree, tip_priors::AbstractMatrix{<:Real}, hidden_to_observed::AbstractVector{<:Integer})
    nobs = maximum(hidden_to_observed)
    sets = [falses(nobs) for _ in 1:tree.nnodes]
    for (row, node) in enumerate(tree.tip_ids)
        s = sets[Int(node)]
        for h in axes(tip_priors, 2)
            tip_priors[row, h] > 0.0 && (s[Int(hidden_to_observed[h])] = true)
        end
        any(s) || fill!(s, true)
    end
    score = 0
    for node in tree.postorder_internal
        first_edge = Int(tree.first_child_edge[node])
        last_edge = Int(tree.last_child_edge[node])
        @views sets[Int(node)] .= sets[Int(tree.child_of_edge[first_edge])]
        for edge in (first_edge + 1):last_edge
            child_set = sets[Int(tree.child_of_edge[edge])]
            inter = sets[Int(node)] .& child_set
            if any(inter)
                sets[Int(node)] .= inter
            else
                sets[Int(node)] .|= child_set
                score += 1
            end
        end
    end
    return score / 2.0
end

function meanchange(tree::CompactTree, tip_priors::AbstractMatrix{<:Real}, hidden_to_observed::AbstractVector{<:Integer})
    total_length = sum(tree.edge_length)
    total_length > 0.0 || return 0.0
    return fitchscore(tree, tip_priors, hidden_to_observed) / total_length
end

function makestarts(
    tree::CompactTree,
    priors::AbstractMatrix{<:Real},
    hidden_to_observed::AbstractVector{<:Integer},
    nparams::Integer;
    Ntrials::Integer,
    lower_bound::Float64,
    upper_bound::Float64,
    ip::Union{Nothing, AbstractVector{<:Real}, Real},
    rng::AbstractRNG,
)
    if ip !== nothing
        return [startrates(nparams, ip)]
    end
    mean_change = meanchange(tree, priors, hidden_to_observed)
    starts = Vector{Vector{Float64}}(undef, Ntrials)
    for k in 1:Ntrials
        start = Vector{Float64}(undef, nparams)
        if mean_change == 0.0
            fill!(start, 0.01 + lower_bound)
        else
            for i in 1:nparams
                start[i] = randexp(rng) * mean_change
            end
            sort!(start; rev = true)
        end
        for i in eachindex(start)
            if start[i] < lower_bound || start[i] > upper_bound
                start[i] = lower_bound
            end
        end
        starts[k] = start
    end
    return starts
end

function startrates(nparams::Integer, ip)
    if ip isa Real
        v = fill(Float64(ip), nparams)
    else
        v = Float64.(collect(ip))
        if length(v) != nparams
            out = Vector{Float64}(undef, nparams)
            for i in 1:nparams
                out[i] = v[mod1(i, length(v))]
            end
            v = out
        end
    end
    all(x -> isfinite(x) && x > 0.0, v) || throw(ArgumentError("ip values must be positive finite rates"))
    return v
end

_corhmm_fitch_score(args...; kwargs...) = fitchscore(args...; kwargs...)
_corhmm_mean_change(args...; kwargs...) = meanchange(args...; kwargs...)
_generate_corhmm_starts(args...; kwargs...) = makestarts(args...; kwargs...)
_corhmm_start_rates(args...; kwargs...) = startrates(args...; kwargs...)


function fitindexmodel(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    index_matrix::AbstractMatrix{<:Integer};
    root_prior,
    rate_cat::Integer,
    hidden_to_observed::AbstractVector{<:Integer},
    order_test::Bool,
    branch_lengths::AbstractVector{<:Real},
    Ntrials::Integer,
    Nthreads::Integer,
    max_iterations::Integer,
    rel_tol::Float64,
    lower_bound::Float64,
    upper_bound::Float64,
    ip::Union{Nothing, AbstractVector{<:Real}, Real},
    rng::AbstractRNG,
)
    nparams = maximum(index_matrix)
    nparams >= 1 || throw(ArgumentError("corHMM index matrix has no free parameters"))
    _validate_parallel_controls(Ntrials, nothing, Nthreads)
    priors = _corhmm_validate_liks(tree, tip_priors)
    root = _corhmm_root_prior(root_prior)
    lower_bound > 0.0 || throw(ArgumentError("lower_bound must be positive"))
    upper_bound > lower_bound || throw(ArgumentError("upper_bound must be greater than lower_bound"))

    startsets = makestarts(
        tree,
        priors,
        hidden_to_observed,
        nparams;
        Ntrials = Ntrials,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        ip = ip,
        rng = rng,
    )

    trial_logliks = fill(-Inf, length(startsets))
    candidates = Vector{Union{Nothing, MkFitResult}}(undef, length(startsets))
    fill!(candidates, nothing)
    nstates = size(index_matrix, 1)

    runtrial! = function (trial::Int, start_rates_in)
        start_rates = Float64.(start_rates_in)
        start_logrates = log.(max.(start_rates, 1e-8))
        Q = zeros(Float64, nstates, nstates)
        ws = _corhmm_pruning_workspace(tree, nstates)

        objective = function (logrates)
            any(!isfinite, logrates) && return Inf
            rates = exp.(Float64.(logrates))
            try
                fill!(Q, 0.0)
                @inbounds for i in axes(index_matrix, 1)
                    rowsum = 0.0
                    for j in axes(index_matrix, 2)
                        i == j && continue
                        idx = Int(index_matrix[i, j])
                        if idx > 0
                            rate = Float64(rates[idx])
                            rate >= 0.0 || return Inf
                            Q[i, j] = rate
                            rowsum += rate
                        end
                    end
                    Q[i, i] = -rowsum
                end
                loglik = _corhmm_loglik_prevalidated(
                    tree,
                    priors,
                    Q;
                    root_prior = root,
                    nparams = nparams,
                    rate_cat = rate_cat,
                    order_test = order_test,
                    branch_lengths = branch_lengths,
                    workspace = ws,
                )
                return isfinite(loglik) ? -loglik : Inf
            catch
                return Inf
            end
        end

        try
            opt = NLopt.Opt(:LN_SBPLX, nparams)
            NLopt.lower_bounds!(opt, fill(log(lower_bound), nparams))
            NLopt.upper_bounds!(opt, fill(log(upper_bound), nparams))
            NLopt.maxeval!(opt, max_iterations)
            NLopt.ftol_rel!(opt, rel_tol)
            NLopt.min_objective!(opt, (x, grad) -> objective(x))
            _, minx, ret = NLopt.optimize(opt, start_logrates)
            rates = exp.(minx)

            fill!(Q, 0.0)
            @inbounds for i in axes(index_matrix, 1)
                rowsum = 0.0
                for j in axes(index_matrix, 2)
                    i == j && continue
                    idx = Int(index_matrix[i, j])
                    if idx > 0
                        rate = Float64(rates[idx])
                        Q[i, j] = rate
                        rowsum += rate
                    end
                end
                Q[i, i] = -rowsum
            end

            like = _corhmm_pruning_cache_prevalidated(
                tree,
                priors,
                Q;
                root_prior = root,
                nparams = nparams,
                rate_cat = rate_cat,
                order_test = order_test,
                branch_lengths = branch_lengths,
                workspace = ws,
            )
            trial_logliks[trial] = like.loglik
            candidates[trial] = MkFitResult(
                success = like.success,
                loglik = like.loglik,
                aic = like.aic,
                nparams = like.nparams,
                nstates = nstates,
                root_prior = like.root_prior,
                nrates = nparams,
                rates = collect(rates),
                transition_matrix = copy(Q),
                start_rates = collect(start_rates),
                trial_logliks = Float64[],
                converged = ret in (NLopt.SUCCESS, NLopt.STOPVAL_REACHED, NLopt.FTOL_REACHED, NLopt.XTOL_REACHED, NLopt.MAXEVAL_REACHED),
                iterations = NLopt.numevals(opt),
                f_calls = NLopt.numevals(opt),
            )
        catch
            trial_logliks[trial] = -Inf
            candidates[trial] = nothing
        end
        return nothing
    end

    if Nthreads > 1 && length(startsets) > 1
        Threads.@threads for trial in eachindex(startsets)
            runtrial!(trial, startsets[trial])
        end
    else
        for trial in eachindex(startsets)
            runtrial!(trial, startsets[trial])
        end
    end

    best = nothing
    for candidate in candidates
        candidate === nothing && continue
        if best === nothing || candidate.loglik > best.loglik
            best = candidate
        end
    end

    best === nothing && return MkFitResult(success = false, nstates = size(index_matrix, 1), nparams = nparams)
    return MkFitResult(
        success = best.success,
        loglik = best.loglik,
        aic = best.aic,
        nparams = best.nparams,
        nstates = best.nstates,
        root_prior = best.root_prior,
        nrates = best.nrates,
        rates = best.rates,
        transition_matrix = best.transition_matrix,
        start_rates = best.start_rates,
        trial_logliks = trial_logliks,
        converged = best.converged,
        iterations = best.iterations,
        f_calls = best.f_calls,
    )
end

_fit_corhmm_index_model(args...; kwargs...) = fitindexmodel(args...; kwargs...)

