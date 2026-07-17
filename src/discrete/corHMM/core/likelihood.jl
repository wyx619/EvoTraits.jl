function _corhmm_validate_liks(tree::CompactTree, liks::AbstractMatrix{<:Real})
    size(liks, 1) == tree.ntips || throw(ArgumentError("corHMM tip likelihood matrix must have $(tree.ntips) rows"))
    size(liks, 2) >= 2 || throw(ArgumentError("corHMM tip likelihood matrix must have at least two states"))
    out = Matrix{Float64}(liks)
    for i in axes(out, 1)
        sum(@view out[i, :]) > 0.0 || throw(ArgumentError("corHMM tip likelihood row $i sums to zero"))
    end
    return out
end

function _corhmm_stationary_root(Q::Matrix{Float64})
    pi = stationary_distribution(Q)
    total = sum(pi)
    total > 0.0 || return fill(1.0 / size(Q, 1), size(Q, 1))
    return pi ./ total
end

function _corhmm_root_vector(root_prior, Q::Matrix{Float64}, root_liks::AbstractVector{<:Real})
    if root_prior === :yang
        return _corhmm_stationary_root(Q)
    elseif root_prior === :flat || root_prior === nothing
        return fill(1.0 / size(Q, 1), size(Q, 1))
    elseif root_prior === :maddfitz
        total = sum(root_liks)
        total > 0.0 || return fill(1.0 / size(Q, 1), size(Q, 1))
        return collect(Float64.(root_liks) ./ total)
    elseif root_prior isa AbstractVector
        p = normalize_probability_vector(root_prior)
        length(p) == size(Q, 1) || throw(ArgumentError("root_prior vector length and Q size disagree"))
        return p
    end
    throw(ArgumentError("Unsupported corHMM root_prior=$root_prior"))
end

function _corhmm_aicc(loglik::Float64, nparams::Integer, ntips::Integer)
    denom = ntips - nparams - 1
    return -2loglik + 2 * nparams * (ntips / denom)
end

function _corhmm_branch_lengths(tree::CompactTree)
    out = copy(tree.edge_length)
    bump = sqrt(eps(Float64))
    for i in eachindex(out)
        out[i] <= eps(Float64) && (out[i] += bump)
    end
    return out
end

function _corhmm_pruning_workspace(tree::CompactTree, nstates::Integer)
    n = Int(nstates)
    return CorHMMPruningWorkspace(
        node_liks = zeros(Float64, tree.nnodes, n),
        P = zeros(Float64, n, n),
        tmp = zeros(Float64, n),
        comp = ones(Float64, tree.nnodes),
        root_prior_probs = zeros(Float64, n),
    )
end

Base.@kwdef struct _CorHMMPruningRun
    success::Bool = false
    loglik::Float64 = NaN
    nstates::Int = 0
    nparams::Int = 0
    root_prior::Symbol = :yang
    node_liks::Matrix{Float64} = zeros(0, 0)
    transition_matrix::Matrix{Float64} = zeros(0, 0)
    comp::Vector{Float64} = Float64[]
    root_prior_probs::Vector{Float64} = Float64[]
end

"""
    _corhmm_run_pruning_prevalidated(...)

Shared corHMM pruning kernel used by both the fast log-likelihood path and the
cache-producing path. Statistical semantics stay unchanged; this only removes
duplicated traversal and root-combination code.
"""
function _corhmm_run_pruning_prevalidated(
    tree::CompactTree,
    liks_tip::Matrix{Float64},
    Qf::Matrix{Float64};
    root_prior = :yang,
    nparams::Union{Nothing, Integer} = nothing,
    rate_cat::Integer = 1,
    order_test::Bool = false,
    branch_lengths::Union{Nothing, AbstractVector{<:Real}} = nothing,
    workspace::Union{Nothing, CorHMMPruningWorkspace} = nothing,
    copy_outputs::Bool = false,
)
    nfree = Int(something(nparams, _default_mk_nparams(Qf)))
    nstates = size(Qf, 1)
    root_mode = root_prior isa Symbol ? root_prior : :custom

    if order_test && !corhmm_order_test(Qf, rate_cat)
        return _CorHMMPruningRun(success = false, loglik = -Inf, nstates = nstates, nparams = nfree, root_prior = root_mode)
    end

    size(liks_tip, 2) == nstates || throw(ArgumentError("corHMM tip likelihoods and Q disagree on number of states"))

    ws =
        workspace === nothing ||
        size(workspace.node_liks, 1) != tree.nnodes ||
        size(workspace.node_liks, 2) != nstates ? _corhmm_pruning_workspace(tree, nstates) : workspace

    node_liks = ws.node_liks
    fill!(node_liks, 0.0)
    for (row, node) in enumerate(tree.tip_ids)
        @views node_liks[node, :] .= liks_tip[row, :]
    end

    evals, V, Vinv = _mk_eigen_cache(Qf)
    edge_length = branch_lengths === nothing ? _corhmm_branch_lengths(tree) : Float64.(branch_lengths)
    P = ws.P
    tmp = ws.tmp
    comp = ws.comp
    fill!(comp, 1.0)

    foreach_postorder_internal(tree) do node
        @views fill!(node_liks[node, :], 1.0)
        foreach_child_edge(tree, node) do edge, child
            _fill_transition_matrix!(P, edge_length[edge], evals, V, Vinv)
            mul!(tmp, P, @view node_liks[child, :])
            @views node_liks[node, :] .*= tmp
        end
        c = sum(@view node_liks[node, :])
        c > 0.0 || return _CorHMMPruningRun(success = false, loglik = -Inf, nstates = nstates, nparams = nfree, root_prior = root_mode)
        comp[node] = c
        @views node_liks[node, :] ./= c
    end

    root = tree.root
    root_vec = ws.root_prior_probs
    root_vec .= _corhmm_root_vector(root_prior, Qf, @view node_liks[root, :])
    root_term = sum(root_vec .* @view node_liks[root, :])
    root_term > 0.0 || return _CorHMMPruningRun(success = false, loglik = -Inf, nstates = nstates, nparams = nfree, root_prior = root_mode)

    scale_loglik = 0.0
    foreach_postorder_internal(tree) do node
        scale_loglik += log(comp[node])
    end
    loglik = scale_loglik + log(root_term)

    return _CorHMMPruningRun(
        success = isfinite(loglik),
        loglik = loglik,
        nstates = nstates,
        nparams = nfree,
        root_prior = root_mode,
        node_liks = copy_outputs ? copy(node_liks) : node_liks,
        transition_matrix = copy_outputs ? copy(Qf) : Qf,
        comp = copy_outputs ? copy(comp) : comp,
        root_prior_probs = copy_outputs ? copy(root_vec) : root_vec,
    )
end

function _corhmm_loglik_prevalidated(
    tree::CompactTree,
    liks_tip::Matrix{Float64},
    Qf::Matrix{Float64};
    root_prior = :yang,
    nparams::Union{Nothing, Integer} = nothing,
    rate_cat::Integer = 1,
    order_test::Bool = false,
    branch_lengths::Union{Nothing, AbstractVector{<:Real}} = nothing,
    workspace::Union{Nothing, CorHMMPruningWorkspace} = nothing,
)
    run = _corhmm_run_pruning_prevalidated(
        tree,
        liks_tip,
        Qf;
        root_prior = root_prior,
        nparams = nparams,
        rate_cat = rate_cat,
        order_test = order_test,
        branch_lengths = branch_lengths,
        workspace = workspace,
        copy_outputs = false,
    )
    return run.loglik
end

function _corhmm_pruning_cache_prevalidated(
    tree::CompactTree,
    liks_tip::Matrix{Float64},
    Qf::Matrix{Float64};
    root_prior = :yang,
    nparams::Union{Nothing, Integer} = nothing,
    rate_cat::Integer = 1,
    order_test::Bool = false,
    branch_lengths::Union{Nothing, AbstractVector{<:Real}} = nothing,
    workspace::Union{Nothing, CorHMMPruningWorkspace} = nothing,
)
    run = _corhmm_run_pruning_prevalidated(
        tree,
        liks_tip,
        Qf;
        root_prior = root_prior,
        nparams = nparams,
        rate_cat = rate_cat,
        order_test = order_test,
        branch_lengths = branch_lengths,
        workspace = workspace,
        copy_outputs = true,
    )
    if !run.success
        return CorHMMPruningCache(success = false, loglik = -Inf, aic = Inf, aicc = Inf, nstates = run.nstates, nparams = run.nparams, root_prior = run.root_prior)
    end
    loglik = run.loglik
    return CorHMMPruningCache(
        success = isfinite(loglik),
        loglik = loglik,
        aic = compute_aic(loglik, run.nparams),
        aicc = _corhmm_aicc(loglik, run.nparams, tree.ntips),
        nstates = run.nstates,
        nparams = run.nparams,
        root_prior = run.root_prior,
        node_liks = run.node_liks,
        transition_matrix = run.transition_matrix,
        comp = run.comp,
        root_prior_probs = run.root_prior_probs,
    )
end

function corhmm_order_test(Q::AbstractMatrix{<:Real}, rate_cat::Integer)
    rc = _validate_corhmm_rate_cat(rate_cat)
    rc == 1 && return true
    size(Q, 1) == size(Q, 2) || return false
    nstates = size(Q, 1)
    nstates % rc == 0 || return false
    nobs = nstates ÷ rc
    rates = Vector{Float64}(undef, rc)
    for r in 1:rc
        lo = (r - 1) * nobs + 1
        hi = r * nobs
        value = NaN
        for i in lo:hi, j in lo:hi
            i == j && continue
            q = Float64(Q[i, j])
            if q >= 0.0
                value = q
                break
            end
        end
        isfinite(value) || return false
        rates[r] = value
    end
    return rates == sort(rates; rev = true)
end

function corhmm_pruning_cache(
    tree::CompactTree,
    tip_liks::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior = :yang,
    nparams::Union{Nothing, Integer} = nothing,
    rate_cat::Integer = 1,
    order_test::Bool = false,
    branch_lengths::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    liks_tip = _corhmm_validate_liks(tree, tip_liks)
    Qf = _validate_rate_matrix(Q)
    return _corhmm_pruning_cache_prevalidated(
        tree,
        liks_tip,
        Qf;
        root_prior = root_prior,
        nparams = nparams,
        rate_cat = rate_cat,
        order_test = order_test,
        branch_lengths = branch_lengths,
    )
end

function corhmm_loglikelihood(tree::CompactTree, state_data::CorHMMStateData, Q::AbstractMatrix{<:Real}; root_prior = :yang, nparams = nothing)
    return corhmm_pruning_cache(tree, state_data.tip_priors_hidden, Q; root_prior = root_prior, nparams = nparams)
end

