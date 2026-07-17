function _mvbm1_pack_sigma(Sigma::AbstractMatrix{<:Real})
    p = size(Sigma, 1)
    L = cholesky(Symmetric(Matrix{Float64}(Sigma))).L
    pars = Float64[]
    for j in 1:p
        push!(pars, log(L[j, j]))
        for i in (j + 1):p
            push!(pars, L[i, j])
        end
    end
    return pars
end

function _mvbm_pack_sigmas(Sigmas::AbstractVector{<:AbstractMatrix})
    pars = Float64[]
    for Sigma in Sigmas
        append!(pars, _mvbm1_pack_sigma(Sigma))
    end
    return pars
end

function _mvbm1_unpack_sigma(pars::AbstractVector{<:Real}, p::Integer)
    expected = div(p * (p + 1), 2)
    length(pars) == expected || throw(ArgumentError("Sigma parameter vector must have $expected entries"))
    L = zeros(Float64, p, p)
    idx = 1
    for j in 1:p
        L[j, j] = exp(Float64(pars[idx]))
        idx += 1
        for i in (j + 1):p
            L[i, j] = Float64(pars[idx])
            idx += 1
        end
    end
    return L * L'
end

Base.@kwdef struct MVBMNLoptResult
    minimizer::Vector{Float64}
    minimum::Float64
    ret
    iterations::Int = 0
end

function _mvbm_optimize_objective(
    objective,
    p0::AbstractVector{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 1000,
    rel_tol::Float64 = 1e-6,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    lower = lower_bounds === nothing ? nothing : Vector{Float64}(lower_bounds)
    safe_objective = x -> begin
        if lower !== nothing
            @inbounds for i in eachindex(lower)
                x[i] < lower[i] && return 1e300
            end
        end
        value = objective(x)
        return isfinite(value) ? value : 1e300
    end
    if optimization in (:LN_SBPLX, :LN_BOBYQA, :LN_NELDERMEAD)
        init = Vector{Float64}(p0)
        opt = NLopt.Opt(optimization, length(init))
        lower_bounds !== nothing && (opt.lower_bounds = Vector{Float64}(lower_bounds))
        opt.ftol_rel = rel_tol
        opt.xtol_rel = rel_tol
        opt.maxeval = Int(max_iterations)
        opt.min_objective = (x, grad) -> begin
            return safe_objective(x)
        end
        minf, minx, ret = NLopt.optimize(opt, init)
        return MVBMNLoptResult(minimizer = minx, minimum = minf, ret = ret, iterations = NLopt.numevals(opt))
    end

    algorithm =
        optimization === :L_BFGS ? Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()) :
        optimization === :NelderMead ? Optim.NelderMead() :
        throw(ArgumentError("Unsupported optimization=$optimization"))
    return Optim.optimize(
        safe_objective,
        Vector{Float64}(p0),
        algorithm,
        Optim.Options(
            iterations = max_iterations,
            f_reltol = rel_tol,
            g_tol = 1e-4,
            allow_f_increases = false,
        ),
    )
end

function _mvbm_sigma_lower_bounds(p::Integer, nregimes::Integer; min_chol_diag::Float64 = 1e-4)
    block = div(p * (p + 1), 2)
    lower = fill(-Inf, block * nregimes)
    log_min = log(min_chol_diag)
    offset = 0
    for _ in 1:nregimes
        idx = 1
        for j in 1:p
            lower[offset + idx] = log_min
            idx += p - j + 1
        end
        offset += block
    end
    return lower
end

_mvbm_result_minimizer(result::MVBMNLoptResult) = result.minimizer
_mvbm_result_minimizer(result) = Optim.minimizer(result)

_mvbm_result_converged(result::MVBMNLoptResult) =
    result.ret in (NLopt.SUCCESS, NLopt.STOPVAL_REACHED, NLopt.FTOL_REACHED, NLopt.XTOL_REACHED)
_mvbm_result_converged(result) = Optim.converged(result)

_mvbm_result_iterations(result::MVBMNLoptResult) = result.iterations
_mvbm_result_iterations(result) = Optim.iterations(result)

_mvbm_result_f_calls(result::MVBMNLoptResult) = result.iterations
_mvbm_result_f_calls(result) = Optim.f_calls(result)

function _mv_edge_q_cache!(
    Qinv::Array{Float64, 3},
    logdet_Q::AbstractVector{Float64},
    chol_work::AbstractMatrix{Float64},
    edge_Q::Array{Float64, 3},
)
    p = size(edge_Q, 1)
    size(edge_Q, 2) == p || throw(ArgumentError("edge_Q must be square by edge"))
    nedges = size(edge_Q, 3)
    size(Qinv) == (p, p, nedges) || throw(ArgumentError("Qinv workspace has incompatible dimensions"))
    length(logdet_Q) == nedges || throw(ArgumentError("logdet_Q workspace has incompatible dimensions"))
    size(chol_work) == (p, p) || throw(ArgumentError("Cholesky workspace has incompatible dimensions"))
    for edge in 1:nedges
        try
            logdet_Q[edge] = _mvou_qinv_logdet_small!(@view(Qinv[:, :, edge]), @view(edge_Q[:, :, edge]), chol_work)
        catch err
            err isa PosDefException || rethrow()
            return (success = false, Qinv = Qinv, logdet_Q = logdet_Q)
        end
    end
    return (success = true, Qinv = Qinv, logdet_Q = logdet_Q)
end

function _mvbm_unpack_sigmas(pars::AbstractVector{<:Real}, p::Integer, nregimes::Integer)
    block = div(p * (p + 1), 2)
    length(pars) == nregimes * block || throw(ArgumentError("Sigma parameter vector length does not match regime count"))
    Sigmas = Vector{Matrix{Float64}}(undef, nregimes)
    idx = 1
    for r in 1:nregimes
        Sigmas[r] = _mvbm1_unpack_sigma(pars[idx:(idx + block - 1)], p)
        idx += block
    end
    return Sigmas
end

function _mvbm_unpack_sigmas!(
    Sigmas::AbstractVector{<:AbstractMatrix{Float64}},
    pars::AbstractVector{<:Real},
    p::Integer,
    nregimes::Integer,
    Lwork::AbstractMatrix{Float64},
)
    block = div(p * (p + 1), 2)
    length(pars) == nregimes * block || throw(ArgumentError("Sigma parameter vector length does not match regime count"))
    length(Sigmas) == nregimes || throw(ArgumentError("Sigma workspace regime count does not match"))
    size(Lwork) == (p, p) || throw(ArgumentError("Sigma factor workspace has incompatible dimensions"))
    idx = 1
    @inbounds for r in 1:nregimes
        S = Sigmas[r]
        size(S) == (p, p) || throw(ArgumentError("Sigma workspace has incompatible dimensions"))
        fill!(Lwork, 0.0)
        for j in 1:p
            Lwork[j, j] = exp(Float64(pars[idx]))
            idx += 1
            for i in (j + 1):p
                Lwork[i, j] = Float64(pars[idx])
                idx += 1
            end
        end
        mul!(S, Lwork, transpose(Lwork))
    end
    return Sigmas
end


function _mvou_observation_info_to_parent_identity(
    y::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    obs = Vector{Float64}(y)
    p = size(Q, 1)
    obs_idx = findall(!isnan, obs)
    isempty(obs_idx) && return (
        success = true,
        precision = zeros(Float64, p, p),
        linear = zeros(Float64, p),
        logconst = 0.0,
    )
    yobs = obs[obs_idx]
    try
        precision = zeros(Float64, p, p)
        linear = zeros(Float64, p)
        Qinv_y, logdet_Q =
            if length(obs_idx) == p && Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
                precision .= Qinv_cached
                (Qinv_cached * yobs, Float64(logdet_Q_cached))
            else
                Qobs = Matrix{Float64}(Q)[obs_idx, obs_idx]
                cholQ = _mvou_cholesky_psd(Qobs)
                Qinv_I = cholQ \ Matrix{Float64}(I, length(obs_idx), length(obs_idx))
                precision[obs_idx, obs_idx] .= Qinv_I
                (cholQ \ yobs, 2.0 * sum(log, diag(cholQ.L)))
            end
        linear[obs_idx] .= Qinv_y
        nobs = length(yobs)
        return (
            success = true,
            precision = (precision + precision') / 2,
            linear = linear,
            logconst = -0.5 * (dot(yobs, Qinv_y) + logdet_Q + nobs * log(2 * pi)),
        )
    catch
        return (success = false, precision = zeros(Float64, p, p), linear = zeros(Float64, p), logconst = -Inf)
    end
end

function _mvou_observation_info_to_parent_identity!(
    ws::MVProfileWorkspace,
    y::AbstractVector{<:Real},
    Q::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    p = size(Q, 1)
    Pout = ws.msg_precision
    hout = ws.msg_linear
    fill!(Pout, 0.0)
    fill!(hout, 0.0)
    nobs = 0
    @inbounds for row in 1:p
        val = Float64(y[row])
        isnan(val) && continue
        nobs += 1
        ws.obs_index[nobs] = row
        ws.yobs[nobs] = val
    end
    nobs == 0 && return (success = true, precision = Pout, linear = hout, logconst = 0.0)
    try
        yv = @view ws.yobs[1:nobs]
        if nobs == p && Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
            copyto!(Pout, Qinv_cached)
            mul!(hout, Qinv_cached, yv)
            logdet_Q = Float64(logdet_Q_cached)
            return (
                success = true,
                precision = Pout,
                linear = hout,
                logconst = -0.5 * (dot(yv, hout) + logdet_Q + nobs * log(2 * pi)),
            )
        end

        @inbounds for j in 1:nobs, i in 1:nobs
            ws.Qobs[i, j] = Q[ws.obs_index[i], ws.obs_index[j]]
        end
        Qinv_y = @view ws.Qinv_y[1:nobs]
        copyto!(Qinv_y, yv)
        Qinv_I = @view ws.solve_matrix[1:nobs, 1:nobs]
        fill!(Qinv_I, 0.0)
        @inbounds for i in 1:nobs
            Qinv_I[i, i] = 1.0
        end
        logdet_Q = _mvou_cholesky_lower_logdet_small!(@view(ws.Qobs[1:nobs, 1:nobs]))
        _mvou_cholesky_solve_vector_small!(Qinv_y, @view(ws.Qobs[1:nobs, 1:nobs]), nobs)
        _mvou_cholesky_solve_matrix_small!(Qinv_I, @view(ws.Qobs[1:nobs, 1:nobs]), nobs, nobs)
        @inbounds for j in 1:nobs
            jj = ws.obs_index[j]
            hout[jj] = Qinv_y[j]
            for i in 1:nobs
                Pout[ws.obs_index[i], jj] = Qinv_I[i, j]
            end
        end
        return (
            success = true,
            precision = Pout,
            linear = hout,
            logconst = -0.5 * (dot(yv, Qinv_y) + logdet_Q + nobs * log(2 * pi)),
        )
    catch
        return (success = false, precision = Pout, linear = hout, logconst = -Inf)
    end
end

function _mvou_info_to_parent_identity(
    child_precision::AbstractMatrix{<:Real},
    child_linear::AbstractVector{<:Real},
    child_logconst::Real,
    Q::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    Pchild = Matrix{Float64}(child_precision)
    hchild = Vector{Float64}(child_linear)
    p = length(hchild)
    try
        Qinv, logdet_Q =
            Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached)) ?
            (Qinv_cached, Float64(logdet_Q_cached)) :
            _mvou_qinv_logdet(Q)
        S = Pchild + Qinv
        cholS = _mvou_cholesky_psd(S)
        Sinv_h = cholS \ hchild
        Sinv_Qinv = cholS \ Qinv
        logdet_S = 2.0 * sum(log, diag(cholS.L))
        precision = Qinv - Qinv' * Sinv_Qinv
        linear = Qinv' * Sinv_h
        logconst = Float64(child_logconst) - 0.5 * (logdet_Q + logdet_S) + 0.5 * dot(hchild, Sinv_h)
        return (
            success = true,
            precision = (precision + precision') / 2,
            linear = linear,
            logconst = logconst,
        )
    catch
        return (success = false, precision = zeros(Float64, p, p), linear = zeros(Float64, p), logconst = -Inf)
    end
end

function _mvou_info_to_parent_identity!(
    ws::MVProfileWorkspace,
    child_precision::AbstractMatrix{<:Real},
    child_linear::AbstractVector{<:Real},
    child_logconst::Real,
    Q::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    p = length(child_linear)
    Pout = ws.msg_precision
    hout = ws.msg_linear
    fill!(Pout, 0.0)
    fill!(hout, 0.0)
    try
        Qinv, logdet_Q =
            Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached)) ?
            (Qinv_cached, Float64(logdet_Q_cached)) :
            _mvou_qinv_logdet(Q)

        S = ws.Qobs
        S .= child_precision
        S .+= Qinv

        solve_h = ws.solve_vector
        copyto!(solve_h, child_linear)

        solve_Qinv = ws.solve_matrix
        copyto!(solve_Qinv, Qinv)
        logdet_S = _mvou_cholesky_lower_logdet_small!(S)
        _mvou_cholesky_solve_vector_small!(solve_h, S, p)
        _mvou_cholesky_solve_matrix_small!(solve_Qinv, S, p, p)

        mul!(Pout, transpose(Qinv), solve_Qinv, -1.0, 0.0)
        Pout .+= Qinv
        mul!(hout, transpose(Qinv), solve_h)

        logconst = Float64(child_logconst) - 0.5 * (logdet_Q + logdet_S) + 0.5 * dot(child_linear, solve_h)
        return (
            success = true,
            precision = Pout,
            linear = hout,
            logconst = logconst,
        )
    catch
        return (success = false, precision = Pout, linear = hout, logconst = -Inf)
    end
end

function _mv_fixed_root_pruning_profile_validated(
    tree::CompactTree,
    data::Matrix{Float64},
    edge_Q::Array{Float64, 3},
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    _validate_binary_tree(tree)
    p = size(data, 2)
    size(edge_Q, 1) == p && size(edge_Q, 2) == p && size(edge_Q, 3) == tree.nedges ||
        throw(ArgumentError("edge_Q dimensions must be ntraits x ntraits x nedges"))
    ws = workspace === nothing ? _mv_profile_workspace(tree, p) : workspace
    edge_cache = _mv_edge_q_cache!(ws.edge_Qinv, ws.edge_logdet_Q, ws.edge_chol_work, edge_Q)
    edge_cache.success || return (success = false, loglik = -Inf, theta = zeros(Float64, p))
    precision = ws.precision
    linear = ws.linear
    logconst = ws.logconst
    tip_index = ws.tip_index

    for node in tree.postorder_internal
        fill!(precision[node], 0.0)
        fill!(linear[node], 0.0)
        logconst[node] = 0.0
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = tree.child_of_edge[edge]
            qinv = @view(edge_cache.Qinv[:, :, edge])
            logdet_q = edge_cache.logdet_Q[edge]
            msg =
                if tree.is_tip[child]
                    _mvou_observation_info_to_parent_identity!(ws, @view(data[tip_index[child], :]), @view(edge_Q[:, :, edge]), qinv, logdet_q)
                else
                    _mvou_info_to_parent_identity!(ws, precision[child], linear[child], logconst[child], @view(edge_Q[:, :, edge]), qinv, logdet_q)
                end
            msg.success || return (success = false, loglik = -Inf, theta = zeros(Float64, p))
            precision[node] .+= msg.precision
            linear[node] .+= msg.linear
            logconst[node] += msg.logconst
        end
    end

    root = Int(tree.root)
    try
        cholP = _mvou_cholesky_psd(precision[root])
        theta = cholP \ linear[root]
        loglik = logconst[root] - 0.5 * dot(theta, precision[root] * theta) + dot(linear[root], theta)
        return (success = true, loglik = loglik, theta = theta)
    catch
        return (success = false, loglik = -Inf, theta = zeros(Float64, p))
    end
end

function _mv_fixed_root_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_Q::Array{Float64, 3};
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    return _mv_fixed_root_pruning_profile_validated(tree, data, edge_Q, workspace)
end

function _mvbm_edge_q_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_Q::Array{Float64, 3},
    Sigma_out;
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
)
    data = trait isa Matrix{Float64} ? trait : _validate_multivariate_trait(tree, trait)
    prof = _mv_fixed_root_pruning_profile_validated(tree, data, edge_Q, workspace)
    return (success = prof.success, loglik = prof.loglik, theta = prof.theta, sigma = Sigma_out)
end

