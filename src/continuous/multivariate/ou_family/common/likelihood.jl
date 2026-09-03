function _mvou_edge_predict_to_child(
    parent_mean::AbstractVector{<:Real},
    parent_cov::AbstractMatrix{<:Real},
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
)
    mean = Matrix{Float64}(Phi) * Vector{Float64}(parent_mean)
    cov = Matrix{Float64}(Phi) * Matrix{Float64}(parent_cov) * Matrix{Float64}(Phi)' + Matrix{Float64}(Q)
    return (mean = mean, cov = (cov + cov') / 2)
end

function _mvou_observation_info_to_parent(
    y::AbstractVector{<:Real},
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    obs = Vector{Float64}(y)
    F_full = Matrix{Float64}(Phi)
    p = size(F_full, 2)
    obs_idx = findall(!isnan, obs)
    isempty(obs_idx) && return (
        success = true,
        precision = zeros(Float64, p, p),
        linear = zeros(Float64, p),
        logconst = 0.0,
    )
    yobs = obs[obs_idx]
    F = F_full[obs_idx, :]
    try
        Qinv_y, Qinv_F, logdet_Q =
            if length(obs_idx) == p && Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
                (Qinv_cached * yobs, Qinv_cached * F, Float64(logdet_Q_cached))
            else
                Qobs = Matrix{Float64}(Q)[obs_idx, obs_idx]
                cholQ = _mvou_cholesky_psd(Qobs)
                (cholQ \ yobs, cholQ \ F, 2.0 * sum(log, diag(cholQ.L)))
            end
        nobs = length(yobs)
        return (
            success = true,
            precision = F' * Qinv_F,
            linear = F' * Qinv_y,
            logconst = -0.5 * (dot(yobs, Qinv_y) + logdet_Q + nobs * log(2 * pi)),
        )
    catch
        return (success = false, precision = zeros(Float64, p, p), linear = zeros(Float64, p), logconst = -Inf)
    end
end

function _mvou_info_to_parent(
    child_precision::AbstractMatrix{<:Real},
    child_linear::AbstractVector{<:Real},
    child_logconst::Real,
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    Pchild = Matrix{Float64}(child_precision)
    hchild = Vector{Float64}(child_linear)
    F = Matrix{Float64}(Phi)
    try
        p = length(hchild)
        Qinv, logdet_Q =
            Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached)) ?
            (Qinv_cached, Float64(logdet_Q_cached)) :
            _mvou_qinv_logdet(Q)
        Qinv_F = Qinv * F
        S = Pchild + Qinv
        cholS = _mvou_cholesky_psd(S)
        Sinv_h = cholS \ hchild
        Sinv_Qinv_F = cholS \ Qinv_F
        logdet_S = 2.0 * sum(log, diag(cholS.L))
        precision = F' * Qinv_F - Qinv_F' * Sinv_Qinv_F
        linear = Qinv_F' * Sinv_h
        logconst = Float64(child_logconst) - 0.5 * (logdet_Q + logdet_S) + 0.5 * dot(hchild, Sinv_h)
        return (
            success = true,
            precision = (precision + precision') / 2,
            linear = linear,
            logconst = logconst,
        )
    catch
        p = length(hchild)
        return (success = false, precision = zeros(Float64, p, p), linear = zeros(Float64, p), logconst = -Inf)
    end
end

function _mvou_root_info_loglik(
    precision::AbstractMatrix{<:Real},
    linear::AbstractVector{<:Real},
    logconst::Real,
    root_mean::AbstractVector{<:Real},
    root_cov::AbstractMatrix{<:Real},
)
    P = Matrix{Float64}(precision)
    h = Vector{Float64}(linear)
    m = Vector{Float64}(root_mean)
    if _mvou_is_zero_cov(root_cov)
        return Float64(logconst) - 0.5 * dot(m, P * m) + dot(h, m)
    end

    C = Matrix{Float64}(root_cov)
    try
        # Whiten the Gaussian root prior before integrating the root state.
        # This keeps the reduced precision symmetric even when P and C do
        # not commute, which is essential for correlated multivariate roots.
        cholC = _mvou_cholesky_psd(C)
        L = Matrix{Float64}(cholC.L)
        d = transpose(L) * (h - P * m)
        whitened_precision = Matrix{Float64}(I, length(h), length(h)) + transpose(L) * P * L
        chol_whitened = _mvou_cholesky_psd(whitened_precision)
        solved_d = chol_whitened \ d
        logdet_whitened = 2.0 * sum(log, diag(chol_whitened.L))
        base = Float64(logconst) + dot(h, m) - 0.5 * dot(m, P * m)
        return base - 0.5 * logdet_whitened + 0.5 * dot(d, solved_d)
    catch
        return -Inf
    end
end

function _mv_recursive_profile(
    tree::CompactTree,
    centered_trait::AbstractMatrix{<:Real},
    edge_Phi::Array{Float64, 3},
    edge_Q::Array{Float64, 3};
    root_mean::AbstractVector{<:Real},
    root_cov::AbstractMatrix{<:Real},
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
    edge_Qinv::Union{Nothing, Array{Float64, 3}} = nothing,
    edge_logdet_Q::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    data = Matrix{Float64}(centered_trait)
    _validate_binary_tree(tree)
    p = size(centered_trait, 2)
    ws = workspace === nothing ? _mv_profile_workspace(tree, p) : workspace
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
            qinv = edge_Qinv === nothing ? nothing : @view(edge_Qinv[:, :, edge])
            logdet_q = edge_logdet_Q === nothing ? NaN : edge_logdet_Q[edge]
            msg =
                if tree.is_tip[child]
                    _mvou_observation_info_to_parent(data[tip_index[child], :], edge_Phi[:, :, edge], edge_Q[:, :, edge], qinv, logdet_q)
                else
                    _mvou_info_to_parent(precision[child], linear[child], logconst[child], edge_Phi[:, :, edge], edge_Q[:, :, edge], qinv, logdet_q)
                end
            msg.success || return (success = false, loglik = -Inf)
            precision[node] .+= msg.precision
            linear[node] .+= msg.linear
            logconst[node] += msg.logconst
        end
    end

    root = Int(tree.root)
    root_loglik = _mvou_root_info_loglik(precision[root], linear[root], logconst[root], root_mean, root_cov)
    isfinite(root_loglik) || return (success = false, loglik = -Inf)
    return (success = true, loglik = root_loglik)
end


function _mv_recursive_asr(
    tree::CompactTree,
    centered_trait::AbstractMatrix{<:Real},
    edge_Phi::Array{Float64, 3},
    edge_Q::Array{Float64, 3};
    root_mean::AbstractVector{<:Real},
    root_cov::AbstractMatrix{<:Real},
    shift::AbstractVector{<:Real},
    model::Symbol,
)
    p = size(centered_trait, 2)
    post = _mv_recursive_posterior_cache(
        tree,
        centered_trait,
        edge_Phi,
        edge_Q;
        root_mean = root_mean,
        root_cov = root_cov,
    )
    post.success || return MVContinuousASRResult(model = model, success = false)

    internal_ids = Int32[node for node in 1:tree.nnodes if !tree.is_tip[node]]
    nint = length(internal_ids)
    estimates = Matrix{Float64}(undef, nint, p)
    node_covariances = Array{Float64, 3}(undef, nint, p, p)
    shift_vec = Vector{Float64}(shift)
    for (i, node) in enumerate(internal_ids)
        estimates[i, :] .= post.full_mean[node] .+ shift_vec
        node_covariances[i, :, :] .= post.full_cov[node]
    end
    all_node_estimates = Matrix{Float64}(undef, tree.nnodes, p)
    all_node_covariances = Array{Float64, 3}(undef, tree.nnodes, p, p)
    for node in 1:tree.nnodes
        all_node_estimates[node, :] .= post.full_mean[node] .+ shift_vec
        all_node_covariances[node, :, :] .= post.full_cov[node]
    end
    tip_ids = Int32.(tree.tip_ids)
    tip_estimates = Matrix{Float64}(undef, tree.ntips, p)
    tip_covariances = Array{Float64, 3}(undef, tree.ntips, p, p)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_estimates[i, :] .= all_node_estimates[Int(tip), :]
        tip_covariances[i, :, :] .= all_node_covariances[Int(tip), :, :]
    end
    node_time_from_root = tree.dist_from_root[internal_ids]
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])
    node_time_before_present = tree_height .- node_time_from_root
    return MVContinuousASRResult(
        model = model,
        success = true,
        node_ids = internal_ids,
        node_labels = tree.node_labels[internal_ids],
        time_from_root = node_time_from_root,
        time_before_present = node_time_before_present,
        estimates = estimates,
        node_covariances = node_covariances,
        all_node_estimates = all_node_estimates,
        all_node_covariances = all_node_covariances,
        tip_ids = tip_ids,
        tip_labels = tree.node_labels[tip_ids],
        tip_estimates = tip_estimates,
        tip_covariances = tip_covariances,
        tip_observed_mask = .!isnan.(Matrix{Float64}(centered_trait)),
    )
end


function _mv_recursive_posterior_cache(
    tree::CompactTree,
    data::AbstractMatrix{<:Real},
    edge_Phi::Array{Float64, 3},
    edge_Q::Array{Float64, 3};
    root_mean::AbstractVector{<:Real},
    root_cov::AbstractMatrix{<:Real},
)
    y = Matrix{Float64}(data)
    p = size(y, 2)
    precision = [zeros(Float64, p, p) for _ in 1:tree.nnodes]
    linear = [zeros(Float64, p) for _ in 1:tree.nnodes]
    logconst = zeros(Float64, tree.nnodes)
    outside_mean = [zeros(Float64, p) for _ in 1:tree.nnodes]
    outside_cov = [zeros(Float64, p, p) for _ in 1:tree.nnodes]
    full_mean = [zeros(Float64, p) for _ in 1:tree.nnodes]
    full_cov = [zeros(Float64, p, p) for _ in 1:tree.nnodes]
    desc_mean = [zeros(Float64, p) for _ in 1:tree.nnodes]
    desc_cov = [zeros(Float64, p, p) for _ in 1:tree.nnodes]

    tip_index = zeros(Int, tree.nnodes)
    for (i, tip) in enumerate(tree.tip_ids)
        tip_index[tip] = i
    end

    upward = Vector{NamedTuple}(undef, tree.nedges)
    edge_context_mean = [zeros(Float64, p) for _ in 1:tree.nedges]
    edge_context_cov = [zeros(Float64, p, p) for _ in 1:tree.nedges]

    local function _condition_tip(mean::Vector{Float64}, cov::Matrix{Float64}, obs::Vector{Float64})
        obs_idx = findall(!isnan, obs)
        isempty(obs_idx) && return (success = true, mean = copy(mean), cov = copy(cov))
        miss_idx = setdiff(collect(1:p), obs_idx)
        if isempty(miss_idx)
            return (success = true, mean = obs, cov = zeros(Float64, p, p))
        end
        Coo = cov[obs_idx, obs_idx]
        try
            chol = _mvou_cholesky_psd(Coo)
            delta = obs[obs_idx] - mean[obs_idx]
            cond_mean = copy(mean)
            cond_cov = zeros(Float64, p, p)
            cond_mean[obs_idx] .= obs[obs_idx]
            Cmo = cov[miss_idx, obs_idx]
            Cmm = cov[miss_idx, miss_idx]
            gain_delta = Cmo * (chol \ delta)
            cond_mean[miss_idx] .= mean[miss_idx] .+ gain_delta
            Ccond = Cmm - Cmo * (chol \ cov[obs_idx, miss_idx])
            cond_cov[miss_idx, miss_idx] .= (Ccond + Ccond') / 2
            return (success = true, mean = cond_mean, cov = cond_cov)
        catch
            return (success = false, mean = zeros(Float64, p), cov = zeros(Float64, p, p))
        end
    end

    for node in tree.postorder_internal
        fill!(precision[node], 0.0)
        fill!(linear[node], 0.0)
        logconst[node] = 0.0
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = tree.child_of_edge[edge]
            msg =
                if tree.is_tip[child]
                    _mvou_observation_info_to_parent(y[tip_index[child], :], edge_Phi[:, :, edge], edge_Q[:, :, edge])
                else
                    _mvou_info_to_parent(precision[child], linear[child], logconst[child], edge_Phi[:, :, edge], edge_Q[:, :, edge])
                end
            msg.success || return (success = false,)
            upward[edge] = msg
            precision[node] .+= msg.precision
            linear[node] .+= msg.linear
            logconst[node] += msg.logconst
        end
    end

    root = Int(tree.root)
    outside_mean[root] .= root_mean
    outside_cov[root] .= root_cov

    for node in tree.preorder
        full =
            if tree.is_tip[node]
                _condition_tip(outside_mean[node], outside_cov[node], y[tip_index[node], :])
            else
                _mvou_context_with_info(outside_mean[node], outside_cov[node], precision[node], linear[node])
            end
        full.success || return (success = false,)
        full_mean[node] .= full.mean
        full_cov[node] .= full.cov
        if !tree.is_tip[node]
            desc = _mvou_context_with_info(zeros(Float64, p), 1.0e12 * Matrix{Float64}(I, p, p), precision[node], linear[node])
            desc_mean[node] .= desc.mean
            desc_cov[node] .= desc.cov
        else
            desc_mean[node] .= full.mean
            desc_cov[node] .= full.cov
        end

        tree.is_tip[node] && continue

        child1 = tree.children[node][1]
        child2 = tree.children[node][2]
        edge1 = tree.first_child_edge[node]
        edge2 = tree.last_child_edge[node]
        sib1 = upward[edge2]
        sib2 = upward[edge1]

        ctx1 = _mvou_context_with_info(outside_mean[node], outside_cov[node], sib1.precision, sib1.linear)
        ctx2 = _mvou_context_with_info(outside_mean[node], outside_cov[node], sib2.precision, sib2.linear)
        ctx1.success || return (success = false,)
        ctx2.success || return (success = false,)
        edge_context_mean[edge1] .= ctx1.mean
        edge_context_cov[edge1] .= ctx1.cov
        edge_context_mean[edge2] .= ctx2.mean
        edge_context_cov[edge2] .= ctx2.cov

        pred1 = _mvou_edge_predict_to_child(ctx1.mean, ctx1.cov, edge_Phi[:, :, edge1], edge_Q[:, :, edge1])
        pred2 = _mvou_edge_predict_to_child(ctx2.mean, ctx2.cov, edge_Phi[:, :, edge2], edge_Q[:, :, edge2])
        outside_mean[child1] .= pred1.mean
        outside_cov[child1] .= pred1.cov
        outside_mean[child2] .= pred2.mean
        outside_cov[child2] .= pred2.cov
    end

    return (
        success = true,
        desc_mean = desc_mean,
        desc_cov = desc_cov,
        outside_mean = outside_mean,
        outside_cov = outside_cov,
        full_mean = full_mean,
        full_cov = full_cov,
        precision = precision,
        linear = linear,
        logconst = logconst,
        upward = upward,
        edge_context_mean = edge_context_mean,
        edge_context_cov = edge_context_cov,
    )
end



function _mvou_copy_edge_matrix!(
    dest::AbstractMatrix{Float64},
    src::Array{Float64, 3},
    edge::Integer,
)
    p = size(dest, 1)
    size(dest, 2) >= p || throw(ArgumentError("edge matrix destination has incompatible dimensions"))
    @inbounds for j in 1:p, i in 1:p
        dest[i, j] = src[i, j, edge]
    end
    return dest
end

function _mvou_profile_theta_recursive(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_Phi::Array{Float64, 3},
    edge_Q::Array{Float64, 3},
    node_designs::Vector{Matrix{Float64}},
    q::Integer;
    workspace::Union{Nothing, MVProfileWorkspace, MVOUThetaProfileWorkspace} = nothing,
    edge_Qinv::Union{Nothing, Array{Float64, 3}} = nothing,
    edge_logdet_Q::Union{Nothing, AbstractVector{<:Real}} = nothing,
    copy_theta::Bool = true,
    root_cov::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)
    data = trait isa Matrix{Float64} ? trait : Matrix{Float64}(trait)
    _validate_binary_tree(tree)
    p = size(data, 2)
    m = p + Int(q)
    ws = workspace === nothing ? _mvou_theta_profile_workspace(tree, p, q) : workspace
    size(ws.precision[1], 1) == m || throw(ArgumentError("workspace dimension must be p + q"))
    precision = ws.precision
    linear = ws.linear
    logconst = ws.logconst
    tip_index = ws.tip_index

    for node in tree.postorder_internal
        fill!(precision[node], 0.0)
        fill!(linear[node], 0.0)
        logconst[node] = 0.0
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = Int(tree.child_of_edge[edge])
            logdet_q = edge_logdet_Q === nothing ? NaN : edge_logdet_Q[edge]
            msg =
                if ws isa MVOUThetaProfileWorkspace
                    success, msg_logconst = tree.is_tip[child] ?
                    _mvou_profile_theta_tip_message!(
                        ws,
                        @view(data[tip_index[child], :]),
                        _mvou_copy_edge_matrix!(ws.transition_work, edge_Phi, edge),
                        _mvou_copy_edge_matrix!(ws.Qobs, edge_Q, edge),
                        node_designs[child],
                        edge_Qinv === nothing ? nothing : _mvou_copy_edge_matrix!(ws.Qinv, edge_Qinv, edge),
                        logdet_q,
                    ) :
                    _mvou_profile_theta_internal_message!(
                        ws,
                        precision[child],
                        linear[child],
                        logconst[child],
                        _mvou_copy_edge_matrix!(ws.transition_work, edge_Phi, edge),
                        _mvou_copy_edge_matrix!(ws.Qobs, edge_Q, edge),
                        p,
                        Int(q),
                        edge_Qinv === nothing ? nothing : _mvou_copy_edge_matrix!(ws.Qinv, edge_Qinv, edge),
                        logdet_q,
                    )
                    success || return (success = false, loglik = -Inf, theta = ws.theta)
                    precision[node] .+= ws.msg_precision
                    linear[node] .+= ws.msg_linear
                    logconst[node] += msg_logconst
                    continue
                else
                    qinv = edge_Qinv === nothing ? nothing : @view(edge_Qinv[:, :, edge])
                    tree.is_tip[child] ?
                    _mvou_profile_theta_tip_message(
                        data[tip_index[child], :],
                        edge_Phi[:, :, edge],
                        edge_Q[:, :, edge],
                        node_designs[child],
                        qinv,
                        logdet_q,
                    ) :
                    _mvou_profile_theta_internal_message(
                        precision[child],
                        linear[child],
                        logconst[child],
                        edge_Phi[:, :, edge],
                        edge_Q[:, :, edge],
                        p,
                        Int(q),
                        qinv,
                        logdet_q,
                    )
                end
            msg.success || return (success = false, loglik = -Inf, theta = ws isa MVOUThetaProfileWorkspace ? ws.theta : zeros(Float64, q))
            precision[node] .+= msg.precision
            linear[node] .+= msg.linear
            logconst[node] += msg.logconst
        end
    end

    root = Int(tree.root)
    P胃 = @view precision[root][(p + 1):m, (p + 1):m]
    h胃 = @view linear[root][(p + 1):m]
    try
        chol = _mvou_cholesky_psd(P胃)
        theta_hat = ws isa MVOUThetaProfileWorkspace ? ws.theta : zeros(Float64, q)
        theta_hat .= h胃
        ldiv!(theta_hat, chol, theta_hat)
        loglik = logconst[root] + 0.5 * dot(h胃, theta_hat)
        isfinite(loglik) || return (success = false, loglik = -Inf, theta = ws isa MVOUThetaProfileWorkspace ? ws.theta : zeros(Float64, q))
        return (success = true, loglik = loglik, theta = copy_theta ? copy(theta_hat) : theta_hat)
    catch
        return (success = false, loglik = -Inf, theta = ws isa MVOUThetaProfileWorkspace ? ws.theta : zeros(Float64, q))
    end
end

function _mvou_profile_theta_tree_pruning_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    bundle::MVOUParameterBundle,
    precalc::MVOUPrecalc,
    spec::MVOUSpec,
    workspace::Union{Nothing, MVProfileWorkspace, MVOUThetaProfileWorkspace} = nothing,
)
    data = trait isa Matrix{Float64} ? trait : _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    q = p * precalc.nregimes
    ws = workspace isa MVOUThetaProfileWorkspace ? workspace : nothing
    same_shared_A =
        ws !== nothing &&
        ws.has_cached_A &&
        size(ws.cached_A) == size(bundle.A) &&
        maximum(abs.(ws.cached_A .- bundle.A)) <= 1e-12

    function _cache_shared_A!()
        ws === nothing && return
        copyto!(ws.cached_A, bundle.A)
        ws.has_cached_A = true
    end

    if spec.model === :mvOU1
        branch =
            ws === nothing ?
            _mvou_branch_cache(precalc, bundle.A, bundle.Sigma) :
            _mvou_branch_cache!(
                ws.edge_Phi,
                ws.edge_Q,
                ws.edge_Qinv,
                ws.edge_logdet_Q,
                precalc,
                bundle.A,
                bundle.Sigma;
                chol_work = ws.edge_chol_work,
                reuse_phi = same_shared_A && ws.shared_phi_valid,
            )
        designs = ws === nothing ? _mvou_node_design_matrices(tree, bundle.A, [SimmapSegment[] for _ in 1:length(precalc.branch_lengths)], 1, precalc.A_decomp) : ws.identity_designs
        if ws !== nothing
            for W in designs
                fill!(W, 0.0)
                @inbounds for i in 1:p
                    W[i, i] = 1.0
                end
            end
            _cache_shared_A!()
            ws.shared_phi_valid = true
        else
            for W in designs
                fill!(W, 0.0)
                @inbounds for i in 1:p
                    W[i, i] = 1.0
                end
            end
        end
        return _mvou_profile_theta_recursive(
            tree,
            data,
            branch.Phi,
            branch.Q,
            designs,
            p;
            workspace = workspace,
            edge_Qinv = branch.Qinv,
            edge_logdet_Q = branch.logdet_Q,
            copy_theta = ws === nothing,
        )
    elseif spec.model === :mvOUM
        branch =
            ws === nothing ?
            _mvou_branch_cache(precalc, bundle.A, bundle.Sigma) :
            _mvou_branch_cache!(
                ws.edge_Phi,
                ws.edge_Q,
                ws.edge_Qinv,
                ws.edge_logdet_Q,
                precalc,
                bundle.A,
                bundle.Sigma;
                chol_work = ws.edge_chol_work,
                reuse_phi = same_shared_A && ws.shared_phi_valid,
            )
        designs =
            ws === nothing ?
            _mvou_node_design_matrices(tree, bundle.A, precalc.edge_segments, precalc.nregimes, precalc.A_decomp) :
            (same_shared_A && ws.shared_design_valid ?
             ws.regime_designs :
             _mvou_node_design_matrices!(
                 ws.regime_designs,
                 ws.design_work,
                 ws.transition_work,
                 tree,
                 bundle.A,
                 precalc.edge_segments,
                 precalc.nregimes,
                 precalc.A_decomp,
             ))
        if ws !== nothing
            _cache_shared_A!()
            ws.shared_phi_valid = true
            ws.shared_design_valid = true
        end
        return _mvou_profile_theta_recursive(
            tree,
            data,
            branch.Phi,
            branch.Q,
            designs,
            q;
            workspace = workspace,
            edge_Qinv = branch.Qinv,
            edge_logdet_Q = branch.logdet_Q,
            copy_theta = ws === nothing,
        )
    elseif spec.model === :mvOUMV
        branch =
            ws === nothing ?
            _mvoumv_branch_cache(precalc, bundle.A, bundle.Sigma_regimes) :
            _mvoumv_branch_cache!(
                ws.edge_Phi,
                ws.edge_Q,
                ws.edge_Qinv,
                ws.edge_logdet_Q,
                precalc,
                bundle.A,
                bundle.Sigma_regimes;
                chol_work = ws.edge_chol_work,
            )
        designs =
            ws === nothing ?
            _mvou_node_design_matrices(tree, bundle.A, precalc.edge_segments, precalc.nregimes, precalc.A_decomp) :
            (same_shared_A && ws.shared_design_valid ?
             ws.regime_designs :
             _mvou_node_design_matrices!(
                 ws.regime_designs,
                 ws.design_work,
                 ws.transition_work,
                 tree,
                 bundle.A,
                 precalc.edge_segments,
                 precalc.nregimes,
                 precalc.A_decomp,
             ))
        if ws !== nothing
            _cache_shared_A!()
            ws.shared_design_valid = true
        end
        return _mvou_profile_theta_recursive(
            tree,
            data,
            branch.Phi,
            branch.Q,
            designs,
            q;
            workspace = workspace,
            edge_Qinv = branch.Qinv,
            edge_logdet_Q = branch.logdet_Q,
            copy_theta = ws === nothing,
        )
    elseif spec.model === :mvOUMA
        branch =
            ws === nothing ?
            _mvouma_branch_cache(precalc, bundle.A_regimes, bundle.Sigma) :
            _mvouma_branch_cache!(
                ws.edge_Phi,
                ws.edge_Q,
                ws.edge_Qinv,
                ws.edge_logdet_Q,
                precalc,
                bundle.A_regimes,
                bundle.Sigma;
                chol_work = ws.edge_chol_work,
            )
        designs =
            ws === nothing ?
            _mvouma_node_design_matrices(tree, bundle.A_regimes, precalc.edge_segments, precalc.nregimes, precalc.A_decomp) :
            _mvouma_node_design_matrices!(
                ws.regime_designs,
                ws.design_work,
                ws.transition_work,
                tree,
                bundle.A_regimes,
                precalc.edge_segments,
                precalc.nregimes,
                precalc.A_decomp,
            )
        return _mvou_profile_theta_recursive(
            tree,
            data,
            branch.Phi,
            branch.Q,
            designs,
            q;
            workspace = workspace,
            edge_Qinv = branch.Qinv,
            edge_logdet_Q = branch.logdet_Q,
            copy_theta = ws === nothing,
        )
    elseif spec.model === :mvOUMVA
        branch =
            ws === nothing ?
            _mvoumva_branch_cache(precalc, bundle.A_regimes, bundle.Sigma_regimes) :
            _mvoumva_branch_cache!(
                ws.edge_Phi,
                ws.edge_Q,
                ws.edge_Qinv,
                ws.edge_logdet_Q,
                precalc,
                bundle.A_regimes,
                bundle.Sigma_regimes;
                chol_work = ws.edge_chol_work,
            )
        designs =
            ws === nothing ?
            _mvouma_node_design_matrices(tree, bundle.A_regimes, precalc.edge_segments, precalc.nregimes, precalc.A_decomp) :
            _mvouma_node_design_matrices!(
                ws.regime_designs,
                ws.design_work,
                ws.transition_work,
                tree,
                bundle.A_regimes,
                precalc.edge_segments,
                precalc.nregimes,
                precalc.A_decomp,
            )
        return _mvou_profile_theta_recursive(
            tree,
            data,
            branch.Phi,
            branch.Q,
            designs,
            q;
            workspace = workspace,
            edge_Qinv = branch.Qinv,
            edge_logdet_Q = branch.logdet_Q,
            copy_theta = ws === nothing,
        )
    end
    throw(ArgumentError("Unsupported profiled theta model $(spec.model)"))
end

function _mvou_profile_theta_tip_message!(
    ws::MVOUThetaProfileWorkspace,
    y::AbstractVector{<:Real},
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    design::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    p = size(Phi, 2)
    q = size(design, 2)
    m = p + q
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
        for col in 1:p
            ws.F[nobs, col] = Phi[row, col]
        end
        for col in 1:q
            ws.F[nobs, p + col] = design[row, col]
        end
    end
    nobs == 0 && return true, 0.0
    if nobs == p && Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
        try
            mul!(ws.Qinv_F, Qinv_cached, ws.F)
            mul!(ws.Qinv_y, Qinv_cached, ws.yobs)
            mul!(Pout, transpose(ws.F), ws.Qinv_F)
            mul!(hout, transpose(ws.F), ws.Qinv_y)
            return true, -0.5 * (dot(ws.yobs, ws.Qinv_y) + Float64(logdet_Q_cached) + p * log(2 * pi))
        catch
            return false, -Inf
        end
    end
    @inbounds for i in 1:nobs, j in 1:nobs
        ws.Qobs[i, j] = Q[ws.obs_index[i], ws.obs_index[j]]
    end
    try
        Fv = @view ws.F[1:nobs, 1:m]
        yv = @view ws.yobs[1:nobs]
        Qinv_F = @view ws.Qinv_F[1:nobs, 1:m]
        Qinv_y = @view ws.Qinv_y[1:nobs]
        logdet_Q =
            if nobs == p && Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
                mul!(Qinv_F, Qinv_cached, Fv)
                mul!(Qinv_y, Qinv_cached, yv)
                Float64(logdet_Q_cached)
            else
                cholQ = _mvou_cholesky_fast!(@view(ws.Qobs[1:nobs, 1:nobs]))
                copyto!(Qinv_F, Fv)
                copyto!(Qinv_y, yv)
                ldiv!(Qinv_F, cholQ, Qinv_F)
                ldiv!(Qinv_y, cholQ, Qinv_y)
                _mvou_logdet_chol(cholQ)
            end
        mul!(@view(Pout[1:m, 1:m]), transpose(Fv), Qinv_F)
        mul!(@view(hout[1:m]), transpose(Fv), Qinv_y)
        return true, -0.5 * (dot(yv, Qinv_y) + logdet_Q + nobs * log(2 * pi))
    catch
        return false, -Inf
    end
end

function _mvou_profile_theta_tip_message(
    y::AbstractVector{<:Real},
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    design::AbstractMatrix{<:Real},
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    obs = Vector{Float64}(y)
    obs_idx = findall(!isnan, obs)
    p = size(Phi, 2)
    q = size(design, 2)
    m = p + q
    isempty(obs_idx) && return (success = true, precision = zeros(Float64, m, m), linear = zeros(Float64, m), logconst = 0.0)
    F = hcat(Matrix{Float64}(Phi)[obs_idx, :], Matrix{Float64}(design)[obs_idx, :])
    yobs = obs[obs_idx]
    Qobs = Matrix{Float64}(Q)[obs_idx, obs_idx]
    try
        Qinv_F, Qinv_y, logdet_Q =
            if length(obs_idx) == p && Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
                (Qinv_cached * F, Qinv_cached * yobs, Float64(logdet_Q_cached))
            else
                cholQ = _mvou_cholesky_psd(Qobs)
                (cholQ \ F, cholQ \ yobs, 2.0 * sum(log, diag(cholQ.L)))
            end
        return (
            success = true,
            precision = F' * Qinv_F,
            linear = F' * Qinv_y,
            logconst = -0.5 * (dot(yobs, Qinv_y) + logdet_Q + length(yobs) * log(2 * pi)),
        )
    catch
        return (success = false, precision = zeros(Float64, m, m), linear = zeros(Float64, m), logconst = -Inf)
    end
end

function _mvou_profile_theta_internal_message!(
    ws::MVOUThetaProfileWorkspace,
    child_precision::AbstractMatrix{<:Real},
    child_linear::AbstractVector{<:Real},
    child_logconst::Real,
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    p::Integer,
    q::Integer,
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    p = Int(p)
    q = Int(q)
    m = p + q
    Pout = ws.msg_precision
    hout = ws.msg_linear
    fill!(Pout, 0.0)
    fill!(hout, 0.0)
    Pvv = @view child_precision[1:p, 1:p]
    Pv胃 = @view child_precision[1:p, (p + 1):m]
    P胃胃 = @view child_precision[(p + 1):m, (p + 1):m]
    hv = @view child_linear[1:p]
    h胃 = @view child_linear[(p + 1):m]
    try
        Qinv = ws.Qinv
        logdet_Q =
            if Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached))
                copyto!(Qinv, Qinv_cached)
                Float64(logdet_Q_cached)
            else
                _mvou_qinv_logdet!(Qinv, Q)
            end
        Qinv_Phi = ws.Qinv_Phi
        mul!(Qinv_Phi, Qinv, Phi)

        Avv = ws.Avv
        Avv .= Pvv
        Avv .+= Qinv

        Ava = @view ws.Ava[1:p, 1:m]
        @views Ava[:, 1:p] .= -Qinv_Phi
        @views Ava[:, (p + 1):m] .= Pv胃

        Aaa = @view ws.Aaa[1:m, 1:m]
        fill!(Aaa, 0.0)
        mul!(@view(Aaa[1:p, 1:p]), transpose(Phi), Qinv_Phi)
        @views Aaa[(p + 1):m, (p + 1):m] .= P胃胃

        ha = @view ws.ha[1:m]
        fill!(ha, 0.0)
        @views ha[(p + 1):m] .= h胃

        solve_Ava = @view ws.solve_Ava[1:p, 1:m]
        copyto!(solve_Ava, Ava)
        solve_hv = @view ws.solve_hv[1:p]
        copyto!(solve_hv, hv)
        logdet_Avv = _mvou_cholesky_lower_logdet_small!(Avv)
        _mvou_cholesky_solve_matrix_small!(solve_Ava, Avv, p, m)
        _mvou_cholesky_solve_vector_small!(solve_hv, Avv, p)

        mul!(@view(Pout[1:m, 1:m]), transpose(Ava), solve_Ava, -1.0, 0.0)
        @views Pout[1:m, 1:m] .+= Aaa
        @inbounds for j in 1:m
            for i in (j + 1):m
                v = 0.5 * (Pout[i, j] + Pout[j, i])
                Pout[i, j] = v
                Pout[j, i] = v
            end
        end
        mul!(@view(hout[1:m]), transpose(Ava), solve_hv, -1.0, 0.0)
        @views hout[1:m] .+= ha
        cparent = Float64(child_logconst) - 0.5 * logdet_Q - 0.5 * logdet_Avv + 0.5 * dot(hv, solve_hv)
        return true, cparent
    catch
        return false, -Inf
    end
end

function _mvou_profile_theta_internal_message(
    child_precision::AbstractMatrix{<:Real},
    child_linear::AbstractVector{<:Real},
    child_logconst::Real,
    Phi::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real},
    p::Integer,
    q::Integer,
    Qinv_cached::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    logdet_Q_cached::Real = NaN,
)
    p = Int(p)
    q = Int(q)
    m = p + q
    P = Matrix{Float64}(child_precision)
    h = Vector{Float64}(child_linear)
    Pvv = P[1:p, 1:p]
    Pv胃 = P[1:p, (p + 1):m]
    P胃胃 = P[(p + 1):m, (p + 1):m]
    hv = h[1:p]
    h胃 = h[(p + 1):m]
    try
        Qinv, logdet_Q =
            Qinv_cached !== nothing && isfinite(Float64(logdet_Q_cached)) ?
            (Qinv_cached, Float64(logdet_Q_cached)) :
            _mvou_qinv_logdet(Q)
        F = Matrix{Float64}(Phi)
        Avv = Pvv + Qinv
        Ava = hcat(-Qinv * F, Pv胃)
        Aaa = zeros(Float64, m, m)
        Aaa[1:p, 1:p] .= F' * Qinv * F
        Aaa[(p + 1):m, (p + 1):m] .= P胃胃
        ha = vcat(zeros(Float64, p), h胃)
        cholA = _mvou_cholesky_psd(Avv)
        Avv_inv_Ava = cholA \ Ava
        Avv_inv_hv = cholA \ hv
        Pparent = Aaa - Ava' * Avv_inv_Ava
        hparent = ha - Ava' * Avv_inv_hv
        logdet_Avv = 2.0 * sum(log, diag(cholA.L))
        cparent = Float64(child_logconst) - 0.5 * logdet_Q - 0.5 * logdet_Avv + 0.5 * dot(hv, Avv_inv_hv)
        return (
            success = true,
            precision = (Pparent + Pparent') / 2,
            linear = hparent,
            logconst = cparent,
        )
    catch
        return (success = false, precision = zeros(Float64, m, m), linear = zeros(Float64, m), logconst = -Inf)
    end
end

