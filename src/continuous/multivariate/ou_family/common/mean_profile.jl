mutable struct MVOUFixedMeanWorkspace
    designs::Vector{Matrix{Float64}}
    design_work::Matrix{Float64}
    transition_work::Matrix{Float64}
    cached_A::Matrix{Float64}
    cached_A_regimes::Array{Float64,3}
    design_valid::Bool
    edge_Phi::Array{Float64,3}
    edge_Q::Array{Float64,3}
    edge_Qinv::Array{Float64,3}
    edge_logdet_Q::Vector{Float64}
    node_factor::Vector{Matrix{Float64}}
    node_values::Vector{Matrix{Float64}}
    node_nrows::Vector{Int}
    stack_factor::Matrix{Float64}
    stack_values::Matrix{Float64}
    covariance::Matrix{Float64}
    product::Matrix{Float64}
    design::Matrix{Float64}
    response::Vector{Float64}
    centered::Matrix{Float64}
    likelihood::MVProfileWorkspace
end

function _mvou_fixed_mean_workspace(tree::CompactTree, p::Integer, q::Integer; nregimes::Integer)
    likelihood = _mv_profile_workspace(tree, p)
    return MVOUFixedMeanWorkspace(
        [zeros(p, q) for _ in 1:tree.nnodes], zeros(p, q), zeros(p, p),
        zeros(p, p), zeros(p, p, nregimes), false,
        zeros(p, p, tree.nedges), likelihood.edge_Q,
        likelihood.edge_Qinv, likelihood.edge_logdet_Q,
        [zeros(p, p) for _ in 1:tree.nnodes],
        [zeros(p, q + 1) for _ in 1:tree.nnodes], zeros(Int, tree.nnodes),
        zeros(2p, p), zeros(2p, q + 1), zeros(p, p), zeros(p, p),
        zeros(tree.ntips * p, q), zeros(tree.ntips * p), zeros(tree.ntips, p),
        likelihood,
    )
end

function _mvou_fixed_mean_branches!(ws::MVOUFixedMeanWorkspace, bundle, precalc, spec)
    args = (ws.edge_Phi, ws.edge_Q, ws.edge_Qinv, ws.edge_logdet_Q, precalc)
    if spec.model in (:mvOU1, :mvOUM)
        return _mvou_branch_cache!(args..., bundle.A, bundle.Sigma; chol_work = ws.covariance)
    elseif spec.model === :mvOUMV
        return _mvoumv_branch_cache!(args..., bundle.A, bundle.Sigma_regimes; chol_work = ws.covariance)
    elseif spec.model === :mvOUMA
        return _mvouma_branch_cache!(args..., bundle.A_regimes, bundle.Sigma; chol_work = ws.covariance)
    end
    return _mvoumva_branch_cache!(args..., bundle.A_regimes, bundle.Sigma_regimes; chol_work = ws.covariance)
end

function _mvou_fixed_mean_designs!(ws::MVOUFixedMeanWorkspace, tree, bundle, precalc)
    shared = isempty(bundle.A_regimes)
    if ws.design_valid && isequal(ws.cached_A, bundle.A) &&
        (shared || isequal(ws.cached_A_regimes, bundle.A_regimes))
        return ws.designs
    end
    _mvou_mean_designs!(ws.designs, ws.design_work, ws.transition_work, tree, bundle, precalc)
    if !shared
        copyto!(ws.cached_A_regimes, bundle.A_regimes)
    end
    copyto!(ws.cached_A, bundle.A)
    ws.design_valid = true
    return ws.designs
end

function _mvou_mean_emit_rows!(ws::MVOUFixedMeanWorkspace, values, first_row::Int)
    for row in axes(values, 1)
        target = first_row + row
        ws.response[target] = values[row, 1]
        @views ws.design[target, :] .= values[row, 2:end]
    end
    return first_row + size(values, 1)
end

# Orthogonal elimination whitens observations and mean columns together. Node
# factors have at most p rows; no tip covariance or normal equations are formed.
function _mvou_whiten_fixed_mean!(ws::MVOUFixedMeanWorkspace, tree, data, branch, root_cov)
    p = size(data, 2)
    fill!(ws.node_nrows, 0)
    nrows = 0
    for node in tree.postorder_internal
        count = 0
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            child = Int(tree.child_of_edge[edge])
            Phi, Q = @view(branch.Phi[:, :, edge]), @view(branch.Q[:, :, edge])
            if tree.is_tip[child]
                tip = ws.likelihood.tip_index[child]
                observed = findall(!isnan, @view(data[tip, :]))
                r = length(observed)
                r == 0 && continue
                F = @view ws.stack_factor[(count + 1):(count + r), :]
                values = @view ws.stack_values[(count + 1):(count + r), :]
                C = @view ws.covariance[1:r, 1:r]
                for (i, trait) in enumerate(observed)
                    F[i, :] .= @view Phi[trait, :]
                    values[i, 1] = data[tip, trait]
                    values[i, 2:end] .= @view ws.designs[child][trait, :]
                    for (j, other) in enumerate(observed)
                        C[i, j] = Q[trait, other]
                    end
                end
            else
                r = ws.node_nrows[child]
                r == 0 && continue
                R = @view ws.node_factor[child][1:r, :]
                F = @view ws.stack_factor[(count + 1):(count + r), :]
                values = @view ws.stack_values[(count + 1):(count + r), :]
                C = @view ws.covariance[1:r, 1:r]
                product = @view ws.product[1:r, :]
                mul!(product, R, Q)
                mul!(C, product, transpose(R))
                for i in 1:r
                    C[i, i] += 1.0
                end
                mul!(F, R, Phi)
                values .= @view ws.node_values[child][1:r, :]
            end
            factor = cholesky!(Symmetric(C))
            ldiv!(factor.L, F)
            ldiv!(factor.L, values)
            count += r
        end
        count == 0 && continue
        F = @view ws.stack_factor[1:count, :]
        values = @view ws.stack_values[1:count, :]
        factor = qr!(F)
        lmul!(adjoint(factor.Q), values)
        retained = min(count, p)
        ws.node_nrows[node] = retained
        @views ws.node_factor[node][1:retained, :] .= factor.R
        @views ws.node_values[node][1:retained, :] .= values[1:retained, :]
        nrows = _mvou_mean_emit_rows!(ws, @view(values[(retained + 1):count, :]), nrows)
    end
    root = Int(tree.root)
    r = ws.node_nrows[root]
    values = @view ws.node_values[root][1:r, :]
    if r > 0 && !iszero(root_cov)
        R = @view ws.node_factor[root][1:r, :]
        C, product = @view(ws.covariance[1:r, 1:r]), @view(ws.product[1:r, :])
        mul!(product, R, root_cov)
        mul!(C, product, transpose(R))
        for i in 1:r
            C[i, i] += 1.0
        end
        ldiv!(cholesky!(Symmetric(C)).L, values)
    end
    return _mvou_mean_emit_rows!(ws, values, nrows)
end

function _mvou_profile_fixed_mean(tree, data, bundle, precalc, spec,
    workspace::Union{Nothing,MVOUFixedMeanWorkspace} = nothing; diagnostics::Bool = false)
    p = size(data, 2)
    q = _mvou_mean_ncoefficients(spec, p, precalc.nregimes)
    ws = workspace === nothing ? _mvou_fixed_mean_workspace(tree, p, q; nregimes = precalc.nregimes) : workspace
    branch = _mvou_fixed_mean_branches!(ws, bundle, precalc, spec)
    _mvou_fixed_mean_designs!(ws, tree, bundle, precalc)
    root_cov = _mvou_root_covariance(precalc, bundle)
    nrows = _mvou_whiten_fixed_mean!(ws, tree, data, branch, root_cov)
    design, response = @view(ws.design[1:nrows, :]), @view(ws.response[1:nrows])
    if nrows == 0
        decomposition = svd(zeros(0, q); full = false)
    else
        factor = qr!(design)
        lmul!(adjoint(factor.Q), response)
        decomposition = svd(Matrix(factor.R); full = false)
    end
    spectrum = decomposition.S
    threshold = isempty(spectrum) ? 0.0 : sqrt(eps(Float64)) * first(spectrum)
    rank = count(>(threshold), spectrum)
    coefficients = zeros(q)
    for j in 1:rank
        projection = dot(@view(decomposition.U[:, j]), @view(response[1:size(decomposition.U, 1)])) / spectrum[j]
        @views coefficients .+= projection .* decomposition.Vt[j, :]
    end
    for (i, tip) in enumerate(tree.tip_ids)
        @views mul!(ws.centered[i, :], ws.designs[Int(tip)], coefficients)
        @views ws.centered[i, :] .= data[i, :] .- ws.centered[i, :]
    end
    profile = _mv_recursive_profile(tree, ws.centered, branch.Phi, branch.Q;
        root_mean = zeros(p), root_cov = root_cov, workspace = ws.likelihood,
        edge_Qinv = branch.Qinv, edge_logdet_Q = branch.logdet_Q)
    info = diagnostics ? OUMeanDiagnostics(rank = rank, ncoefficients = q,
        singular_values = copy(spectrum),
        identifiable = [sum(abs2, @view(decomposition.Vt[1:rank, j])) >= 1.0 - 1e-8 for j in 1:q],
        condition_number = rank == q ? first(spectrum) / last(spectrum) : Inf) : nothing
    return merge(profile, (theta = coefficients[1:(p * precalc.nregimes)],
        theta0 = nothing,
        mean_diagnostics = info))
end
