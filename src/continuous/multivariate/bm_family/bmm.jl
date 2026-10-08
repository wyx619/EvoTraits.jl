struct MVBMCovarianceWorkspace
    means::Vector{Vector{Float64}}
    covariances::Vector{Matrix{Float64}}
    constants::Vector{Float64}
end

function _mvbm_covariance_workspace(tree, data)
    p = size(data, 2)
    complete = all(isfinite, data)
    return MVBMCovarianceWorkspace(
        complete ? [zeros(p) for _ in 1:tree.nnodes] : Vector{Float64}[],
        complete ? [zeros(p,p) for _ in 1:tree.nnodes] : Matrix{Float64}[],
        complete ? zeros(tree.nnodes) : Float64[])
end

function _mvbmm_profile(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    Sigmas::AbstractVector{<:AbstractMatrix},
    mapped_edge::AbstractMatrix{<:Real};
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
    return_sigma::Bool = true,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    nregimes = size(mapped, 2)
    length(Sigmas) == nregimes || throw(ArgumentError("Sigmas must match regime count"))
    p = size(data, 2)
    Sigma_mats =
        all(S -> S isa Matrix{Float64}, Sigmas) ?
        Sigmas :
        [Matrix{Float64}(S) for S in Sigmas]
    return _mvbmm_profile_validated(tree, data, Sigma_mats, mapped; workspace = workspace, return_sigma = return_sigma)
end

function _mvbmm_profile_validated(
    tree::CompactTree,
    data::Matrix{Float64},
    Sigma_mats::AbstractVector{<:AbstractMatrix{Float64}},
    mapped::AbstractMatrix{<:Real};
    workspace::Union{Nothing, MVProfileWorkspace} = nothing,
    covariance_workspace::Union{Nothing, MVBMCovarianceWorkspace} = nothing,
    return_sigma::Bool = true,
)
    p = size(data, 2)
    nregimes = size(mapped, 2)
    length(Sigma_mats) == nregimes || throw(ArgumentError("Sigmas must match regime count"))
    all(size(S, 1) == p && size(S, 2) == p for S in Sigma_mats) || throw(ArgumentError("All Sigmas must match trait dimension"))
    for S in Sigma_mats
        issymmetric(S) || return (success = false, loglik = -Inf, theta = zeros(p, nregimes))
        try
            cholesky(Symmetric(S))
        catch
            return (success = false, loglik = -Inf, theta = zeros(p, nregimes))
        end
    end

    sigma_array = return_sigma ? cat(Sigma_mats...; dims = 3) : nothing
    edge_Q = workspace === nothing ? Array{Float64, 3}(undef, p, p, tree.nedges) : workspace.edge_Q
    size(edge_Q) == (p, p, tree.nedges) || throw(ArgumentError("edge_Q workspace has incompatible dimensions"))
    @inbounds for e in 1:tree.nedges
        for j in 1:p, i in 1:p
            v = 0.0
            for r in 1:nregimes
                v += Float64(mapped[e, r]) * Sigma_mats[r][i, j]
            end
            edge_Q[i, j, e] = v
        end
        for j in 1:p, i in (j + 1):p
            v = 0.5 * (edge_Q[i, j, e] + edge_Q[j, i, e])
            edge_Q[i, j, e] = v
            edge_Q[j, i, e] = v
        end
    end
    prof = if all(isfinite, data)
        _mvbm_complete_covariance_profile(tree, data, edge_Q; workspace, covariance_workspace)
    else
        _mvbm_missing_covariance_profile(tree, data, edge_Q; workspace)
    end
    return (
        success = prof.success,
        loglik = prof.loglik,
        theta = reshape(prof.theta, 1, p),
        sigma = sigma_array,
    )
end

function _mvbm_merge_partial(left, right)
    isempty(left.indices) && return right
    isempty(right.indices) && return left
    common = intersect(left.indices, right.indices)
    unique_right = findall(i -> !(i in left.indices), right.indices)
    indices = vcat(left.indices, right.indices[unique_right])
    n1, n2 = length(left.indices), length(right.indices)
    # Joint independent noise factor and representative observations of the union.
    L = zeros(n1+n2, size(left.factor,2)+size(right.factor,2))
    L[1:n1,1:size(left.factor,2)] .= left.factor
    L[n1+1:end,size(left.factor,2)+1:end] .= right.factor
    representative = vcat(collect(1:n1), n1 .+ unique_right)
    R = L[representative,:]
    mean = vcat(left.mean, right.mean[unique_right])
    constant = left.constant + right.constant
    if isempty(common)
        return (indices=indices,mean=mean,factor=R,constant=constant)
    end
    i1 = [findfirst(==(i),left.indices) for i in common]
    i2 = [findfirst(==(i),right.indices) for i in common]
    D = L[n1 .+ i2,:] - L[i1,:]
    delta = right.mean[i2] - left.mean[i1]
    k = length(common)
    size(D,2) >= k || throw(PosDefException(k))
    decomposition = qr(Matrix(D'))
    T = UpperTriangular(Matrix(decomposition.R)[1:k,1:k])
    all(x -> isfinite(x) && x != 0, diag(T)) || throw(PosDefException(k))
    Q = decomposition.Q * Matrix{Float64}(I,size(D,2),size(D,2))
    whitened = T' \ delta
    mean -= (R * Q[:,1:k]) * whitened
    # Keep the orthogonal residual factor, not C - B*inv(V)*B'.
    residual = R * Q[:,k+1:end]
    constant -= 0.5 * (dot(whitened,whitened) + 2sum(log,abs.(diag(T))) + k*log(2pi))
    return (indices=indices,mean=mean,factor=residual,constant=constant)
end

function _mvbm_missing_covariance_profile(tree,data,edge_Q;workspace=nothing)
    p = size(data,2)
    empty_message = (indices=Int[],mean=Float64[],factor=zeros(0,0),constant=0.0)
    messages = [empty_message for _ in 1:tree.nnodes]
    for (i,tip) in enumerate(tree.tip_ids)
        observed = findall(!isnan,@view data[i,:])
        messages[tip] = (indices=observed,mean=data[i,observed],factor=zeros(length(observed),0),constant=0.0)
    end
    try
        for node in tree.postorder_internal
            combined = empty_message
            for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
                message = messages[tree.child_of_edge[edge]]
                isempty(message.indices) && continue
                covariance = edge_Q[message.indices,message.indices,edge]
                noise = iszero(covariance) ? zeros(length(message.indices),0) :
                    Matrix(cholesky(Symmetric(covariance)).L)
                combined_factor = hcat(message.factor,noise)
                factor = isempty(combined_factor) ? combined_factor :
                    Matrix(qr(Matrix(combined_factor')).R)'
                propagated = (indices=message.indices,mean=message.mean,factor=factor,constant=message.constant)
                combined = _mvbm_merge_partial(combined,propagated)
            end
            messages[node] = combined
        end
        root = messages[tree.root]
        length(root.indices)==p || throw(PosDefException(p))
        order = sortperm(root.indices)
        theta = root.mean[order]
        covariance = root.factor[order,:] * root.factor[order,:]'
        factor = cholesky(Symmetric(covariance))
        ll = root.constant - 0.5*(logdet(factor)+p*log(2pi))
        all(isfinite,theta) && isfinite(ll) || throw(PosDefException(p))
        if workspace !== nothing
            precision = factor \ Matrix{Float64}(I,p,p)
            workspace.precision[tree.root] .= precision
            workspace.linear[tree.root] .= precision*theta
            workspace.logconst[tree.root] = ll - 0.5*dot(theta,precision*theta)
        end
        return (success=true,loglik=ll,theta=theta)
    catch err
        err isa PosDefException || err isa SingularException || rethrow()
        return (success=false,loglik=-Inf,theta=zeros(p))
    end
end

# Covariance-form independent contrasts avoid subtracting two enormous Q inverses.
function _mvbm_complete_covariance_profile(tree, data, edge_Q; workspace = nothing, covariance_workspace = nothing)
    p = size(data, 2)
    if covariance_workspace !== nothing && isempty(covariance_workspace.means)
        append!(covariance_workspace.means, [zeros(p) for _ in 1:tree.nnodes])
        append!(covariance_workspace.covariances, [zeros(p,p) for _ in 1:tree.nnodes])
        append!(covariance_workspace.constants, zeros(tree.nnodes))
    end
    means, covariances, constants = if covariance_workspace === nothing
        ([zeros(p) for _ in 1:tree.nnodes], [zeros(p,p) for _ in 1:tree.nnodes], zeros(tree.nnodes))
    else
        (covariance_workspace.means, covariance_workspace.covariances, covariance_workspace.constants)
    end
    for (i, tip) in enumerate(tree.tip_ids)
        means[tip] .= @view data[i,:]
        fill!(covariances[tip], 0.0)
        constants[tip] = 0.0
    end
    try
        for node in tree.postorder_internal
            firstedge = tree.first_child_edge[node]
            child = tree.child_of_edge[firstedge]
            mean = copy(means[child])
            covariance = covariances[child] + @view(edge_Q[:,:,firstedge])
            constant = constants[child]
            for edge in firstedge+1:tree.last_child_edge[node]
                child = tree.child_of_edge[edge]
                other = covariances[child] + @view(edge_Q[:,:,edge])
                sumcov = covariance + other
                factor = cholesky(Symmetric(sumcov))
                delta = means[child] - mean
                solved = factor \ delta
                constant += constants[child] - 0.5 *
                    (dot(delta, solved) + logdet(factor) + p*log(2pi))
                mean += covariance * solved
                # Parallel sum: C1(C1+C2)^-1 C2, without an explicit precision inverse.
                covariance = covariance * (factor \ other)
                covariance = (covariance + covariance') / 2
            end
            means[node] .= mean
            covariances[node] .= covariance
            constants[node] = constant
        end
        root = Int(tree.root)
        factor = cholesky(Symmetric(covariances[root]))
        theta = means[root]
        ll = constants[root] - 0.5 * (logdet(factor) + p*log(2pi))
        all(isfinite,theta) && isfinite(ll) || return (success=false,loglik=-Inf,theta=zeros(p))
        if workspace !== nothing
            precision = factor \ Matrix{Float64}(I,p,p)
            workspace.precision[root] .= precision
            workspace.linear[root] .= precision * theta
            workspace.logconst[root] = ll - 0.5 * dot(theta,precision*theta)
        end
        return (success=true,loglik=ll,theta=copy(theta))
    catch err
        err isa PosDefException || rethrow()
        return (success=false,loglik=-Inf,theta=zeros(p))
    end
end

function _mvbmm_asr(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    fit::MVContinuousMultiBMResult,
    mapped_edge::AbstractMatrix{<:Real},
)
    data = _validate_multivariate_trait(tree, trait)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    fit.success || throw(ArgumentError("mvBMM ASR requires a successful fit"))
    p = fit.ntraits
    size(data, 2) == p || throw(ArgumentError("Trait dimension does not match fit"))
    nregimes = fit.nregimes
    edge_Phi = Array{Float64, 3}(undef, p, p, tree.nedges)
    edge_Q = Array{Float64, 3}(undef, p, p, tree.nedges)
    I_p = Matrix{Float64}(I, p, p)
    for e in 1:tree.nedges
        edge_Phi[:, :, e] .= I_p
        fill!(view(edge_Q, :, :, e), 0.0)
        for r in 1:nregimes
            edge_Q[:, :, e] .+= Float64(mapped[e, r]) .* fit.sigma[:, :, r]
        end
        edge_Q[:, :, e] .= (edge_Q[:, :, e] .+ edge_Q[:, :, e]') ./ 2
    end
    centered = data .- reshape(vec(fit.theta), 1, p)
    return _mv_recursive_asr(
        tree,
        centered,
        edge_Phi,
        edge_Q;
        root_mean = zeros(Float64, p),
        root_cov = zeros(Float64, p, p),
        shift = vec(fit.theta),
        model = :mvBMM,
    )
end

"""
    mvbmm_loglikelihood(tree, trait, mapped_edge, Sigmas)

Evaluate the multivariate multi-regime Brownian-motion (`mvBMM`) log-likelihood
for fixed regime-specific trait covariance matrices `Sigmas`. The root mean
vectors are profiled by generalized least squares.
"""
function mvbmm_loglikelihood(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    mapped_edge::AbstractMatrix{<:Real},
    Sigmas::AbstractVector{<:AbstractMatrix};
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    prof = _mvbmm_profile(tree, data, Sigmas, mapped_edge)
    p = size(data, 2)
    nregimes = size(mapped_edge, 2)
    nsigma = nregimes * (div(p * (p + 1), 2))
    nparams = nsigma + p
    return MVContinuousMultiBMResult(
        model = :mvBMM,
        backend = :tree_pruning,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, nregimes, "regime"),
        sigma = prof.success ? prof.sigma : zeros(p, p, nregimes),
        theta = prof.theta,
    )
end

"""
    fit_mvbmm(tree, trait, mapped_edge; kwargs...)

Fit the multivariate multi-regime Brownian-motion (`mvBMM`) model by maximum
likelihood using one trait covariance matrix per regime and profiled
regime-specific root mean vectors.
"""
function fit_mvbmm(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    mapped_edge::AbstractMatrix{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 1000,
    rel_tol::Float64 = 1e-6,
    guess_sigma::Union{Nothing, AbstractVector{<:AbstractMatrix}} = nothing,
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    _validate_binary_tree(tree)
    _validate_ultrametric_tree(tree)
    mapped = _validate_mapped_edge(tree, mapped_edge)
    nt, p = size(data)
    nregimes = size(mapped, 2)

    sigma0 =
        if guess_sigma === nothing
            empirical = _mv_complete_rows_cov(data)
            empirical = (empirical + empirical') / 2 + 1e-6 * Matrix{Float64}(I, p, p)
            [copy(empirical) for _ in 1:nregimes]
        else
            length(guess_sigma) == nregimes || throw(ArgumentError("guess_sigma must have one matrix per regime"))
            [Matrix{Float64}(S) for S in guess_sigma]
        end
    for S in sigma0
        size(S, 1) == p == size(S, 2) || throw(ArgumentError("All guess_sigma matrices must match trait dimension"))
        S .= (S + S') / 2
        S .+= 1e-8 * Matrix{Float64}(I, p, p)
    end
    p0 = _mvbm_pack_sigmas(sigma0)
    workspace = _mv_profile_workspace(tree, p)
    covariance_workspace = _mvbm_covariance_workspace(tree, data)
    sigma_workspace = [zeros(Float64, p, p) for _ in 1:nregimes]
    sigma_factor_work = zeros(Float64, p, p)

    objective = function (pars)
        _mvbm_unpack_sigmas!(sigma_workspace, pars, p, nregimes, sigma_factor_work)
        prof = _mvbmm_profile_validated(tree, data, sigma_workspace, mapped; workspace, covariance_workspace, return_sigma = false)
        return prof.success ? -prof.loglik : Inf
    end

    result = _mvbm_optimize_objective(
        objective,
        p0;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = _mvbm_sigma_lower_bounds(p, nregimes),
    )

    fitted_sigmas = _mvbm_unpack_sigmas(_mvbm_result_minimizer(result), p, nregimes)
    prof = _mvbmm_profile(tree, data, fitted_sigmas, mapped)
    nsigma = nregimes * (div(p * (p + 1), 2))
    nparams = nsigma + p
    return MVContinuousMultiBMResult(
        model = :mvBMM,
        backend = :tree_pruning,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, nregimes, "regime"),
        sigma = prof.sigma,
        theta = prof.theta,
        converged = _mvbm_result_converged(result),
        iterations = _mvbm_result_iterations(result),
        f_calls = _mvbm_result_f_calls(result),
    )
end

function fit_mvbmm(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvbmm(tree, trait, simmap.mapped_edge; trait_names = trait_names, regime_names = simmap.state_labels, kwargs...)
end

function fit_mvbmm(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    simmap::SimmapSample;
    kwargs...,
)
    return fit_mvbmm(tree, trait, simmap.mapped_edge; regime_names = simmap.state_labels, kwargs...)
end

