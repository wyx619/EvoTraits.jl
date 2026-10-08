function _shift_pack_general_A(A)
    # Schur-vector signs are arbitrary; choose a Cayley chart away from I + Q singularity.
    p = size(A, 1)
    factor = _shift_ou_factor(A, p, :decomposable_positive)
    Q, T = copy(factor.basis), copy(factor.triangular.A)
    original_Q, original_T = copy(Q), copy(T)
    signs = [Q[i,i] < 0 ? -1.0 : 1.0 for i in 1:p]
    D = Diagonal(signs)
    Q, T = Q * D, D * T * D
    if det(Q) < 0
        signs[argmin(abs.(diag(Q)))] *= -1
        D = Diagonal(signs)
        Q, T = original_Q * D, D * original_T * D
    end
    identity = Matrix{Float64}(I, p, p)
    cayley = try
        (identity + Q) \ (identity - Q)
    catch err
        err isa SingularException || rethrow()
        throw(ArgumentError("Initial A has a singular orthogonal Cayley chart"))
    end
    cayley = (cayley - cayley') / 2
    pars = vcat(log.(diag(T)), Float64[cayley[i,j] for j in 1:p-1 for i in j+1:p],
        Float64[T[i,j] for j in 2:p for i in 1:j-1])
    reconstructed = _mvou_unpack_decomposable_positive(pars, p; min_eigenvalue = 0.0)
    isapprox(reconstructed, A; rtol = 1e-8, atol = 1e-12) ||
        throw(ArgumentError("Initial A cannot be represented accurately by this Cayley chart"))
    return pars
end

function _fit_mvshift_sr_constrained(tree, trait, mapping;
    Sigma_decomp::Symbol, guess_sigma = nothing, trait_names = nothing,
    optimization::Symbol = :L_BFGS, max_iterations::Integer = 1000, rel_tol::Float64 = 1e-6)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    block = _shift_sigma_nparams(p, Sigma_decomp)
    max_iterations > 0 && isfinite(rel_tol) && rel_tol > 0 || throw(ArgumentError("Invalid optimization budget or tolerance"))
    names = _mv_checked_names(trait_names, p, "trait")
    sigmas = if guess_sigma === nothing
        empirical = _mv_complete_rows_cov(data)
        eig = eigen(Symmetric(empirical))
        initial = eig.vectors * Diagonal(max.(eig.values, max(maximum(abs, eig.values), 1.0) * 1e-6)) * eig.vectors'
        initial = (initial + initial') / 2
        initial = _shift_initial_sigma(initial / mapping.height, Sigma_decomp)
        [copy(initial), copy(initial)]
    else
        length(guess_sigma) == 2 || throw(ArgumentError("SR requires two initial diffusion matrices"))
        [_shift_sigma(S, p, Sigma_decomp) for S in guess_sigma]
    end
    p0 = vcat((_shift_pack_sigma(S, Sigma_decomp) for S in sigmas)...)
    workspace = _mv_profile_workspace(tree, p)
    covariance_workspace = _mvbm_covariance_workspace(tree, data)
    unpack(pars) = [_shift_unpack_sigma(@view(pars[(r-1)*block+1:r*block]), p, Sigma_decomp) for r in 1:2]
    objective = function (pars)
        all(isfinite, pars) || return Inf
        matrices = unpack(pars)
        all(S -> all(isfinite, S) && isposdef(Symmetric(S)), matrices) || return Inf
        prof = _mvbmm_profile_validated(tree, data, matrices, mapping.mapped_edge;
            workspace, covariance_workspace, return_sigma = false)
        return prof.success ? -prof.loglik : Inf
    end
    optimized = _mvbm_optimize_objective(objective, p0; optimization = optimization,
        max_iterations = max_iterations, rel_tol = rel_tol)
    evaluated = _mvshift_sr_loglikelihood(tree, data, unpack(_mvbm_result_minimizer(optimized));
        mapped_edge = mapping.mapped_edge, Sigma_decomp = Sigma_decomp, trait_names = names)
    fit = evaluated.fit
    values = (; (name => getfield(fit, name) for name in fieldnames(typeof(fit)))...)
    fit = MVContinuousMultiBMResult(; merge(values, (
        converged = fit.success && _mvbm_result_converged(optimized),
        iterations = _mvbm_result_iterations(optimized), f_calls = _mvbm_result_f_calls(optimized)))...)
    return merge(_shift_sr_result(fit, mapping, count(!isnan, data)), (Sigma_decomp = Sigma_decomp,))
end

function _shift_pack_A(A, decomp)
    decomp === :equaldiagonal && return _shift_pack_sigma(A,:equaldiagonal)
    decomp === :diagonal && return _mvou_pack_diagonal_A(A)
    decomp === :cholesky && return _mvbm1_pack_sigma(A)
    decomp === :decomposable_positive && return _shift_pack_general_A(A)
    decomp in (:upper, :lower) || throw(ArgumentError("Unsupported A_decomp=$decomp"))
    p = size(A, 1)
    pars = Float64[]
    for j in 1:p, i in 1:p
        (decomp === :upper ? i < j : i > j) && push!(pars, A[i, j])
    end
    append!(pars, log.(diag(A)))
    return pars
end

function _shift_unpack_A(pars, p, decomp)
    length(pars) == _shift_A_nparams(p, decomp) || throw(ArgumentError("A parameter count mismatch"))
    decomp === :equaldiagonal && return _shift_unpack_sigma(pars,p,:equaldiagonal)
    decomp === :diagonal && return _mvou_unpack_diagonal_A(pars, p)
    decomp === :cholesky && return _mvbm1_unpack_sigma(pars, p)
    decomp === :decomposable_positive && return _mvou_unpack_decomposable_positive(pars, p; min_eigenvalue = 0.0)
    decomp in (:upper, :lower) || throw(ArgumentError("Unsupported A_decomp=$decomp"))
    A = zeros(p, p)
    index = 1
    for j in 1:p, i in 1:p
        if decomp === :upper ? i < j : i > j
            A[i, j] = pars[index]
            index += 1
        end
    end
    for j in 1:p
        A[j, j] = exp(pars[index])
        index += 1
    end
    return A
end

"""
    _fit_mvshift_process(tree, trait; model=:OUBM, ...)

Internal OU/BM and EB shift fitting with profiled constant theta. `guess_sigma`
refers to OU diffusion when present, otherwise BM; `guess_sig` is the other component.
Symmetric positive-definite, diagonal, triangular, and general positive-real-spectrum A are supported.
"""
function _fit_mvshift_process(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real};
    model::Symbol = :OUBM,
    age = nothing,
    mapped_edge = nothing,
    precalc = nothing,
    A_decomp::Symbol = :cholesky,
    Sigma_decomp::Symbol = :cholesky,
    guess_A::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    guess_sigma::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    guess_sig::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    guess_beta::Union{Nothing, Real} = nothing,
    beta_bounds = (-2.0, 2.0),
    optimization::Symbol = :LN_SBPLX,
    max_iterations::Integer = 1000,
    rel_tol::Float64 = 1e-6,
    trait_names = nothing,
)
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    isfinite(rel_tol) && rel_tol > 0 || throw(ArgumentError("rel_tol must be finite and positive"))
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    _shift_sigma_nparams(p, Sigma_decomp)
    names = _mv_checked_names(trait_names, p, "trait")
    spec = _shift_process_spec(model)
    beta0 = if spec.eb_period > 0
        _shift_beta_bounds(beta_bounds)
        value = guess_beta === nothing ? 0.01 : Float64(guess_beta)
        isfinite(value) || throw(ArgumentError("guess_beta must be finite"))
        value
    else
        guess_beta === nothing || throw(ArgumentError("$(spec.model) has no EB component; omit guess_beta"))
        0.0
    end
    mapping = _shift_mapping(tree, p; age = age, mapped_edge = mapped_edge, precalc = precalc)
    A0 = if spec.ou_period > 0
        guess_A === nothing ? Matrix{Float64}(I, p, p) / mapping.height : Matrix{Float64}(guess_A)
    else
        guess_A === nothing || throw(ArgumentError("$(spec.model) has no OU component; omit guess_A"))
        nothing
    end
    factor0 = A0 === nothing ? nothing : _shift_ou_factor(A0, p, A_decomp)
    empirical = if guess_sigma === nothing || (spec.independent && guess_sig === nothing)
        cov = _mv_complete_rows_cov(data)
        cov = (cov + cov') / 2
        # Pairwise covariance with missing entries need not be positive definite.
        # This projection initializes diffusion only; likelihood operators never get jitter.
        eig = eigen(Symmetric(cov))
        floor_value = max(maximum(abs, eig.values), 1.0) * 1e-6
        cov = eig.vectors * Diagonal(max.(eig.values, floor_value)) * eig.vectors'
        cov = (cov + cov') / 2
        _shift_initial_sigma(cov / mapping.height, Sigma_decomp)
    else
        nothing
    end
    sigma0 = _shift_sigma(guess_sigma === nothing ? empirical : guess_sigma, p, Sigma_decomp)
    if spec.independent
        sig0 = _shift_sigma(guess_sig === nothing ? empirical : guess_sig, p, Sigma_decomp)
    else
        guess_sig === nothing || throw(ArgumentError("Shared diffusion models do not accept guess_sig"))
        sig0 = sigma0
    end
    a0 = A0 === nothing ? Float64[] : _shift_pack_A(A0, A_decomp)
    p0 = vcat(a0, _shift_pack_sigma(sigma0, Sigma_decomp),
        spec.independent ? _shift_pack_sigma(sig0, Sigma_decomp) : Float64[],
        spec.eb_period > 0 ? [beta0] : Float64[])
    unpack(pars) = _shift_process_unpack(pars, p, A_decomp, Sigma_decomp, spec)
    initial = _shift_oubm_profile(tree, data, mapping, spec,
        factor0, sigma0, sig0; beta = spec.eb_period > 0 ? _shift_effective_beta(beta0, beta_bounds) : 0.0)
    best_value = Ref(initial.success ? -initial.loglik : Inf)
    best_pars = copy(p0)
    objective = function (pars)
        all(isfinite, pars) || return Inf
        A, sigma, sig, beta = unpack(pars)
        (A === nothing || all(isfinite, A)) && all(isfinite, sigma) && all(isfinite, sig) || return Inf
        isposdef(Symmetric(sigma)) && isposdef(Symmetric(sig)) || return Inf
        factor = if A === nothing
            nothing
        elseif A_decomp === :decomposable_positive
            try
                _shift_ou_factor(A, p, A_decomp)
            catch err
                err isa ArgumentError || rethrow()
                return Inf
            end
        elseif A_decomp in (:upper, :lower)
            all(>(0), diag(A)) || return Inf
            ShiftAsymmetricFactor(A)
        else
            eig = _mvou_transition_eigen(A)
            all(>(0), eig.values) || return Inf
            eig
        end
        prof = _shift_oubm_profile(tree, data, mapping, spec, factor, sigma, sig;
            beta = spec.eb_period > 0 ? _shift_effective_beta(beta, beta_bounds) : 0.0)
        value = prof.success ? -prof.loglik : Inf
        if value < best_value[]
            best_value[] = value
            copyto!(best_pars, pars)
        end
        return value
    end
    optimized = _mvbm_optimize_objective(objective, p0;
        optimization = optimization, max_iterations = max_iterations, rel_tol = rel_tol)
    terminal_pars = _mvbm_result_minimizer(optimized)
    terminal_value = objective(terminal_pars)
    used_best = !isfinite(terminal_value) || terminal_value > best_value[]
    selected = used_best ? best_pars : terminal_pars
    A, sigma, sig, beta = unpack(selected)
    extra = spec.independent ? (sig = sig,) : NamedTuple()
    ebextra = spec.eb_period > 0 ? (beta = beta, beta_bounds = beta_bounds) : NamedTuple()
    evaluated = _mvshift_process_loglikelihood(tree, data, A, sigma;
        model = model, mapped_edge = mapping.mapped_edge, A_decomp = A_decomp,
        trait_names = names, Sigma_decomp = Sigma_decomp, extra..., ebextra...)
    converged = evaluated.success && !used_best && _mvbm_result_converged(optimized)
    termination = optimized isa MVBMNLoptResult ? string(optimized.ret) :
        (_mvbm_result_converged(optimized) ? "CONVERGED" : "NOT_CONVERGED")
    return merge(evaluated, (
        converged = converged, optimization = optimization, termination = termination,
        iterations = _mvbm_result_iterations(optimized), f_calls = _mvbm_result_f_calls(optimized),
        max_iterations = Int(max_iterations), rel_tol = rel_tol,
        initial_loglik = initial.loglik, used_best_candidate = used_best,
    ))
end

function _shift_process_unpack(pars, p, A_decomp, Sigma_decomp, spec)
    nalpha = spec.ou_period > 0 ? _shift_A_nparams(p, A_decomp) : 0
    block = _shift_sigma_nparams(p, Sigma_decomp)
    nprocess = nalpha + block * (spec.independent ? 2 : 1)
    length(pars) == nprocess + (spec.eb_period > 0 ? 1 : 0) ||
        throw(ArgumentError("Shift parameter count mismatch"))
    A = nalpha == 0 ? nothing : _shift_unpack_A(@view(pars[1:nalpha]), p, A_decomp)
    sigma = _shift_unpack_sigma(@view(pars[nalpha+1:nalpha+block]), p, Sigma_decomp)
    sig = spec.independent ? _shift_unpack_sigma(@view(pars[nalpha+block+1:nprocess]), p, Sigma_decomp) : sigma
    beta = spec.eb_period > 0 ? pars[end] : 0.0
    return A, sigma, sig, beta
end

function _fit_mvshift_oubm(tree::CompactTree, trait::AbstractMatrix{<:Real}; model::Symbol = :OUBM, kwargs...)
    _shift_oubm_spec(model)
    return _fit_mvshift_process(tree, trait; model = model, kwargs...)
end

function _fit_mvshift_eb(tree::CompactTree, trait::AbstractMatrix{<:Real}; model::Symbol = :EBBM, kwargs...)
    spec = _shift_process_spec(model)
    spec.eb_period > 0 || throw(ArgumentError("An EB shift model is required"))
    return _fit_mvshift_process(tree, trait; model = model, kwargs...)
end
