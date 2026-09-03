Base.@kwdef struct MVOUNLoptResult
    minimizer::Vector{Float64}
    minimum::Float64
    ret
    iterations::Int = 0
end

Base.@kwdef struct MVOUCompositeOptResult
    minimizer::Vector{Float64}
    minimum::Float64
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end

@inline _mvou_profiles_theta(spec::MVOUSpec) =
    spec.root_cov_mode === :fixed &&
    (spec.model === :mvOU1 || spec.root_mean_mode === :stationary_design)

function _mvou_spd_block_valid(M::AbstractMatrix{<:Real}; mineig::Float64, maxeig::Float64 = 1e8)
    all(isfinite, M) || return false
    p = size(M, 1)
    size(M, 2) == p || return false
    if p == 1
        v = Float64(M[1, 1])
        return mineig < v < maxeig
    elseif p == 2
        a = Float64(M[1, 1])
        d = Float64(M[2, 2])
        b = 0.5 * (Float64(M[1, 2]) + Float64(M[2, 1]))
        tr = a + d
        disc = sqrt(max((a - d) * (a - d) + 4.0 * b * b, 0.0))
        λmin = 0.5 * (tr - disc)
        λmax = 0.5 * (tr + disc)
        return λmin > mineig && λmax < maxeig
    end
    vals = eigvals(Symmetric(Matrix{Float64}(M)))
    return minimum(vals) > mineig && maximum(vals) < maxeig
end

function _mvou_bundle_valid_for_objective(bundle::MVOUParameterBundle, spec::MVOUSpec)
    _stable_pos_realpart(M::AbstractMatrix{<:Real}; mineig::Float64, maxeig::Float64 = 1e8) = begin
        all(isfinite, M) || return false
        size(M, 1) == size(M, 2) || return false
        vals = eigvals(Matrix{Float64}(M))
        reals = real.(vals)
        minimum(reals) > mineig && maximum(reals) < maxeig
    end

    if spec.A_mode === :by_regime
        @inbounds for r in axes(bundle.A_regimes, 3)
            if spec.A_decomp === :cholesky
                _mvou_spd_block_valid(@view(bundle.A_regimes[:, :, r]); mineig = 1e-8, maxeig = 10.0) || return false
            elseif spec.A_decomp === :schur
                _stable_pos_realpart(@view(bundle.A_regimes[:, :, r]); mineig = 1e-8, maxeig = 10.0) || return false
            else
                return false
            end
        end
    else
        if spec.A_decomp === :cholesky
            _mvou_spd_block_valid(bundle.A; mineig = 1e-8, maxeig = 10.0) || return false
        elseif spec.A_decomp === :schur
            _stable_pos_realpart(bundle.A; mineig = 1e-8, maxeig = 10.0) || return false
        else
            return false
        end
    end
    if spec.Sigma_mode === :by_regime
        @inbounds for r in axes(bundle.Sigma_regimes, 3)
            _mvou_spd_block_valid(@view(bundle.Sigma_regimes[:, :, r]); mineig = 1e-5) || return false
        end
    else
        _mvou_spd_block_valid(bundle.Sigma; mineig = 1e-5) || return false
    end
    return true
end

function _mvou_build_objective(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    spec::MVOUSpec,
    precalc::MVOUPrecalc,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    workspace = _mv_profile_workspace(tree, p)
    cov_unpack_workspace =
        _mvou_profiles_theta(spec) ?
        _mvou_cov_unpack_workspace(spec, p, precalc.nregimes) :
        nothing
    theta_workspace =
        _mvou_profiles_theta(spec) ?
        _mvou_theta_profile_workspace(tree, p, p * precalc.nregimes; nregimes = precalc.nregimes) :
        workspace
    objective = function (pars)
        try
            prof =
                if _mvou_profiles_theta(spec)
                    bundle = _mvou_unpack_cov_params!(
                        cov_unpack_workspace.bundle,
                        cov_unpack_workspace.L,
                        spec,
                        pars,
                        p,
                        precalc.nregimes,
                    )
                    _mvou_bundle_valid_for_objective(bundle, spec) || return Inf
                    _mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec, theta_workspace)
                else
                    bundle = _mvou_unpack_params(spec, pars, p)
                    _mvou_bundle_valid_for_objective(bundle, spec) || return Inf
                    _mvou_profile_dispatch(tree, data, bundle, precalc, workspace)
                end
            value = prof.success ? -prof.loglik : Inf
            return isfinite(value) ? value : Inf
        catch err
            err isa LinearAlgebra.SingularException && return Inf
            err isa PosDefException && return Inf
            err isa ArgumentError && occursin("Infs or NaNs", sprint(showerror, err)) && return Inf
            rethrow()
        end
    end
    return (data = data, p = p, objective = objective)
end

function _mvou_spd_diag_parameter_offsets(offset::Integer, p::Integer)
    out = Int[]
    idx = Int(offset)
    for j in 1:p
        push!(out, idx)
        idx += p - j + 1
    end
    return out
end

function _mvou_parameter_lower_bounds(spec::MVOUSpec, p::Integer, nregimes::Integer)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    ntheta = p * nregimes
    lower = fill(-Inf, (spec.A_mode === :by_regime ? nregimes * Ablock : Ablock) + (spec.Sigma_mode === :by_regime ? nregimes * Sblock : Sblock) + ntheta)
    offset = 1
    for _ in 1:(spec.A_mode === :by_regime ? nregimes : 1)
        lower[_mvou_A_diag_parameter_offsets(spec, offset, p)] .= 1e-8
        offset += Ablock
    end
    for _ in 1:(spec.Sigma_mode === :by_regime ? nregimes : 1)
        lower[_mvou_spd_diag_parameter_offsets(offset, p)] .= 1e-8
        offset += Sblock
    end
    return lower
end

function _mvou_cov_parameter_lower_bounds(spec::MVOUSpec, p::Integer, nregimes::Integer)
    Ablock = _mvou_A_block_nparams(spec, p)
    Sblock = _mvou_sigma_block_nparams(p)
    lower = fill(-Inf, (spec.A_mode === :by_regime ? nregimes * Ablock : Ablock) + (spec.Sigma_mode === :by_regime ? nregimes * Sblock : Sblock))
    offset = 1
    for _ in 1:(spec.A_mode === :by_regime ? nregimes : 1)
        lower[_mvou_A_diag_parameter_offsets(spec, offset, p)] .= 1e-8
        offset += Ablock
    end
    for _ in 1:(spec.Sigma_mode === :by_regime ? nregimes : 1)
        lower[_mvou_spd_diag_parameter_offsets(offset, p)] .= 1e-8
        offset += Sblock
    end
    return lower
end

function _mvou_tip_regime_group_means(
    data::AbstractMatrix{<:Real},
    precalc::MVOUPrecalc,
    p::Integer,
)
    precalc.nregimes > 1 || return nothing
    theta = repeat(vec(mean(data; dims = 1)), precalc.nregimes)
    counts = zeros(Int, precalc.nregimes)
    sums = zeros(Float64, precalc.nregimes, p)
    for i in 1:size(data, 1)
        r = Int(precalc.tip_terminal_regime[i])
        1 <= r <= precalc.nregimes || continue
        counts[r] += 1
        @views sums[r, :] .+= data[i, :]
    end
    for r in 1:precalc.nregimes
        counts[r] == 0 && continue
        theta[((r - 1) * p + 1):(r * p)] .= sums[r, :] ./ counts[r]
    end
    return theta
end

function _mvou_A_diag_parameter_offsets(spec::MVOUSpec, offset::Integer, p::Integer)
    if spec.A_decomp === :cholesky
        return _mvou_spd_diag_parameter_offsets(offset, p)
    elseif spec.A_decomp === :schur
        p == 2 || throw(ArgumentError("A_decomp=:schur is currently supported only for p=2"))
        return [Int(offset), Int(offset) + 1]
    end
    throw(ArgumentError("Unsupported multivariate OU A_decomp=$(spec.A_decomp)"))
end

function _mvou_tip_regime_covariances(
    data::AbstractMatrix{<:Real},
    precalc::MVOUPrecalc,
    p::Integer,
)
    precalc.nregimes > 1 || return Matrix{Float64}[]
    covs = Matrix{Float64}[]
    for r in 1:precalc.nregimes
        rows = findall(==(Int32(r)), precalc.tip_terminal_regime)
        isempty(rows) && continue
        sub = @view data[rows, :]
        C = _mv_complete_rows_cov(sub)
        C = (C + C') / 2 + 1e-8 * Matrix{Float64}(I, p, p)
        push!(covs, C)
    end
    return covs
end

function _mvou_unique_cov_candidates(covs::AbstractVector{<:AbstractMatrix{<:Real}})
    out = Matrix{Float64}[]
    for C0 in covs
        C = Matrix{Float64}(C0)
        any(existing -> maximum(abs.(existing .- C)) <= 1e-10, out) && continue
        push!(out, C)
    end
    return out
end

function _mvou_scalar_A_candidates(tree::CompactTree, p::Integer)
    height = maximum(tree.dist_from_root[tree.tip_ids])
    time_scaled = [log(2.0) / max(height / d, 1e-8) for d in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0)]
    absolute = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    scales = unique(filter(isfinite, vcat(time_scaled, absolute)))
    return [Matrix{Float64}(I, p, p) .* max(s, 1e-8) for s in scales]
end

function _mvou_A_candidates(tree::CompactTree, p::Integer)
    candidates = _mvou_scalar_A_candidates(tree, p)
    p == 2 || return candidates
    for scale in (0.4, 0.8, 1.6), rho in (-0.8, -0.5, 0.5), diag_scale in ((1.0, 1.0), (0.7, 1.3), (1.3, 0.7))
        D = Diagonal(collect(Float64, diag_scale))
        C = [1.0 rho; rho 1.0]
        push!(candidates, Float64(scale) .* Matrix(D * C * D))
    end
    for scale in (1.0, 1.1, 1.2), rho in (-0.9, -0.8, -0.7), diag_scale in ((0.8, 1.25), (0.85, 1.3), (1.25, 0.8), (1.3, 0.85))
        D = Diagonal(collect(Float64, diag_scale))
        C = [1.0 rho; rho 1.0]
        push!(candidates, Float64(scale) .* Matrix(D * C * D))
    end
    return candidates
end

function _mvou_schur_A_candidates(tree::CompactTree, p::Integer)
    p == 2 || throw(ArgumentError("A_decomp=:schur is currently supported only for p=2"))
    height = maximum(tree.dist_from_root[tree.tip_ids])
    scales = unique(filter(isfinite, vcat([log(2.0) / max(height / d, 1e-8) for d in (1.0, 2.0, 4.0, 8.0)], [0.1, 0.4, 1.6])))
    rotations = (0.0, pi / 8, -pi / 8)
    couplings = (-0.8, -0.2, 0.0, 0.2, 0.8)
    out = Matrix{Float64}[]
    for λ1 in scales, λ2 in scales, angle in rotations, u in couplings
        c = cos(angle)
        s = sin(angle)
        Q = [c -s; s c]
        T = [λ1 u; 0.0 λ2]
        push!(out, Q * T * Q')
    end
    return out
end

function _mvou_unique_matrix_candidates(
    mats::Vector{Matrix{Float64}};
    atol::Float64 = 1e-10,
)
    out = Matrix{Float64}[]
    for M in mats
        duplicate = false
        for kept in out
            if size(kept) == size(M) && maximum(abs.(kept .- M)) <= atol
                duplicate = true
                break
            end
        end
        duplicate || push!(out, M)
    end
    return out
end

function _mvou_scaled_blocks(base::AbstractMatrix{<:Real}, scales::AbstractVector{<:Real})
    p = size(base, 1)
    out = Array{Float64, 3}(undef, p, p, length(scales))
    B = Matrix{Float64}(base)
    @inbounds for (r, scale) in enumerate(scales)
        out[:, :, r] .= B .* Float64(scale)
    end
    return out
end

function _mvou_congruence_blocks(base::AbstractMatrix{<:Real}, diag_scales)
    p = size(base, 1)
    out = Array{Float64, 3}(undef, p, p, length(diag_scales))
    B = Matrix{Float64}(base)
    for (r, scales) in enumerate(diag_scales)
        length(scales) == p || throw(ArgumentError("diag scale length must match trait dimension"))
        D = Diagonal(Float64.(collect(scales)))
        out[:, :, r] .= Matrix(D * B * D)
    end
    return out
end

function _mvou_regime_cov_candidates(
    spec::MVOUSpec,
    tree::CompactTree,
    data::AbstractMatrix{<:Real},
    precalc::MVOUPrecalc,
    p::Integer,
    A0::AbstractMatrix{<:Real},
    Sigma0::AbstractMatrix{<:Real},
)
    precalc.nregimes > 1 || return Vector{Vector{Float64}}()
    (spec.A_mode === :by_regime || spec.Sigma_mode === :by_regime) || return Vector{Vector{Float64}}()

    scale_sets = (
        fill(1.0, precalc.nregimes),
        range(0.55, 1.7; length = precalc.nregimes),
        range(1.7, 0.55; length = precalc.nregimes),
    )
    function cycle_to_n(template, n)
        return [template[mod1(i, length(template))] for i in 1:n]
    end
    diag_sets =
        p == 2 ?
        (
            cycle_to_n([(1.8, 0.7), (0.7, 1.8), (1.35, 1.0), (1.0, 1.35)], precalc.nregimes),
            cycle_to_n([(0.65, 1.55), (1.55, 0.65), (0.85, 1.25), (1.25, 0.85)], precalc.nregimes),
        ) :
        ()

    candidates = Vector{Vector{Float64}}()
    for scales in scale_sets
        A_candidate =
            spec.A_mode === :by_regime ? _mvou_scaled_blocks(A0, scales) : A0
        Sigma_candidate =
            spec.Sigma_mode === :by_regime ? _mvou_scaled_blocks(Sigma0, scales) : Sigma0
        push!(
            candidates,
            _mvou_pack_initial_cov_params(
                spec,
                p,
                precalc.nregimes,
                A_candidate,
                Sigma_candidate,
            ),
        )
    end
    for diag in diag_sets
        A_candidate =
            spec.A_mode === :by_regime ? _mvou_congruence_blocks(A0, diag) : A0
        Sigma_candidate =
            spec.Sigma_mode === :by_regime ? _mvou_congruence_blocks(Sigma0, diag) : Sigma0
        push!(
            candidates,
            _mvou_pack_initial_cov_params(
                spec,
                p,
                precalc.nregimes,
                A_candidate,
                Sigma_candidate,
            ),
        )
    end
    return candidates
end

function _mvou_try_push_candidate!(builder, candidates::Vector{Vector{Float64}})
    try
        push!(candidates, builder())
    catch err
        err isa PosDefException || rethrow()
    end
    return candidates
end

function _mvou_unique_parameter_candidates(
    candidates::Vector{Vector{Float64}};
    atol::Float64 = 1e-10,
)
    out = Vector{Vector{Float64}}()
    for cand in candidates
        duplicate = false
        for kept in out
            if length(kept) == length(cand) && maximum(abs.(kept .- cand)) <= atol
                duplicate = true
                break
            end
        end
        duplicate || push!(out, cand)
    end
    return out
end

function _mvou_initial_candidates(
    spec::MVOUSpec,
    tree::CompactTree,
    data::AbstractMatrix{<:Real},
    precalc::MVOUPrecalc,
    p::Integer,
)
    base = _mvou_initial_params(
        spec,
        tree,
        data;
        nregimes = precalc.nregimes,
    )

    empirical = _mv_complete_rows_cov(data)
    empirical = (empirical + empirical') / 2 + 1e-8 * Matrix{Float64}(I, p, p)
    theta_candidates = [base[(end - p * precalc.nregimes + 1):end]]
    if spec.theta_mode === :by_regime
        grouped = _mvou_tip_regime_group_means(data, precalc, p)
        grouped !== nothing && push!(theta_candidates, grouped)
    end

    sigma_scales = spec.Sigma_mode === :shared ? (0.5, 1.0, 2.0) : (1.0,)
    candidates = Vector{Vector{Float64}}()
    A_candidates =
        if spec.A_decomp === :schur
            _mvou_unique_matrix_candidates(_mvou_schur_A_candidates(tree, p))
        else
            _mvou_A_candidates(tree, p)
        end
    for A0 in A_candidates, sigma_scale in sigma_scales, theta0 in theta_candidates
        scatter = (A0 * empirical + empirical * A0')
        scatter = (scatter + scatter') / 2
        _mvou_try_push_candidate!(candidates) do
            _mvou_pack_initial_params(
                spec,
                p,
                precalc.nregimes,
                A0,
                scatter .* sigma_scale,
                theta0,
            )
        end
    end
    push!(candidates, base)
    return _mvou_unique_parameter_candidates(candidates)
end

function _mvou_initial_cov_candidates(
    spec::MVOUSpec,
    tree::CompactTree,
    data::AbstractMatrix{<:Real},
    precalc::MVOUPrecalc,
    p::Integer,
)
    base = _mvou_initial_cov_params(
        spec,
        tree,
        data;
        nregimes = precalc.nregimes,
    )

    empirical = _mv_complete_rows_cov(data)
    empirical = (empirical + empirical') / 2 + 1e-8 * Matrix{Float64}(I, p, p)
    regime_covs = _mvou_tip_regime_covariances(data, precalc, p)
    empirical_covs = _mvou_unique_cov_candidates(vcat([empirical], regime_covs))
    sigma_scales = spec.Sigma_mode === :shared ? (0.5, 1.0, 2.0) : (1.0,)
    candidates = Vector{Vector{Float64}}()
    base_bundle = _mvou_unpack_cov_params(spec, base, p, precalc.nregimes)
    append!(
        candidates,
        _mvou_regime_cov_candidates(
            spec,
            tree,
            data,
            precalc,
            p,
            base_bundle.A,
            base_bundle.Sigma,
        ),
    )
    A_candidates =
        if spec.A_decomp === :schur
            _mvou_unique_matrix_candidates(_mvou_schur_A_candidates(tree, p))
        else
            _mvou_A_candidates(tree, p)
        end
    for A0 in A_candidates, empirical_cov in empirical_covs, sigma_scale in sigma_scales
        scatter = (A0 * empirical_cov + empirical_cov * A0')
        scatter = (scatter + scatter') / 2
        _mvou_try_push_candidate!(candidates) do
            _mvou_pack_initial_cov_params(
                spec,
                p,
                precalc.nregimes,
                A0,
                scatter .* sigma_scale,
            )
        end
    end
    push!(candidates, base)
    return _mvou_unique_parameter_candidates(candidates)
end

function _mvou_optimize_objective(
    objective,
    p0::AbstractVector{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    if optimization in (:LN_SBPLX, :LN_BOBYQA, :LN_NELDERMEAD)
        init = Vector{Float64}(p0)
        opt = NLopt.Opt(optimization, length(init))
        opt.ftol_rel = rel_tol
        opt.xtol_rel = rel_tol
        opt.maxeval = Int(max_iterations)
        lower_bounds !== nothing && (opt.lower_bounds = Vector{Float64}(lower_bounds))
        opt.min_objective = (x, grad) -> begin
            value = objective(x)
            return isfinite(value) ? value : 1e300
        end
        minf, minx, ret = NLopt.optimize(opt, init)
        return MVOUNLoptResult(minimizer = minx, minimum = minf, ret = ret, iterations = NLopt.numevals(opt))
    end

    if optimization == :LD_TNEWTON
        init = Vector{Float64}(p0)
        opt = NLopt.Opt(:LD_TNEWTON, length(init))
        opt.ftol_rel = rel_tol
        opt.xtol_rel = rel_tol
        opt.maxeval = Int(max_iterations)
        lower_bounds !== nothing && (opt.lower_bounds = Vector{Float64}(lower_bounds))
        xwork = Vector{Float64}(p0)
        opt.min_objective = (x, grad) -> begin
            fx = objective(x)
            if !isfinite(fx)
                length(grad) > 0 && fill!(grad, 0.0)
                return 1e300
            end
            if length(grad) > 0
                copyto!(xwork, x)
                @inbounds for i in eachindex(x)
                    xi = Float64(x[i])
                    h = sqrt(eps(Float64)) * max(abs(xi), 1.0)
                    xwork[i] = xi + h
                    fplus = objective(xwork)
                    if isfinite(fplus)
                        grad[i] = (fplus - fx) / h
                    else
                        xwork[i] = xi - h
                        if lower_bounds !== nothing && xwork[i] <= lower_bounds[i]
                            grad[i] = 0.0
                        else
                            fminus = objective(xwork)
                            grad[i] = isfinite(fminus) ? (fx - fminus) / h : 0.0
                        end
                    end
                    xwork[i] = xi
                end
            end
            return fx
        end
        minf, minx, ret = NLopt.optimize(opt, init)
        return MVOUNLoptResult(minimizer = minx, minimum = minf, ret = ret, iterations = NLopt.numevals(opt))
    end

    algorithm =
        optimization === :L_BFGS ? Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()) :
        optimization === :NelderMead ? Optim.NelderMead() :
        throw(ArgumentError("Unsupported optimization=$optimization"))

    if optimization === :L_BFGS
        xwork = Vector{Float64}(p0)
        cache_x = Vector{Float64}(p0)
        cache_valid = false
        cache_fx = Ref(Inf)
        function cached_objective(x)
            if cache_valid && length(x) == length(cache_x)
                same = true
                @inbounds for i in eachindex(x)
                    if x[i] != cache_x[i]
                        same = false
                        break
                    end
                end
                same && return cache_fx[]
            end
            fx = objective(x)
            copyto!(cache_x, x)
            cache_fx[] = fx
            cache_valid = true
            return fx
        end
        function grad!(G, x)
            fx = cached_objective(x)
            if !isfinite(fx)
                fill!(G, 0.0)
                return G
            end
            copyto!(xwork, x)
            @inbounds for i in eachindex(x)
                xi = Float64(x[i])
                h = sqrt(eps(Float64)) * max(abs(xi), 1.0)
                xwork[i] = xi + h
                fplus = cached_objective(xwork)
                if isfinite(fplus)
                    G[i] = (fplus - fx) / h
                else
                    xwork[i] = xi - h
                    if lower_bounds !== nothing && xwork[i] <= lower_bounds[i]
                        G[i] = 0.0
                    else
                        fminus = cached_objective(xwork)
                        G[i] = isfinite(fminus) ? (fx - fminus) / h : 0.0
                    end
                end
                xwork[i] = xi
            end
            return G
        end
        od = Optim.OnceDifferentiable(cached_objective, grad!, Vector{Float64}(p0))
        try
            return Optim.optimize(
                od,
                Vector{Float64}(p0),
                algorithm,
                Optim.Options(
                    iterations = max_iterations,
                    f_reltol = rel_tol,
                    g_tol = 1e-4,
                    allow_f_increases = false,
                ),
            )
        catch err
            if err isa AssertionError
                return _mvou_optimize_objective(
                    objective,
                    p0;
                    optimization = :LN_SBPLX,
                    max_iterations = max_iterations,
                    rel_tol = rel_tol,
                    lower_bounds = lower_bounds,
                )
            end
            rethrow()
        end
    end

    try
        return Optim.optimize(
            objective,
            Vector{Float64}(p0),
            algorithm,
            Optim.Options(
                iterations = max_iterations,
                f_reltol = rel_tol,
                g_tol = 1e-4,
                allow_f_increases = false,
            ),
        )
    catch err
        if optimization === :L_BFGS && err isa AssertionError
            return _mvou_optimize_objective(
                objective,
                p0;
                optimization = :LN_SBPLX,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end
        rethrow()
    end
end

_mvou_result_minimizer(result::MVOUNLoptResult) = result.minimizer
_mvou_result_minimizer(result::MVOUCompositeOptResult) = result.minimizer
_mvou_result_minimizer(result) = Optim.minimizer(result)

_mvou_result_converged(result::MVOUNLoptResult) =
    result.ret in (NLopt.SUCCESS, NLopt.STOPVAL_REACHED, NLopt.FTOL_REACHED, NLopt.XTOL_REACHED)
_mvou_result_converged(result::MVOUCompositeOptResult) = result.converged
_mvou_result_converged(result) = Optim.converged(result)

_mvou_result_iterations(result::MVOUNLoptResult) = result.iterations
_mvou_result_iterations(result::MVOUCompositeOptResult) = result.iterations
_mvou_result_iterations(result) = Optim.iterations(result)

_mvou_result_f_calls(result::MVOUNLoptResult) = result.iterations
_mvou_result_f_calls(result::MVOUCompositeOptResult) = result.f_calls
_mvou_result_f_calls(result) = Optim.f_calls(result)

function _mvou_two_stage_result(
    objective,
    p0::AbstractVector{<:Real};
    rough_iterations::Integer,
    polish_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}},
)
    rough = _mvou_optimize_objective(
        objective,
        p0;
        optimization = :LN_SBPLX,
        max_iterations = rough_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )
    polish = _mvou_optimize_objective(
        objective,
        _mvou_result_minimizer(rough);
        optimization = :L_BFGS,
        max_iterations = polish_iterations,
        rel_tol = min(rel_tol, 1e-7),
        lower_bounds = lower_bounds,
    )
    rough_min = Float64(rough.minimum)
    polish_min =
        polish isa MVOUNLoptResult ? Float64(polish.minimum) :
        Float64(Optim.minimum(polish))
    best = polish_min <= rough_min ? polish : rough
    return MVOUCompositeOptResult(
        minimizer = Vector{Float64}(_mvou_result_minimizer(best)),
        minimum = min(rough_min, polish_min),
        converged = _mvou_result_converged(rough) || _mvou_result_converged(polish),
        iterations = _mvou_result_iterations(rough) + _mvou_result_iterations(polish),
        f_calls = _mvou_result_f_calls(rough) + _mvou_result_f_calls(polish),
    )
end

function _mvou_best_candidate_indices(values::AbstractVector{<:Real}, max_count::Integer)
    finite_indices = [i for i in eachindex(values) if isfinite(values[i])]
    if isempty(finite_indices)
        return [argmin(values)]
    end
    sort!(finite_indices; by = i -> values[i])
    return finite_indices[1:min(Int(max_count), length(finite_indices))]
end

@inline _mvou_parallel_multistart_model(model::Symbol) =
    model in (:mvOU1, :mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA)

@inline _mvou_multistart_candidate_count(model::Symbol) =
    10

@inline function _mvou_parallel_candidate_scoring(spec::MVOUSpec, ncandidates::Integer)
    return Threads.nthreads() > 1 && Int(ncandidates) > 16 && spec.model in (:mvOU1, :mvOUM, :mvOUMV, :mvOUMA, :mvOUMVA)
end

function _mvou_candidate_objective_values(
    tree::CompactTree,
    data::AbstractMatrix{<:Real},
    spec::MVOUSpec,
    precalc::MVOUPrecalc,
    candidates::Vector{Vector{Float64}},
    built_objective,
)
    values = Vector{Float64}(undef, length(candidates))
    if _mvou_parallel_candidate_scoring(spec, length(candidates))
        local_builts = [_mvou_build_objective(tree, data, spec, precalc) for _ in 1:Threads.maxthreadid()]
        Threads.@threads :dynamic for i in eachindex(candidates)
            local_built = local_builts[Threads.threadid()]
            values[i] = local_built.objective(candidates[i])
        end
    else
        for i in eachindex(candidates)
            values[i] = built_objective(candidates[i])
        end
    end
    return values
end

function _mvou_optimize_multistart_candidates(
    tree::CompactTree,
    data::AbstractMatrix{<:Real},
    spec::MVOUSpec,
    precalc::MVOUPrecalc,
    candidates::Vector{Vector{Float64}},
    candidate_indices::AbstractVector{<:Integer};
    rough_iterations::Integer,
    polish_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::Union{Nothing, AbstractVector{<:Real}},
)
    nstarts = length(candidate_indices)
    results = Vector{Union{Nothing, MVOUCompositeOptResult}}(nothing, nstarts)
    if _mvou_parallel_multistart_model(spec.model) && Threads.nthreads() > 1 && nstarts > 1
        local_builts = [_mvou_build_objective(tree, data, spec, precalc) for _ in 1:Threads.maxthreadid()]
        Threads.@threads :dynamic for pos in eachindex(candidate_indices)
            local_built = local_builts[Threads.threadid()]
            results[pos] = _mvou_two_stage_result(
                local_built.objective,
                candidates[Int(candidate_indices[pos])];
                rough_iterations = rough_iterations,
                polish_iterations = polish_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end
    else
        local_built = _mvou_build_objective(tree, data, spec, precalc)
        for pos in eachindex(candidate_indices)
            results[pos] = _mvou_two_stage_result(
                local_built.objective,
                candidates[Int(candidate_indices[pos])];
                rough_iterations = rough_iterations,
                polish_iterations = polish_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end
    end

    best_result = results[1]
    best_min = best_result === nothing ? Inf : best_result.minimum
    for pos in 2:nstarts
        candidate_result = results[pos]
        candidate_result === nothing && continue
        if candidate_result.minimum < best_min
            best_min = candidate_result.minimum
            best_result = candidate_result
        end
    end
    return best_result
end

function _mvou_finalize_fit_result(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    spec::MVOUSpec,
    precalc::MVOUPrecalc,
    result;
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    bundle =
        if _mvou_profiles_theta(spec)
            cov_bundle = _mvou_unpack_cov_params(spec, _mvou_result_minimizer(result), p, precalc.nregimes)
            prof_tmp = _mvou_profile_theta_tree_pruning_profile(tree, data, cov_bundle, precalc, spec)
            MVOUParameterBundle(
                theta = prof_tmp.success ? prof_tmp.theta : zeros(Float64, p * precalc.nregimes),
                A = cov_bundle.A,
                A_regimes = cov_bundle.A_regimes,
                Sigma = cov_bundle.Sigma,
                Sigma_regimes = cov_bundle.Sigma_regimes,
            )
        else
            _mvou_unpack_params(spec, _mvou_result_minimizer(result), p)
        end
    prof =
        if _mvou_profiles_theta(spec)
            _mvou_profile_theta_tree_pruning_profile(tree, data, bundle, precalc, spec)
        else
            _mvou_profile_dispatch(tree, data, bundle, precalc)
        end
    nparams =
        (spec.A_mode === :by_regime ? _mvou_A_block_nparams(spec, p) * precalc.nregimes : _mvou_A_block_nparams(spec, p)) +
        (spec.Sigma_mode === :by_regime ? _mvou_sigma_block_nparams(p) * precalc.nregimes : _mvou_sigma_block_nparams(p)) +
        p * precalc.nregimes
    A_blocks =
        if spec.A_mode === :by_regime
            bundle.A_regimes
        else
            reshape(bundle.A, p, p, 1)
        end
    sigma_blocks =
        if spec.Sigma_mode === :by_regime
            bundle.Sigma_regimes
        else
            reshape(bundle.Sigma, p, p, 1)
        end
    return MVContinuousOUResult(
        model = spec.model,
        success = prof.success,
        loglik = prof.loglik,
        aic = prof.success ? compute_aic(prof.loglik, nparams) : Inf,
        nparams = nparams,
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = precalc.nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, precalc.nregimes, "regime"),
        theta = reshape(bundle.theta, p, precalc.nregimes)',
        A = A_blocks,
        Sigma = sigma_blocks,
        A_decomp = spec.A_decomp,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
        converged = _mvou_result_converged(result),
        iterations = _mvou_result_iterations(result),
        f_calls = _mvou_result_f_calls(result),
    )
end

function _fit_mvou_recursive(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    spec::MVOUSpec;
    edge_segments::Union{Nothing, Vector{Vector{SimmapSegment}}} = nothing,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 400,
    rel_tol::Float64 = 1e-6,
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    precalc = _mvou_precalc(tree, spec; edge_segments = edge_segments)
    built = _mvou_build_objective(tree, data, spec, precalc)
    candidates =
        if _mvou_profiles_theta(spec)
            _mvou_initial_cov_candidates(
                spec,
                tree,
                data,
                precalc,
                built.p,
            )
        else
            _mvou_initial_candidates(
                spec,
                tree,
                data,
                precalc,
                built.p,
            )
        end
    values = _mvou_candidate_objective_values(tree, data, spec, precalc, candidates, built.objective)
    best_index = argmin(values)
    p0 = candidates[best_index]
    lower_bounds =
        _mvou_profiles_theta(spec) ?
        _mvou_cov_parameter_lower_bounds(spec, built.p, precalc.nregimes) :
        _mvou_parameter_lower_bounds(spec, built.p, precalc.nregimes)
    result =
        if optimization in (:SBPLX_L_BFGS, :LN_SBPLX_L_BFGS, :TWO_STAGE)
            rough_iterations = Int(max_iterations)
            polish_iterations = 100
            candidate_indices = _mvou_best_candidate_indices(
                values,
                _mvou_multistart_candidate_count(spec.model),
            )
            _mvou_optimize_multistart_candidates(
                tree,
                data,
                spec,
                precalc,
                candidates,
                candidate_indices;
                rough_iterations = rough_iterations,
                polish_iterations = polish_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        else
            _mvou_optimize_objective(
                built.objective,
                p0;
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = lower_bounds,
            )
        end
    return _mvou_finalize_fit_result(tree, data, spec, precalc, result; trait_names = trait_names, regime_names = regime_names)
end



