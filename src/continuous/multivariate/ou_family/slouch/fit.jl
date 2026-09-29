function _mvoubm_initial_candidates(
    tree::CompactTree,
    data::AbstractMatrix{Float64},
    precalc::MVOUPrecalc,
    kY::Int,
    kX::Int,
    A_decomp::Symbol,
)
    spec = _mvoubm_A_spec(A_decomp)
    Y = data[:, 1:kY]
    X = data[:, kY+1:end]
    global_theta = vec(mean(Y; dims = 1))
    mPsi0 = repeat(global_theta, 1, precalc.nregimes)
    counts = zeros(Int, precalc.nregimes)
    sums = zeros(Float64, precalc.nregimes, kY)
    for i in 1:tree.ntips
        r = Int(precalc.tip_terminal_regime[i])
        1 <= r <= precalc.nregimes || continue
        counts[r] += 1
        for j in 1:kY
            v = Y[i, j]
            isnan(v) || (sums[r, j] += v)
        end
    end
    for r in 1:precalc.nregimes
        if counts[r] > 0
            mPsi0[:, r] .= sums[r, :] ./ counts[r]
        end
    end
    complete = vec(.!isnan.(vec(sum(Y; dims = 2))))
    covY = try
        Matrix{Float64}(cov(Y[complete, :]))
    catch
        Matrix{Float64}(Diagonal(vec(var(Y[complete, :]; dims = 1))))
    end
    vX00 = vec(mean(X; dims = 1))
    varX = vec(var(X; dims = 1))
    SigmaX0 = Matrix{Float64}(Diagonal(max.(varX, 1e-6)))
    A_cands = if A_decomp === :schur
        _mvou_unique_matrix_candidates(_mvou_schur_A_candidates(tree, kY))
    elseif A_decomp === :upper_triangular
        [Matrix{Float64}(cholesky(Symmetric(Matrix{Float64}(M))).U) for M in _mvou_A_candidates(tree, kY)]
    elseif A_decomp === :lower_triangular
        [Matrix{Float64}(cholesky(Symmetric(Matrix{Float64}(M))).L) for M in _mvou_A_candidates(tree, kY)]
    else
        _mvou_A_candidates(tree, kY)
    end
    candidates = Vector{Vector{Float64}}()
    for A0 in A_cands[1:min(4, end)], s in (1.0, 0.5, 2.0)
        _mvou_try_push_candidate!(candidates) do
            vcat(
                _mvoubm_pack_A_block(A_decomp, A0),
                zeros(Float64, kY * kX),
                _mvou_pack_spd(covY .* s),
                _mvou_pack_spd(SigmaX0 .* s),
                vec(mPsi0),
                vX00,
            )
        end
    end
    return _mvou_unique_parameter_candidates(candidates)
end

function fit_mvoubm(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    edge_segments::Vector{Vector{SimmapSegment}},
    ;
    kY::Integer = size(trait, 2) - 1,
    optimization::Symbol = :SBPLX_L_BFGS,
    max_iterations::Integer = 300,
    rel_tol::Float64 = 1e-5,
    A_decomp::Symbol = :cholesky,
    trait_names = nothing,
    regime_names = nothing,
)
    data = _validate_multivariate_trait(tree, trait)
    p = size(data, 2)
    kY = Int(kY)
    1 <= kY < p || throw(ArgumentError("kY must satisfy 1 <= kY < number of traits"))
    A_decomp in (:cholesky, :schur, :decomposable_real, :upper_triangular, :lower_triangular) ||
        throw(ArgumentError("Unsupported fit_mvoubm A_decomp=$A_decomp"))
    kX = p - kY
    spec = MVOUSpec(
        model = :mvOUBM,
        theta_mode = :by_regime,
        A_mode = :shared,
        Sigma_mode = :shared,
        A_decomp = A_decomp,
        root_mean_mode = :theta,
        root_cov_mode = :fixed,
    )
    precalc = _mvou_build_precalc(tree, spec; edge_segments = edge_segments)
    ws = _mvoubm_workspace(tree, precalc, kY, kX)
    objective = x -> begin
        ll = try
            _mvoubm_loglik!(tree, data, precalc, ws, x)
        catch
            -Inf
        end
        return isfinite(ll) ? -ll : 1e300
    end
    candidates = _mvoubm_initial_candidates(tree, data, precalc, kY, kX, A_decomp)
    vals = Float64[objective(c) for c in candidates]
    best = nothing
    top = _mvou_best_candidate_indices(vals, 3)
    results = if Threads.nthreads() > 1 && length(top) > 1
        tasks = map(top) do i
            Threads.@spawn begin
                ws_i = _mvoubm_workspace(tree, precalc, kY, kX)
                obj_i = x -> begin
                    ll_i = try
                        _mvoubm_loglik!(tree, data, precalc, ws_i, x)
                    catch
                        -Inf
                    end
                    return isfinite(ll_i) ? -ll_i : 1e300
                end
                _mvou_two_stage_result(
                    obj_i,
                    candidates[i];
                    rough_iterations = max_iterations,
                    polish_iterations = max_iterations,
                    rel_tol = rel_tol,
                    lower_bounds = nothing,
                )
            end
        end
        Any[fetch(t) for t in tasks]
    else
        Any[
            _mvou_two_stage_result(
                objective,
                candidates[i];
                rough_iterations = max_iterations,
                polish_iterations = max_iterations,
                rel_tol = rel_tol,
                lower_bounds = nothing,
            )
            for i in top
        ]
    end
    for r in results
        (best === nothing || r.minimum < best.minimum) && (best = r)
    end
    pars = _mvou_result_minimizer(best)
    ll = _mvoubm_loglik!(tree, data, precalc, ws, pars)
    up = _mvoubm_unpack_params(pars, kY, kX, precalc.nregimes, A_decomp)
    Sigma_joint = [
        up.SigmaY zeros(Float64, kY, kX)
        zeros(Float64, kX, kY) up.SigmaX
    ]
    return MVContinuousOUResult(
        model = :mvOUBM,
        success = isfinite(ll),
        loglik = ll,
        aic = isfinite(ll) ? compute_aic(ll, _mvoubm_nparams(kY, kX, precalc.nregimes, A_decomp)) : Inf,
        nparams = _mvoubm_nparams(kY, kX, precalc.nregimes, A_decomp),
        nobs = count(!isnan, data),
        ntraits = p,
        nregimes = precalc.nregimes,
        trait_names = _mv_checked_names(trait_names, p, "trait"),
        regime_names = _mv_checked_names(regime_names, precalc.nregimes, "regime"),
        theta = Matrix{Float64}(permutedims(up.mPsi)),
        A = reshape(up.A, kY, kY, 1),
        Sigma = reshape(Sigma_joint, p, p, 1),
        A_decomp = A_decomp,
        root_mean_mode = :theta,
        root_cov_mode = :fixed,
        converged = _mvou_result_converged(best),
        iterations = _mvou_result_iterations(best),
        f_calls = _mvou_result_f_calls(best),
        B = up.B,
        vX0 = up.vX0,
    )
end

function fit_mvoubm(
    tree::CompactTree,
    data::AbstractDataFrame,
    simmap::SimmapSample;
    kwargs...,
)
    trait, trait_names = _mv_align_dataframe_traits(tree, data)
    return fit_mvoubm(
        tree,
        trait,
        simmap.edge_segments;
        trait_names = trait_names,
        regime_names = simmap.state_labels,
        kwargs...,
    )
end

function fit_mvoubm(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    simmap::SimmapSample;
    kwargs...,
)
    return fit_mvoubm(
        tree,
        trait,
        simmap.edge_segments;
        regime_names = simmap.state_labels,
        kwargs...,
    )
end
