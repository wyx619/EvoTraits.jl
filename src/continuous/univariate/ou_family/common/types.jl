Base.@kwdef struct OUSpec
    model::Symbol
    theta_mode::Symbol = :shared
    alpha_mode::Symbol = :shared
    sigma_mode::Symbol = :shared
    root_mean_mode::Symbol = :theta
    root_cov_mode::Symbol = :nonstationary
end

"""
    ou_spec(model; root_mean_mode=nothing, root_cov_mode=nothing)

Construct the univariate OU specification. For OUMV/OUMA/OUMVA, the OUwie
3.02 root convention is `:regime` with `:fixed` (the default,
`get.root.theta=false`). Public fitting and fixed-parameter APIs for these
models accept only this combination. Other specifications remain
available internally to reconstruct results saved under earlier conventions.

OU1/OUM public APIs accept `:regime` and
`:stationary`, with `:fixed` or `:random` covariance. The first
mean convention with fixed covariance is the common default across OU models.

`:stationary_design` starts the deterministic path mean at zero; it is not
mvMORPH's row-standardized `root="stationary"`. `:free` and `:nonstationary`
covariance both profile the root state without a finite Gaussian prior and
do not estimate an independent root variance.
"""
function ou_spec(
    model::Symbol;
    root_mean_mode::Union{Nothing, Symbol} = nothing,
    root_cov_mode::Union{Nothing, Symbol} = nothing,
)
    defaults = _ou_default_spec(model)
    mean_mode = something(root_mean_mode, defaults.root_mean_mode)
    cov_mode = something(root_cov_mode, defaults.root_cov_mode)
    _validate_ou_root_mean_mode(mean_mode)
    cov_mode in (:fixed, :random, :nonstationary, :free) ||
        throw(ArgumentError("Unsupported OU root_cov_mode=$cov_mode"))
    return OUSpec(
        model = defaults.model,
        theta_mode = defaults.theta_mode,
        alpha_mode = defaults.alpha_mode,
        sigma_mode = defaults.sigma_mode,
        root_mean_mode = mean_mode,
        root_cov_mode = cov_mode,
    )
end

function _validate_ou_root_mean_mode(mean_mode::Symbol)
    mean_mode === :free_theta0 &&
        throw(ArgumentError("root_mean_mode=:free_theta0 has been removed. Refit with root_mean_mode=:regime; saved free-root results cannot be used for likelihood or reconstruction."))
    mean_mode in (:theta, :regime, :stationary_design, :stationary) ||
        throw(ArgumentError("Unsupported OU root_mean_mode=$mean_mode"))
    return nothing
end

function _validate_regime_ou_root(model::Symbol, mean_mode::Symbol, cov_mode::Symbol)
    _validate_ou_root_mean_mode(mean_mode)
    mean_mode === :regime && cov_mode === :fixed ||
        throw(ArgumentError("$model supports root_mean_mode=:regime with root_cov_mode=:fixed. Other root conventions are retained only for restoring legacy results."))
    return nothing
end

function _validate_ou_theta0(mean_mode::Symbol, theta0)
    _validate_ou_root_mean_mode(mean_mode)
    theta0 === nothing || throw(ArgumentError("Independent theta0 is no longer supported; use the root regime optimum as the root mean."))
    return nothing
end

function _ou_public_spec(model::Symbol; root_mean_mode::Symbol, root_cov_mode::Symbol)
    if model in (:OU1, :OUM)
        _validate_shared_ou_root(model, root_mean_mode, root_cov_mode)
    else
        _validate_regime_ou_root(model, root_mean_mode, root_cov_mode)
    end
    return ou_spec(model; root_mean_mode, root_cov_mode)
end

function _validate_shared_ou_root(model::Symbol, mean_mode::Symbol, cov_mode::Symbol)
    _validate_ou_root_mean_mode(mean_mode)
    mean_mode in (:regime, :stationary) &&
        cov_mode in (:fixed, :random) ||
        throw(ArgumentError("$model supports root_mean_mode=:regime or :stationary and root_cov_mode=:fixed or :random. Legacy root modes are only retained for restoring saved results."))
    return nothing
end

function _ou_default_spec(model::Symbol)
    if model === :OU1
        return OUSpec(model = :OU1, theta_mode = :shared, alpha_mode = :shared, sigma_mode = :shared, root_mean_mode = :regime, root_cov_mode = :fixed)
    elseif model === :OUM
        return OUSpec(model = :OUM, theta_mode = :by_regime, alpha_mode = :shared, sigma_mode = :shared, root_mean_mode = :regime, root_cov_mode = :fixed)
    elseif model === :OUMV
        return OUSpec(model = :OUMV, theta_mode = :by_regime, alpha_mode = :shared, sigma_mode = :by_regime, root_mean_mode = :regime, root_cov_mode = :fixed)
    elseif model === :OUMA
        return OUSpec(model = :OUMA, theta_mode = :by_regime, alpha_mode = :by_regime, sigma_mode = :shared, root_mean_mode = :regime, root_cov_mode = :fixed)
    elseif model === :OUMVA
        return OUSpec(model = :OUMVA, theta_mode = :by_regime, alpha_mode = :by_regime, sigma_mode = :by_regime, root_mean_mode = :regime, root_cov_mode = :fixed)
    end
    throw(ArgumentError("Unsupported OU model $model"))
end

Base.@kwdef struct OUParameterBundle
    theta::Vector{Float64} = Float64[]
    alpha::Vector{Float64} = Float64[]
    sigma2::Vector{Float64} = Float64[]
    theta0::Union{Nothing, Float64} = nothing
end

struct OUMEdgeSegmentCache
    nregimes::Int
    root_regime::Int
    edge_first_segment::Vector{Int32}
    edge_last_segment::Vector{Int32}
    segment_states::Vector{Int32}
    segment_lengths::Vector{Float64}
end

"""Read-only inputs shared by all OUM-family starting points."""
struct OULikelihoodContext
    tree::CompactTree
    trait::Vector{Float64}
    spec::OUSpec
    cache::Union{Nothing,OUMEdgeSegmentCache}
    tip_index::Vector{Int}
end

"""Mutable arrays reused by one multistart candidate."""
mutable struct OULikelihoodWorkspace
    edge_a::Vector{Float64}
    edge_b::Vector{Float64}
    edge_v::Vector{Float64}
    precision::Vector{Float64}
    linear::Vector{Float64}
    logconst::Vector{Float64}
end

function OULikelihoodWorkspace(tree::CompactTree)
    return OULikelihoodWorkspace(
        zeros(Float64, tree.nedges),
        zeros(Float64, tree.nedges),
        zeros(Float64, tree.nedges),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
        zeros(Float64, tree.nnodes),
    )
end

@inline _ou_regime_value(mode::Symbol, values::Vector{Float64}, state::Integer) =
    mode === :shared ? values[1] : values[state]

function _ou_nparams(spec::OUSpec, nregimes::Integer)
    ntheta = spec.theta_mode === :shared ? 1 : nregimes
    nalpha = spec.alpha_mode === :shared ? 1 : nregimes
    nsigma = spec.sigma_mode === :shared ? 1 : nregimes
    return ntheta + nalpha + nsigma
end

function _ou_unpack_params(spec::OUSpec, pars::AbstractVector{<:Real}, nregimes::Integer)
    idx = 1
    nalphas = spec.alpha_mode === :shared ? 1 : nregimes
    nsigmas = spec.sigma_mode === :shared ? 1 : nregimes
    nthetas = spec.theta_mode === :shared ? 1 : nregimes

    length(pars) == nalphas + nsigmas + nthetas ||
        throw(ArgumentError("Parameter vector length does not match OUSpec $(spec.model)"))

    alpha = max.(Float64.(pars[idx:(idx + nalphas - 1)]), 1e-8)
    idx += nalphas
    sigma2 = max.(Float64.(pars[idx:(idx + nsigmas - 1)]), 1e-8)
    idx += nsigmas
    theta = Float64.(pars[idx:(idx + nthetas - 1)])
    return OUParameterBundle(theta = theta, alpha = alpha, sigma2 = sigma2)
end

function _ou_initial_params(
    spec::OUSpec,
    nregimes::Integer;
    tree::CompactTree,
    trait::AbstractVector{<:Real},
)
    tree_height = maximum(tree.dist_from_root)
    observed_trait = filter(!isnan, Float64.(trait))
    alpha_scalar = max(log(2.0) / max(tree_height / 4.0, 1e-8), 1e-8)
    sigma_scalar = max(var(observed_trait), 1e-8)

    alpha_init =
        if spec.alpha_mode === :shared
            [alpha_scalar]
        else
            fill(alpha_scalar, nregimes)
        end
    sigma_init =
        if spec.sigma_mode === :shared
            [sigma_scalar]
        else
            fill(sigma_scalar, nregimes)
        end

    theta_init =
        if spec.theta_mode === :shared
            [mean(observed_trait)]
        else
            fill(mean(observed_trait), nregimes)
        end

    parts = Vector{Float64}()
    append!(parts, alpha_init)
    append!(parts, sigma_init)
    append!(parts, theta_init)
    return parts
end

function _ou_initial_params_from_starts(
    spec::OUSpec,
    nregimes::Integer;
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    start_alpha::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_sigma2::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
    start_theta::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    init = _ou_initial_params(spec, nregimes; tree = tree, trait = trait)
    idx = 1
    nalphas = spec.alpha_mode === :shared ? 1 : nregimes
    nsigmas = spec.sigma_mode === :shared ? 1 : nregimes
    nthetas = spec.theta_mode === :shared ? 1 : nregimes

    if start_alpha !== nothing
        if start_alpha isa AbstractVector
            length(start_alpha) == nalphas || throw(ArgumentError("start_alpha must have $nalphas entries"))
            init[idx:(idx + nalphas - 1)] .= max.(Float64.(start_alpha), 1e-8)
        else
            init[idx:(idx + nalphas - 1)] .= max(Float64(start_alpha), 1e-8)
        end
    end
    idx += nalphas

    if start_sigma2 !== nothing
        if start_sigma2 isa AbstractVector
            length(start_sigma2) == nsigmas || throw(ArgumentError("start_sigma2 must have $nsigmas entries"))
            init[idx:(idx + nsigmas - 1)] .= max.(Float64.(start_sigma2), 1e-8)
        else
            init[idx:(idx + nsigmas - 1)] .= max(Float64(start_sigma2), 1e-8)
        end
    end
    idx += nsigmas

    if spec.theta_mode === :shared
        start_theta !== nothing && (init[idx] = Float64(start_theta))
    elseif start_theta_regimes !== nothing
        length(start_theta_regimes) == nthetas || throw(ArgumentError("start_theta_regimes must have $nthetas entries"))
        init[idx:(idx + nthetas - 1)] .= Float64.(start_theta_regimes)
    end
    return init
end

@inline _ou_result_theta(spec::OUSpec, bundle::OUParameterBundle, root_state::Float64) =
    spec.theta_mode === :shared ? bundle.theta[1] : root_state
@inline _ou_result_alpha(spec::OUSpec, bundle::OUParameterBundle) =
    spec.alpha_mode === :shared ? bundle.alpha[1] : NaN
@inline _ou_result_alpha_regimes(spec::OUSpec, bundle::OUParameterBundle) =
    spec.alpha_mode === :shared ? Float64[] : copy(bundle.alpha)

function _ou_multiregime_result(
    spec::OUSpec,
    bundle::OUParameterBundle,
    profile,
    nregimes::Integer;
    result = nothing,
    trait_name = nothing,
    regime_names = nothing,
)
    converged =
        if result === nothing
            false
        else
            _continuous_result_converged(result)
        end
    iterations =
        if result === nothing
            0
        else
            _continuous_result_iterations(result)
        end
    f_calls =
        if result === nothing
            0
        else
            _continuous_result_f_calls(result)
        end
    return ContinuousMultiRegimeResult(
        model = spec.model,
        success = profile.success,
        loglik = profile.loglik,
        aic = profile.success ? compute_aic(profile.loglik, _ou_nparams(spec, nregimes)) : Inf,
        nparams = _ou_nparams(spec, nregimes),
        trait_name = _continuous_checked_trait_name(trait_name),
        regime_names = _continuous_checked_regime_names(regime_names, nregimes),
        root_treatment = spec.root_cov_mode,
        root_mean_mode = spec.root_mean_mode,
        root_cov_mode = spec.root_cov_mode,
        sigma2 = copy(bundle.sigma2),
        alpha = _ou_result_alpha(spec, bundle),
        alpha_regimes = _ou_result_alpha_regimes(spec, bundle),
        theta = _ou_result_theta(spec, bundle, profile.root_state),
        theta_regimes = copy(bundle.theta),
        root_state = profile.root_state,
        theta0 = bundle.theta0,
        mean_diagnostics = hasproperty(profile, :mean_diagnostics) ? profile.mean_diagnostics : nothing,
        nregimes = nregimes,
        converged = converged,
        iterations = iterations,
        f_calls = f_calls,
    )
end

function _ou_bundle_from_result(fit::ContinuousMultiRegimeResult)
    _validate_ou_theta0(fit.root_mean_mode, fit.theta0)
    return OUParameterBundle(
        theta = fit.theta_regimes,
        alpha = isempty(fit.alpha_regimes) ? [fit.alpha] : fit.alpha_regimes,
        sigma2 = fit.sigma2,
        theta0 = fit.theta0,
    )
end
