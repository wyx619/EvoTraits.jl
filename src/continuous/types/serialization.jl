function _restore_ou_root_names(values::NamedTuple)
    values.model in (:OU1, :OUM, :OUMV, :OUMA, :OUMVA) || return values
    # Migrate saved API names without changing legacy zero-root mean designs.
    mean_mode = values.root_mean_mode
    cov_mode = values.root_cov_mode
    modes = (
        root_mean_mode = mean_mode === :root_regime_theta ? :regime :
            mean_mode === :normalized_design ? :stationary : mean_mode,
        root_cov_mode = cov_mode === :stationary ? :random : cov_mode,
    )
    if haskey(values, :root_treatment)
        treatment = values.root_treatment === :stationary ? :random : values.root_treatment
        return merge(values, modes, (; root_treatment = treatment))
    end
    return merge(values, modes)
end

const _CONTINUOUS_FIT_LEGACY_FIELDS = (
    :model, :success, :loglik, :aic, :nparams, :trait_name, :root_treatment,
    :root_mean_mode, :root_cov_mode, :sigma2, :alpha, :beta, :theta,
    :root_state, :converged, :iterations, :f_calls,
)

function ContinuousFitResult(values::Vararg{Any,17})
    legacy = NamedTuple{_CONTINUOUS_FIT_LEGACY_FIELDS}(values)
    return ContinuousFitResult(; _restore_ou_root_names(legacy)...)
end

function Serialization.serialize(s::Serialization.AbstractSerializer, fit::ContinuousFitResult)
    Serialization.serialize_type(s, ContinuousFitResult)
    Serialization.serialize(s, :EvoTraitsContinuousFitV1)
    values = NamedTuple{fieldnames(ContinuousFitResult)}(
        ntuple(i -> getfield(fit, i), fieldcount(ContinuousFitResult)))
    Serialization.serialize(s, values)
    return nothing
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{ContinuousFitResult})
    first_value = Serialization.deserialize(s)
    first_value === :EvoTraitsContinuousFitV1 &&
        return ContinuousFitResult(; _restore_ou_root_names(Serialization.deserialize(s))...)
    values = ntuple(length(_CONTINUOUS_FIT_LEGACY_FIELDS)) do i
        i == 1 ? first_value : Serialization.deserialize(s)
    end
    legacy = NamedTuple{_CONTINUOUS_FIT_LEGACY_FIELDS}(values)
    return ContinuousFitResult(; _restore_ou_root_names(legacy)...)
end

# Legacy streams contain the original 22 fields without a version marker.
const _CONTINUOUS_MULTIREGIME_LEGACY_FIELDS = (
    :model, :success, :loglik, :aic, :nparams, :trait_name, :regime_names,
    :root_treatment, :root_mean_mode, :root_cov_mode, :sigma2, :alpha,
    :alpha_regimes, :beta, :beta_regimes, :theta, :theta_regimes,
    :root_state, :nregimes, :converged, :iterations, :f_calls,
)

function ContinuousMultiRegimeResult(values::Vararg{Any,22})
    legacy = NamedTuple{_CONTINUOUS_MULTIREGIME_LEGACY_FIELDS}(values)
    return ContinuousMultiRegimeResult(; _restore_ou_root_names(legacy)...)
end

function Serialization.serialize(s::Serialization.AbstractSerializer, fit::ContinuousMultiRegimeResult)
    Serialization.serialize_type(s, ContinuousMultiRegimeResult)
    Serialization.serialize(s, :EvoTraitsContinuousMultiRegimeV1)
    values = NamedTuple{fieldnames(ContinuousMultiRegimeResult)}(
        ntuple(i -> getfield(fit, i), fieldcount(ContinuousMultiRegimeResult)),
    )
    Serialization.serialize(s, values)
    return nothing
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{ContinuousMultiRegimeResult})
    first_value = Serialization.deserialize(s)
    if first_value === :EvoTraitsContinuousMultiRegimeV1
        values = Serialization.deserialize(s)
        return ContinuousMultiRegimeResult(; _restore_ou_root_names(values)...)
    end
    values = ntuple(length(_CONTINUOUS_MULTIREGIME_LEGACY_FIELDS)) do i
        i == 1 ? first_value : Serialization.deserialize(s)
    end
    legacy = NamedTuple{_CONTINUOUS_MULTIREGIME_LEGACY_FIELDS}(values)
    # Old free-root fits did not store theta0. Do not substitute its posterior.
    return ContinuousMultiRegimeResult(; _restore_ou_root_names(legacy)...)
end
