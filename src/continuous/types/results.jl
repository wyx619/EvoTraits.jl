"""
    ContinuousFitResult

Result type returned by single-trait continuous model fits such as `BM1`,
`OU1`, and `EB`.
"""
Base.@kwdef struct ContinuousFitResult <: AbstractContinuousFitResult
    model::Symbol = :unknown
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    trait_name::String = "trait"
    root_treatment::Symbol = :unknown
    root_mean_mode::Symbol = :unknown
    root_cov_mode::Symbol = :unknown
    sigma2::Float64 = NaN
    alpha::Float64 = NaN
    beta::Float64 = NaN
    theta::Float64 = NaN
    root_state::Float64 = NaN
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end

"""
    ContinuousMultiRegimeResult

Result type returned by multi-regime continuous model fits such as `BMM`,
`OUM`, `OUMV`, `OUMA`, `OUMVA`.
"""
Base.@kwdef struct ContinuousMultiRegimeResult <: AbstractContinuousFitResult
    model::Symbol = :unknown
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    trait_name::String = "trait"
    regime_names::Vector{String} = String[]
    root_treatment::Symbol = :unknown
    root_mean_mode::Symbol = :unknown
    root_cov_mode::Symbol = :unknown
    sigma2::Vector{Float64} = Float64[]
    alpha::Float64 = NaN
    alpha_regimes::Vector{Float64} = Float64[]
    beta::Float64 = NaN
    beta_regimes::Vector{Float64} = Float64[]
    theta::Float64 = NaN
    theta_regimes::Vector{Float64} = Float64[]
    root_state::Float64 = NaN
    nregimes::Int = 0
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end

"""
    MVContinuousBMResult

Result type returned by multivariate Brownian-motion fits such as `mvBM1`.
"""
Base.@kwdef struct MVContinuousBMResult <: AbstractMVContinuousFitResult
    model::Symbol = :unknown
    backend::Symbol = :unknown
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    nobs::Int = 0
    ntraits::Int = 0
    trait_names::Vector{String} = String[]
    sigma::Matrix{Float64} = zeros(0, 0)
    theta::Vector{Float64} = Float64[]
    beta::Float64 = NaN
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end

"""
    MVContinuousMultiBMResult

Result type returned by multi-regime multivariate Brownian-motion fits such as
`mvBMM`.
"""
Base.@kwdef struct MVContinuousMultiBMResult <: AbstractMVContinuousFitResult
    model::Symbol = :unknown
    backend::Symbol = :unknown
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    nobs::Int = 0
    ntraits::Int = 0
    nregimes::Int = 0
    trait_names::Vector{String} = String[]
    regime_names::Vector{String} = String[]
    sigma::Array{Float64, 3} = zeros(0, 0, 0)
    theta::Matrix{Float64} = zeros(0, 0)
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end
