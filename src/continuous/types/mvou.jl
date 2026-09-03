"""
    MVOUPrecalc

Precomputed tree-level quantities shared across multivariate OU evaluations.
"""
Base.@kwdef struct MVOUPrecalc
    branch_lengths::Vector{Float64} = Float64[]
    edge_order::Vector{Int32} = Int32[]
    tip_terminal_regime::Vector{Int32} = Int32[]
    edge_segments::Vector{Vector{SimmapSegment}} = Vector{Vector{SimmapSegment}}()
    nregimes::Int = 1
    root_regime::Int = 1
    A_decomp::Symbol = :cholesky
    root_mean_mode::Symbol = :theta
    root_cov_mode::Symbol = :fixed
end

"""
    MVOUParameterBundle

Structured parameter bundle for multivariate OU-family models.
"""
Base.@kwdef struct MVOUParameterBundle
    theta::Vector{Float64} = Float64[]
    A::Matrix{Float64} = zeros(0, 0)
    A_regimes::Array{Float64, 3} = zeros(0, 0, 0)
    Sigma::Matrix{Float64} = zeros(0, 0)
    Sigma_regimes::Array{Float64, 3} = zeros(0, 0, 0)
    theta0::Union{Nothing, Vector{Float64}} = nothing
end

"""
    MVContinuousOUResult

Result type returned by multivariate OU-family fits.
"""
Base.@kwdef struct MVContinuousOUResult <: AbstractMVContinuousFitResult
    model::Symbol = :unknown
    success::Bool = false
    loglik::Float64 = NaN
    aic::Float64 = NaN
    nparams::Int = 0
    nobs::Int = 0
    ntraits::Int = 0
    nregimes::Int = 1
    trait_names::Vector{String} = String[]
    regime_names::Vector{String} = String[]
    theta::Matrix{Float64} = zeros(0, 0)
    A::Array{Float64, 3} = zeros(0, 0, 0)
    Sigma::Array{Float64, 3} = zeros(0, 0, 0)
    A_decomp::Symbol = :cholesky
    root_mean_mode::Symbol = :theta
    root_cov_mode::Symbol = :fixed
    converged::Bool = false
    iterations::Int = 0
    f_calls::Int = 0
end

function Base.summary(res::MVContinuousOUResult)
    return string(
        res.model,
        " multivariate OU fit",
        " (logLik=", round(res.loglik; digits = 6),
        ", AIC=", round(res.aic; digits = 6),
        ", nparams=", res.nparams,
        ")",
    )
end

function Base.show(io::IO, res::MVContinuousOUResult)
    print(io, summary(res))
end

function Base.show(io::IO, ::MIME"text/plain", res::MVContinuousOUResult)
    trait_names = isempty(res.trait_names) ? _mv_default_trait_names(res.ntraits) : res.trait_names
    regime_names = isempty(res.regime_names) ? _mv_default_regime_names(res.nregimes) : res.regime_names
    println(io, res.model, " multivariate OU fit")
    println(io, "  success:   ", res.success)
    println(io, "  converged: ", res.converged)
    println(io, "  logLik:    ", res.loglik)
    println(io, "  AIC:       ", res.aic)
    println(io, "  nparams:   ", res.nparams)
    println(io, "  traits:    ", join(trait_names, ", "))
    println(io, "  regimes:   ", join(regime_names, ", "))
    println(io, "  A decomp:  ", res.A_decomp)
    println(io, "  iterations:", res.iterations)
    println(io, "  f_calls:   ", res.f_calls)
    println(io)
    println(io, "theta by regime")
    for r in 1:res.nregimes
        _mv_print_vector_row(io, @view(res.theta[r, :]), regime_names[r], trait_names)
    end
    println(io)
    _mv_show_ou_matrix_blocks(io, "alpha (A)", res.A, trait_names, regime_names)
    println(io)
    _mv_show_ou_matrix_blocks(io, "sigma", res.Sigma, trait_names, regime_names)
end

function _mv_show_ou_matrix_blocks(io::IO, title::AbstractString, blocks::Array{Float64, 3}, trait_names::Vector{String}, regime_names::Vector{String})
    k = size(blocks, 3)
    if k == 1
        println(io, title)
        _mv_print_matrix(io, @view(blocks[:, :, 1]), trait_names, trait_names)
    else
        println(io, title, " by regime")
        for r in 1:k
            label = r <= length(regime_names) ? regime_names[r] : "regime$(r)"
            println(io, "  ", label)
            _mv_print_matrix(io, @view(blocks[:, :, r]), trait_names, trait_names; indent = "    ")
        end
    end
end
