using Test
using EvoTraits
using Random
using LinearAlgebra
using CSV
using DataFrames

const TEST_PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

const SUBSET_TESTS = Dict(
    "core/load_and_tree" => "core/load_and_tree.jl",
    "simulate/simulation" => "simulate/simulation.jl",
    "discrete/MK/mk" => "discrete/MK/mk.jl",
    "discrete/corHMM/corhmm" => "discrete/corHMM/corhmm.jl",
    "discrete/SIMMAP/simmap" => "discrete/SIMMAP/simmap.jl",
    "tools/phyloref" => "tools/phyloref.jl",
    "tools/node_time_summary" => "tools/node_time_summary.jl",
    "continuous/univariate/bm_family" => "continuous/univariate/bm_family.jl",
    "continuous/univariate/eb_family" => "continuous/univariate/eb_family.jl",
    "continuous/univariate/ou_family" => "continuous/univariate/ou_family.jl",
    "continuous/multivariate/bm_family" => "continuous/multivariate/bm_family.jl",
    "continuous/multivariate/ou_family" => "continuous/multivariate/ou_family.jl",
    "continuous/multivariate/real_validation" => "continuous/multivariate/real_validation.jl",
    "continuous/multivariate/simulation_and_assets" => "continuous/multivariate/simulation_and_assets.jl",
    "continuous/reconstruction/reconstruction" => "continuous/reconstruction/reconstruction.jl",
    "workflows" => "workflows.jl",
)

function _normalize_subset_arg(arg::AbstractString)
    normalized = replace(arg, '\\' => '/')
    normalized = startswith(normalized, "test/") ? normalized[6:end] : normalized
    normalized = endswith(normalized, ".jl") ? normalized[1:end - 3] : normalized
    return normalized
end

function _print_usage()
    println("Usage:")
    println("  julia --project=. test/run_subset.jl <subset> [<subset> ...]")
    println()
    println("Available subsets:")
    for name in sort!(collect(keys(SUBSET_TESTS)))
        println("  ", name)
    end
end

if isempty(ARGS)
    _print_usage()
    exit(1)
end

for raw_arg in ARGS
    subset = _normalize_subset_arg(raw_arg)
    if !haskey(SUBSET_TESTS, subset)
        println(stderr, "Unknown test subset: ", raw_arg)
        println(stderr)
        _print_usage()
        exit(1)
    end

    @testset "subset: $subset" begin
        include(joinpath(@__DIR__, SUBSET_TESTS[subset]))
    end
end
