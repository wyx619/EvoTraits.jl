using Test
using EvoTraits
using Random
using LinearAlgebra
using CSV
using DataFrames

const TEST_PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

include("core/load_and_tree.jl")
include("simulate/simulation.jl")

include("continuous/multivariate/simulation_and_assets.jl")
include("continuous/multivariate/bm_family.jl")
include("continuous/multivariate/ou_family.jl")

include("discrete/MK/mk.jl")
include("discrete/corHMM/corhmm.jl")
include("discrete/SIMMAP/simmap.jl")
include("tools/phyloref.jl")
include("tools/node_time_summary.jl")

include("continuous/univariate/bm_family.jl")
include("continuous/univariate/ou_family.jl")
include("continuous/univariate/eb_family.jl")

include("continuous/reconstruction/reconstruction.jl")
include("workflows.jl")


