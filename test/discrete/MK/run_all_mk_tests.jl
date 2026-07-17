using Test
using EvoTraits
using Random
using LinearAlgebra
using CSV
using DataFrames

const MK_TEST_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

@testset "Mk ASR on real tree (ER)" begin
    asset_root = joinpath(MK_TEST_ROOT, "validation", "seed_H")
    tree_path = joinpath(asset_root, "no_subshrub_phylo.tre")
    data_path = joinpath(asset_root, "no_subshrub_H.csv")

    big_tree = to_compact_tree(load_newick_tree(tree_path))
    df = CSV.read(data_path, DataFrame)
    tip_order = [string(t) for t in big_tree.tip_labels]
    growth_vec = [df[findfirst(==(tip), df.species), :growth_checked] for tip in tip_order]
    state_map = Dict("herb" => 1, "shrub" => 2, "tree" => 3)
    states = Int32[state_map[g] for g in growth_vec]

    asr = asr_mk(
        big_tree,
        3;
        tip_states = states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 100,
    )

    @test asr.success
    @test asr.nstates == 3
    @test size(asr.transition_matrix) == (3, 3)
    @test length(asr.node_ids) == big_tree.nnodes - big_tree.ntips
    @test size(asr.ancestral_likelihoods) == (big_tree.nnodes - big_tree.ntips, 3)
    @test all(isapprox.(sum(asr.ancestral_likelihoods; dims = 2), 1.0; atol = 1e-8))
    @test all(1 .<= asr.ancestral_states .<= 3)
    @test isfinite(asr.loglik)
    @test isfinite(asr.aic)
end

@testset "Mk ASR on real tree (ARD)" begin
    asset_root = joinpath(MK_TEST_ROOT, "validation", "seed_H")
    tree_path = joinpath(asset_root, "no_subshrub_phylo.tre")
    data_path = joinpath(asset_root, "no_subshrub_H.csv")

    big_tree = to_compact_tree(load_newick_tree(tree_path))
    df = CSV.read(data_path, DataFrame)
    tip_order = [string(t) for t in big_tree.tip_labels]
    growth_vec = [df[findfirst(==(tip), df.species), :growth_checked] for tip in tip_order]
    state_map = Dict("herb" => 1, "shrub" => 2, "tree" => 3)
    states = Int32[state_map[g] for g in growth_vec]

    asr = asr_mk(
        big_tree,
        3;
        tip_states = states,
        rate_model = :ARD,
        root_prior = :flat,
        max_iterations = 100,
    )

    @test asr.success
    @test asr.nstates == 3
    @test size(asr.transition_matrix) == (3, 3)
    @test length(asr.node_ids) == big_tree.nnodes - big_tree.ntips
    @test size(asr.ancestral_likelihoods) == (big_tree.nnodes - big_tree.ntips, 3)
    @test all(isapprox.(sum(asr.ancestral_likelihoods; dims = 2), 1.0; atol = 1e-8))
    @test all(1 .<= asr.ancestral_states .<= 3)
    @test isfinite(asr.loglik)
    @test isfinite(asr.aic)
end


