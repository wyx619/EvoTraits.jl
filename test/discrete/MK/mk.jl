@testset "Mk likelihood kernel" begin
    mk_tree_path = joinpath(mktempdir(), "toy_tree.tre")
    write(mk_tree_path, "((A:1,B:1)100.0:1,(C:1,D:1)95.0:1);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(toy_tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]

    fit_like = mk_loglikelihood(toy_tree, priors, Q; root_prior=:likelihoods)
    fit_flat = mk_loglikelihood(toy_tree, priors, Q; root_prior=:flat)

    @test fit_like.success
    @test fit_flat.success
    @test isfinite(fit_like.loglik)
    @test isfinite(fit_like.aic)
    @test fit_like.nstates == 2
    @test fit_like.nparams == 2
    @test fit_flat.loglik != fit_like.loglik
end

@testset "Mk likelihood smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = to_compact_tree(load_newick_tree(tree_path))

    states = Int32[mod1(i, 4) for i in 1:big_tree.ntips]
    priors = tip_priors_from_states(big_tree, states, 4)
    Q = [
        -0.30 0.10 0.10 0.10;
         0.10 -0.30 0.10 0.10;
         0.10 0.10 -0.30 0.10;
         0.10 0.10 0.10 -0.30;
    ]

    fit = mk_loglikelihood(big_tree, priors, Q; root_prior=:flat)
    @test fit.success
    @test isfinite(fit.loglik)
    @test fit.nstates == 4
end

@testset "Mk ARD fitting" begin
    mk_tree_path = joinpath(mktempdir(), "toy_fit_tree.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    fit = fit_mk(
        toy_tree,
        2;
        tip_states = states,
        rate_model = :ARD,
        root_prior = :flat,
        Ntrials = 2,
        Nscouts = 6,
        Nthreads = 1,
        max_iterations = 40,
    )

    @test fit.success
    @test fit.nstates == 2
    @test fit.nrates == 2
    @test length(fit.rates) == 2
    @test size(fit.transition_matrix) == (2, 2)
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
    @test length(fit.trial_logliks) == 2
    @test_throws ArgumentError fit_mk(
        toy_tree,
        2;
        tip_states = states,
        rate_model = :ARD,
        Ntrials = 2,
        Nscouts = 0,
        Nthreads = 1,
    )
    @test_throws ArgumentError fit_mk(
        toy_tree,
        2;
        tip_states = states,
        rate_model = :ARD,
        Ntrials = 2,
        Nscouts = 6,
        Nthreads = 3,
    )
end

@testset "Mk ER and SYM fitting" begin
    mk_tree_path = joinpath(mktempdir(), "toy_fit_tree_er_sym.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]

    fit_er = fit_mk(
        toy_tree,
        2;
        tip_states = states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
    )
    @test fit_er.success
    @test fit_er.nrates == 1
    @test length(fit_er.rates) == 1
    @test size(fit_er.transition_matrix) == (2, 2)
    @test isapprox(fit_er.transition_matrix[1, 2], fit_er.transition_matrix[2, 1]; atol = 1e-8)

    fit_sym = fit_mk(
        toy_tree,
        3;
        tip_states = Int32[1, 2, 2, 3],
        rate_model = :SYM,
        root_prior = :flat,
        max_iterations = 40,
    )
    @test fit_sym.success
    @test fit_sym.nrates == 3
    @test length(fit_sym.rates) == 3
    @test size(fit_sym.transition_matrix) == (3, 3)
    for i in 1:3, j in 1:3
        if i != j
            @test isapprox(fit_sym.transition_matrix[i, j], fit_sym.transition_matrix[j, i]; atol = 1e-8)
        end
    end

    @test mk_nrates(:ER, 4) == 1
    @test mk_nrates(:SYM, 4) == 6
    @test mk_nrates(:SUEDE, 4) == 2
    @test mk_nrates(:SRD, 4) == 6
    @test mk_nrates(:ARD, 4) == 12
    @test_throws ArgumentError mk_nrates(:BAD, 4)
end

@testset "Mk SUEDE and SRD fitting" begin
    mk_tree_path = joinpath(mktempdir(), "toy_fit_tree_suede_srd.tre")
    write(mk_tree_path, "((((A:1,B:1):1,C:1):1,D:1):1,E:4);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 2, 2, 3, 4]

    fit_suede = fit_mk(
        toy_tree,
        4;
        tip_states = states,
        rate_model = :SUEDE,
        root_prior = :flat,
        max_iterations = 40,
    )
    @test fit_suede.success
    @test fit_suede.nrates == 2
    @test length(fit_suede.rates) == 2
    for i in 1:4, j in 1:4
        if abs(i - j) > 1
            @test isapprox(fit_suede.transition_matrix[i, j], 0.0; atol = 1e-10)
        end
    end
    @test isapprox(fit_suede.transition_matrix[1, 2], fit_suede.transition_matrix[2, 3]; atol = 1e-8)
    @test isapprox(fit_suede.transition_matrix[2, 1], fit_suede.transition_matrix[3, 2]; atol = 1e-8)

    fit_srd = fit_mk(
        toy_tree,
        4;
        tip_states = states,
        rate_model = :SRD,
        root_prior = :flat,
        max_iterations = 40,
    )
    @test fit_srd.success
    @test fit_srd.nrates == 6
    @test length(fit_srd.rates) == 6
    for i in 1:4, j in 1:4
        if abs(i - j) > 1
            @test isapprox(fit_srd.transition_matrix[i, j], 0.0; atol = 1e-10)
        end
    end
end

@testset "Mk pruning cache and simmap endpoints" begin
    mk_tree_path = joinpath(mktempdir(), "toy_simmap_tree.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(toy_tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]

    cache = mk_pruning_cache(toy_tree, priors, Q; root_prior = :flat)
    @test cache.success
    @test isfinite(cache.loglik)
    @test size(cache.node_priors) == (toy_tree.nnodes, 2)
    @test size(cache.logpost) == (toy_tree.nnodes, 2)

    sample = simmap_endpoints(toy_tree, priors, Q; root_prior = :flat, rng = MersenneTwister(42))
    @test sample.success
    @test sample.nstates == 2
    @test length(sample.node_states) == toy_tree.nnodes
    @test length(sample.edge_start_states) == toy_tree.nedges
    @test length(sample.edge_end_states) == toy_tree.nedges
    @test all(1 .<= sample.node_states .<= 2)
    @test all(1 .<= sample.edge_start_states .<= 2)
    @test all(1 .<= sample.edge_end_states .<= 2)
end

@testset "Mk ASR fitting wrapper" begin
    mk_tree_path = joinpath(mktempdir(), "toy_asr_tree.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    asr = asr_mk(
        toy_tree,
        2;
        tip_states = states,
        rate_model = :ARD,
        root_prior = :flat,
        max_iterations = 40,
    )

    @test asr.success
    @test asr.nstates == 2
    @test size(asr.transition_matrix) == (2, 2)
    @test length(asr.node_ids) == toy_tree.nnodes - toy_tree.ntips
    @test size(asr.ancestral_likelihoods) == (toy_tree.nnodes - toy_tree.ntips, 2)
    @test length(asr.ancestral_states) == toy_tree.nnodes - toy_tree.ntips
    @test all(isapprox.(sum(asr.ancestral_likelihoods; dims = 2), 1.0; atol = 1e-8))
    @test all(1 .<= asr.ancestral_states .<= 2)
    @test isfinite(asr.loglik)
    @test isfinite(asr.aic)
end

@testset "Mk ASR with tip_priors" begin
    mk_tree_path = joinpath(mktempdir(), "toy_asr_tree2.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(toy_tree, states, 2)
    asr = asr_mk(toy_tree, 2; tip_priors = priors, rate_model = :ER, max_iterations = 40)
    @test asr.success
end

@testset "Mk ASR with empirical root prior" begin
    mk_tree_path = joinpath(mktempdir(), "toy_asr_tree3.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    asr = asr_mk(toy_tree, 2; tip_states = states, rate_model = :ER, root_prior = :empirical, max_iterations = 40)
    @test asr.success
end

@testset "Mk ASR reroot=false" begin
    mk_tree_path = joinpath(mktempdir(), "toy_asr_tree4.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    asr = asr_mk(toy_tree, 2; tip_states = states, rate_model = :ER, reroot = false, max_iterations = 40)
    @test asr.success
    @test !asr.reroot
    @test all(isapprox.(sum(asr.ancestral_likelihoods; dims = 2), 1.0; atol = 1e-8))
end

@testset "Mk ASR argument errors" begin
    mk_tree_path = joinpath(mktempdir(), "toy_asr_tree5.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = to_compact_tree(load_newick_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(toy_tree, states, 2)
    @test_throws ArgumentError asr_mk(toy_tree, 2; rate_model = :ER)
    @test_throws ArgumentError asr_mk(toy_tree, 2; tip_states = states, tip_priors = priors)
end

@testset "Mk ASR on real tree" begin
    project_root = TEST_PROJECT_ROOT
    asset_root = joinpath(project_root, "validation", "seed_H")
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





