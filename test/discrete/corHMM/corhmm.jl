@testset "corHMM state parsing with polymorphic and missing states" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))

    states = Dict("A" => "red", "B" => "red&blue", "C" => "?", "D" => "red&blue&yellow")
    fit = fit_corhmm(
        tree,
        states;
        model = :ARD,
        rate_cat = 1,
        root_prior = :flat,
        state_order = ["red", "blue", "yellow"],
        max_iterations = 40,
    )

    @test fit.success
    @test fit.observed_labels == ["red", "blue", "yellow"]
    @test fit.hidden_labels == ["red", "blue", "yellow"]
    @test fit.rate_cat == 1
    @test fit.diagnostics[:polymorphic_tips] == 2
    @test fit.diagnostics[:missing_tips] == 1
    @test size(fit.tip_priors_observed) == (tree.ntips, 3)
    @test isapprox(sum(fit.tip_priors_observed[2, :]), 2.0; atol = 1e-8)
    @test fit.tip_priors_observed[3, :] == [1.0, 1.0, 1.0]
    @test vec(sum(fit.tip_priors_hidden; dims = 2)) == [1.0, 2.0, 3.0, 3.0]
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
end

@testset "corHMM DataFrame and vector inputs" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_df_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))

    df = DataFrame(taxon = ["A", "B", "C", "D"], state = ["x", "y", missing, "x&y"])
    fit_df = fit_corhmm(tree, df; model = :ER, rate_cat = 1, root_prior = :flat, max_iterations = 40)
    @test fit_df.success
    @test fit_df.observed_labels == ["x", "y"]
    @test fit_df.diagnostics[:missing_tips] == 1
    @test fit_df.diagnostics[:polymorphic_tips] == 1

    fit_vec = fit_corhmm(tree, ["x", "y", "x", "y"]; model = :ER, root_prior = :flat, max_iterations = 40)
    @test fit_vec.success
    @test fit_vec.observed_labels == ["x", "y"]

    @test_throws ArgumentError fit_corhmm(tree, df; model = :ER, collapse = false, max_iterations = 10)
    fit_possible = fit_corhmm(tree, df; model = :ER, collapse = false, state_order = ["x", "y", "z"], root_prior = :flat, max_iterations = 20)
    @test fit_possible.success
    @test fit_possible.observed_labels == ["x", "y", "z"]
end

@testset "corHMM rate_cat hidden states and Mk model families" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_hidden_tree.tre")
    write(tree_path, "((((A:1,B:1):1,C:1):1,D:1):1,E:4);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "s1", "B" => "s2", "C" => "s2", "D" => "s3", "E" => "s1")

    for model in (:ER, :SYM, :ARD, :SUEDE, :SRD)
        fit = fit_corhmm(
            tree,
            states;
            model = model,
            rate_cat = 2,
            root_prior = :flat,
            state_order = ["s1", "s2", "s3"],
            max_iterations = 30,
        )
        @test fit.success
        @test fit.rate_cat == 2
        @test length(fit.hidden_labels) == 6
        @test fit.hidden_to_observed == [1, 2, 3, 1, 2, 3]
        @test size(fit.transition_matrix) == (6, 6)
        @test size(fit.index_matrix) == (6, 6)
        expected_np = 2 * mk_nrates(model, 3) + 2
        @test maximum(fit.index_matrix) == expected_np
        @test all(diag(fit.index_matrix) .== 0)
        @test vec(sum(fit.tip_priors_hidden; dims = 2)) == fill(2.0, tree.ntips)
    end
end

@testset "corHMM ASR modes" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_asr_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "red", "B" => "red", "C" => "blue", "D" => "?")

    fit = fit_corhmm(tree, states; model = :ER, rate_cat = 2, root_prior = :flat, node_states = :none, max_iterations = 40)
    @test fit.success
    @test fit.asr === nothing

    for mode in (:marginal, :scaled, :joint)
        asr = asr_corhmm(fit; mode = mode)
        @test asr.success
        @test asr.mode == mode
        @test length(asr.node_ids) == tree.nnodes - tree.ntips
        @test asr.node_ids == [Int(n) for n in tree.preorder if !tree.is_tip[n]]
        @test size(asr.hidden_likelihoods, 2) == 4
        @test size(asr.observed_likelihoods, 2) == 2
        @test length(asr.hidden_states) == length(asr.node_ids)
        @test length(asr.observed_states) == length(asr.node_ids)
        @test all(isapprox.(sum(asr.observed_likelihoods; dims = 2), 1.0; atol = 1e-8))
        if mode === :joint
            @test length(asr.tip_hidden_states) == tree.ntips
            @test length(asr.tip_observed_states) == tree.ntips
            @test isfinite(asr.joint_loglik)
        end
    end

    none_asr = asr_corhmm(fit; mode = :none)
    @test none_asr.success
    @test isempty(none_asr.node_ids)
end

@testset "corHMM SIMMAP sampling and collapse" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_simmap_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "red", "B" => "red&blue", "C" => "blue", "D" => "?")

    fit = fit_corhmm(tree, states; model = :ER, rate_cat = 2, root_prior = :flat, node_states = :none, max_iterations = 40)
    sims = simmap_corhmm(fit; nsim = 2, rng = MersenneTwister(123))

    @test sims.success
    @test length(sims.samples) == 2
    @test length(sims.collapsed_samples) == 2
    @test all(sample.nstates == 4 for sample in sims.samples)
    @test all(sample.nstates == 2 for sample in sims.collapsed_samples)
    @test all(sample.state_labels == ["blue", "red"] for sample in sims.collapsed_samples)
    for sample in sims.collapsed_samples
        @test size(sample.mapped_edge) == (tree.nedges, 2)
        @test length(sample.edge_segments) == tree.nedges
        for edge in 1:tree.nedges
            @test isapprox(sum(seg.length for seg in sample.edge_segments[edge]), tree.edge_length[edge]; atol = 1e-8)
            @test isapprox(sum(@view sample.mapped_edge[edge, :]), tree.edge_length[edge]; atol = 1e-8)
        end
        @test isapprox(sum(sample.mapped_edge), sum(tree.edge_length); atol = 1e-6)
    end
    @test_throws ArgumentError simmap_corhmm(fit; max_attempt = 0)
end

@testset "corHMM SIMMAP applies the Yang root prior once" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_root_prior_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "red", "B" => "red", "C" => "blue", "D" => "blue")
    fit = fit_corhmm(
        tree,
        states;
        model = :ARD,
        rate_cat = 1,
        root_prior = :yang,
        node_states = :none,
        max_iterations = 40,
    )

    @test fit.success
    raw = EvoTraits.corhmm_pruning_cache(
        fit.tree,
        fit.tip_priors_hidden,
        fit.transition_matrix;
        root_prior = EvoTraits._corhmm_root_prior(fit.root_prior),
        nparams = fit.nparams,
        rate_cat = fit.rate_cat,
        branch_lengths = fit.branch_lengths,
    )
    conditional = EvoTraits._corhmm_conditional_node_likelihoods(fit)
    root = fit.tree.root
    @test conditional.node_liks[root, :] ≈ raw.node_liks[root, :]

    expected = raw.node_liks[root, :] .* raw.root_prior_probs
    expected ./= sum(expected)
    evals, V, Vinv = EvoTraits._mk_eigen_cache(conditional.transition_matrix)
    counts = zeros(Int, length(expected))
    rng = MersenneTwister(20260826)
    for _ in 1:5_000
        endpoints = EvoTraits.sample_conditioned_endpoints(
            fit.tree,
            conditional.node_liks,
            conditional.transition_matrix;
            root_prior_probs = conditional.root_prior_probs,
            branch_lengths = fit.branch_lengths,
            rng = rng,
            evals = evals,
            V = V,
            Vinv = Vinv,
        )
        counts[Int(endpoints.root_state)] += 1
    end
    @test all(isapprox.(counts ./ sum(counts), expected; atol = 0.025))
end

@testset "corHMM zero branch length adjustment" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_zero_tree.tre")
    write(tree_path, "((A:0,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "x", "B" => "y", "C" => "x", "D" => "y")
    fit = fit_corhmm(tree, states; model = :ER, root_prior = :flat, node_states = :none, max_iterations = 20)
    @test fit.success
    @test any(fit.branch_lengths .> tree.edge_length)
    @test all(fit.branch_lengths .>= tree.edge_length)
end

@testset "corHMM rate_cat=1 equivalence to Mk tip-prior fitting" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_equiv_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "red", "B" => "red", "C" => "blue", "D" => "blue")

    cor = fit_corhmm(tree, states; model = :ER, rate_cat = 1, root_prior = :flat, node_states = :none, max_iterations = 40)
    mk_like = mk_loglikelihood(tree, cor.tip_priors_hidden, cor.transition_matrix; root_prior = :flat, nparams = cor.nparams)

    @test cor.success
    @test mk_like.success
    @test isapprox(cor.loglik, mk_like.loglik; atol = 1e-8)
    @test isapprox(cor.aic, mk_like.aic; atol = 1e-8)
end

@testset "corHMM order test and custom rate matrix" begin
    idx = EvoTraits.rateindex(2; model = :ARD, rate_cat = 2)
    q_bad = EvoTraits.qfromindex([0.1, 0.1, 0.2, 0.2, 0.4, 0.4], idx)
    @test !EvoTraits.corhmm_order_test(q_bad, 2)
    q_good = EvoTraits.qfromindex([0.8, 0.8, 0.4, 0.4, 0.2, 0.2], idx)
    @test EvoTraits.corhmm_order_test(q_good, 2)

    custom = [0 1; 2 0]
    norm = EvoTraits.normalize_ratematrix(custom)
    @test norm == custom
    Q = EvoTraits.qfromindex([0.3, 0.7], norm)
    @test Q == [-0.3 0.3; 0.7 -0.7]
end

@testset "corHMM hidden-rate index matrix structure" begin
    idx = EvoTraits.rateindex(2; model = :ARD, rate_cat = 2)
    @test idx == [
        0 2 6 0;
        1 0 0 6;
        5 0 0 4;
        0 5 3 0;
    ]

    idx_er = EvoTraits.rateindex(3; model = :ER, rate_cat = 2)
    @test maximum(idx_er) == 4
    @test idx_er[1, 2] == idx_er[1, 3] == 1
    @test idx_er[4, 5] == idx_er[4, 6] == 2
    @test idx_er[1, 4] == idx_er[2, 5] == idx_er[3, 6] == 4
    @test idx_er[4, 1] == idx_er[5, 2] == idx_er[6, 3] == 3
end

@testset "corHMM fixed nodes, tip fog, and Lewis correction" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_update_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))
    states = Dict("A" => "red", "B" => "red", "C" => "blue", "D" => "blue")

    base = fit_corhmm(tree, states; model = :ER, root_prior = :flat, node_states = :none, max_iterations = 30)
    fixed_fog = fit_corhmm(tree, states; model = :ER, root_prior = :flat, node_states = :none, tip_fog = 0.05, max_iterations = 30)
    estimated_fog = fit_corhmm(tree, states; model = :ER, root_prior = :flat, node_states = :none, tip_fog = [1, 1], max_iterations = 30)

    @test base.success
    @test fixed_fog.success
    @test fixed_fog.tip_fog == [0.05, 0.05]
    @test fixed_fog.nparams == base.nparams
    @test estimated_fog.success
    @test length(estimated_fog.tip_fog) == 2
    @test estimated_fog.nparams == base.nparams + 1
    @test all(0.0 <= p < 0.5 for p in estimated_fog.tip_fog)

    fixed_node = first(tree.postorder_internal)
    fixed_label = base.observed_labels[1]
    fixed = fit_corhmm(
        tree,
        states;
        model = :ER,
        root_prior = :flat,
        node_states = :marginal,
        fixed_nodes = Dict(Int(fixed_node) => fixed_label),
        max_iterations = 30,
    )
    @test fixed.success
    @test fixed.diagnostics[:fixed_node_states][Int(fixed_node)] == 1
    row = findfirst(==(Int(fixed_node)), fixed.asr.node_ids)
    @test row !== nothing
    @test fixed.asr.observed_likelihoods[row, 1] == 1.0
    @test fixed.asr.observed_likelihoods[row, 2] == 0.0

    hidden_fixed = fit_corhmm(
        tree,
        states;
        model = :ER,
        rate_cat = 2,
        root_prior = :flat,
        node_states = :marginal,
        fixed_nodes = Dict(Int(fixed_node) => fixed_label),
        max_iterations = 30,
    )
    @test hidden_fixed.success
    row_hidden = findfirst(==(Int(fixed_node)), hidden_fixed.asr.node_ids)
    @test hidden_fixed.asr.observed_likelihoods[row_hidden, 1] == 1.0
    @test hidden_fixed.asr.observed_likelihoods[row_hidden, 2] == 0.0

    state_data = EvoTraits.parsestates(tree, states; state_order = ["blue", "red"])
    Q = EvoTraits.qfromindex([0.3], EvoTraits.rateindex(2; model = :ER))
    raw = EvoTraits.corhmm_pruning_cache(tree, state_data.tip_priors_hidden, Q; root_prior = :yang, nparams = 1)
    corrected = EvoTraits.corhmm_pruning_cache(tree, state_data.tip_priors_hidden, Q; root_prior = :yang, nparams = 1, lewis_asc_bias = true)
    @test raw.success
    @test corrected.success
    @test corrected.loglik < raw.loglik
    @test corrected.aic == -2 * corrected.loglik + 2
end

@testset "corHMM combined multi-character hidden-rate features" begin
    tree_path = joinpath(mktempdir(), "toy_corhmm_combined_tree.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    tree = serialize_tree(read_tree(tree_path))
    states = DataFrame(
        taxon = ["A", "B", "C", "D"],
        growth = ["herb", "shrub", "tree", "herb"],
        plate = ["scalariform", "scalariform", "simple", "simple"],
    )
    order = [["herb", "shrub", "tree"], ["scalariform", "simple"]]

    hidden_fog = fit_corhmm(
        tree,
        states;
        model = :ER,
        rate_cat = 2,
        state_order = order,
        root_prior = :flat,
        node_states = :none,
        tip_fog = [1, 2, 1, 2],
        max_iterations = 20,
    )
    @test hidden_fog.success
    @test hidden_fog.diagnostics[:multi_character]
    @test hidden_fog.diagnostics[:tip_fog_groups] == [1, 2, 1, 2]
    @test length(hidden_fog.tip_fog) == length(hidden_fog.observed_labels)
    @test hidden_fog.nparams == maximum(hidden_fog.index_matrix) + 2

    fixed_node = first(tree.postorder_internal)
    fixed = fit_corhmm(
        tree,
        states;
        model = :ER,
        rate_cat = 2,
        state_order = order,
        root_prior = :flat,
        node_states = :marginal,
        fixed_nodes = Dict(Int(fixed_node) => "herb_scalariform"),
        max_iterations = 20,
    )
    @test fixed.success
    @test fixed.diagnostics[:multi_character]
    @test fixed.diagnostics[:fixed_node_states][Int(fixed_node)] == findfirst(==("herb_scalariform"), fixed.observed_labels)
    fixed_row = findfirst(==(Int(fixed_node)), fixed.asr.node_ids)
    @test fixed.asr.observed_likelihoods[fixed_row, fixed.diagnostics[:fixed_node_states][Int(fixed_node)]] == 1.0

    state_data = EvoTraits.parsestates(tree, states; state_order = order, rate_cat = 2)
    index_matrix = EvoTraits.rateindex(length(state_data.observed_labels); model = :ER, rate_cat = 2)
    Q = EvoTraits.qfromindex(fill(0.3, maximum(index_matrix)), index_matrix)
    raw = EvoTraits.corhmm_pruning_cache(
        tree,
        state_data.tip_priors_hidden,
        Q;
        root_prior = :yang,
        nparams = maximum(index_matrix),
        rate_cat = 2,
        hidden_to_observed = state_data.hidden_to_observed,
    )
    corrected = EvoTraits.corhmm_pruning_cache(
        tree,
        state_data.tip_priors_hidden,
        Q;
        root_prior = :yang,
        nparams = maximum(index_matrix),
        rate_cat = 2,
        hidden_to_observed = state_data.hidden_to_observed,
        lewis_asc_bias = true,
    )
    @test raw.success
    @test corrected.success
    @test isfinite(raw.loglik)
    @test isfinite(corrected.loglik)
    @test corrected.loglik < raw.loglik
end
