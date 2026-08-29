@testset "Simmap endpoints smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = serialize_tree(read_tree(tree_path))

    states = Int32[mod1(i, 4) for i in 1:big_tree.ntips]
    priors = tip_priors_from_states(big_tree, states, 4)
    Q = [
        -0.30 0.10 0.10 0.10;
         0.10 -0.30 0.10 0.10;
         0.10 0.10 -0.30 0.10;
         0.10 0.10 0.10 -0.30;
    ]

    sample = simmap_endpoints(big_tree, priors, Q; root_prior = :flat, rng = MersenneTwister(7))
    @test sample.success
    @test sample.nstates == 4
    @test length(sample.edge_end_states) == big_tree.nedges
end

@testset "Full simmap sampling" begin
    mk_tree_path = joinpath(mktempdir(), "toy_full_simmap_tree.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = serialize_tree(read_tree(mk_tree_path))

    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(toy_tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]

    simmap = simmap_sample(toy_tree, priors, Q; root_prior = :flat, rng = MersenneTwister(99))
    @test simmap.success
    @test simmap.nstates == 2
    @test length(simmap.edge_segments) == toy_tree.nedges
    @test size(simmap.mapped_edge) == (toy_tree.nedges, 2)
    summary = describe_simmap(toy_tree, simmap)
    @test summary.success
    @test summary.ntips == toy_tree.ntips
    @test summary.nedges == toy_tree.nedges
    @test summary.nstates == 2
    @test isapprox(summary.total_branch_length, sum(toy_tree.edge_length); atol = 1e-8)
    @test isapprox(sum(summary.mapped_lengths), summary.total_branch_length; atol = 1e-8)
    @test occursin("SimmapSample", sprint(show, simmap))
    @test occursin("changes are of the following types", sprint(show, MIME("text/plain"), summary))
    for edge in 1:toy_tree.nedges
        segsum = sum(seg.length for seg in simmap.edge_segments[edge])
        @test isapprox(segsum, toy_tree.edge_length[edge]; atol = 1e-8)
        @test isapprox(sum(simmap.mapped_edge[edge, :]), toy_tree.edge_length[edge]; atol = 1e-8)
    end
end

@testset "Simmap transition timing and rates" begin
    mk_tree_path = joinpath(mktempdir(), "toy_transition_simmap_tree.tre")
    write(mk_tree_path, "((A:1,B:1):1,C:2);")
    toy_tree = serialize_tree(read_tree(mk_tree_path))

    edge_segments = [
        [SimmapSegment(state = Int32(1), length = 0.25), SimmapSegment(state = Int32(2), length = 0.75)],
        [SimmapSegment(state = Int32(2), length = 1.0)],
        [SimmapSegment(state = Int32(1), length = 1.0)],
        [SimmapSegment(state = Int32(1), length = 2.0)],
    ]
    mapped_edge = zeros(Float64, toy_tree.nedges, 2)
    for edge in 1:toy_tree.nedges, seg in edge_segments[edge]
        mapped_edge[edge, seg.state] += seg.length
    end
    simmap = SimmapSample(
        success = true,
        nstates = 2,
        root_state = Int32(1),
        state_labels = ["red", "blue"],
        edge_segments = edge_segments,
        mapped_edge = mapped_edge,
    )

    events = transition_times(toy_tree, simmap)
    @test length(events) == 1
    @test events[1].edge == 1
    @test events[1].from_label == "red"
    @test events[1].to_label == "blue"
    @test events[1].transition == "red -> blue"
    @test isapprox(events[1].nodeheight, 0.25; atol = 1e-8)
    @test isapprox(events[1].time, 1.75; atol = 1e-8)

    replicate_events = transition_times(toy_tree, [simmap, simmap])
    @test length(replicate_events) == 2
    @test Set(event.sim_id for event in replicate_events) == Set([1, 2])

    df_single = transition_events_DataFrame(toy_tree, simmap)
    @test names(df_single) == ["sim_id", "edge_id", "from", "to", "time_before_present"]
    @test size(df_single, 1) == 1
    @test df_single.sim_id == [1]
    @test df_single.edge_id == [1]
    @test df_single.from == ["red"]
    @test df_single.to == ["blue"]
    @test isapprox(df_single.time_before_present[1], 1.75; atol = 1e-8)

    df_single_R = transition_events_DataFrame(toy_tree, simmap; R_order = :postorder)
    @test names(df_single_R) == ["sim_id", "edge_id", "from", "to", "time_before_present", "R_edge_id", "R_parent_node_id", "R_child_node_id", "branch_length", "tipX", "tipY", "descendant_signature"]
    @test df_single_R.R_edge_id == [3]
    @test df_single_R.tipX == ["A"]
    @test df_single_R.tipY == ["B"]
    @test df_single_R.descendant_signature == ["A|B"]

    df_multi = transition_events_DataFrame(toy_tree, [simmap, simmap])
    @test size(df_multi, 1) == 2
    @test Set(df_multi.sim_id) == Set([1, 2])
    @test all(df_multi.from .== "red")
    @test all(df_multi.to .== "blue")

    df_multi_R = transition_events_DataFrame(toy_tree, [simmap, simmap]; R_order = :cladewise)
    @test size(df_multi_R, 1) == 2
    @test all(df_multi_R.R_edge_id .== 1)
    @test all(df_multi_R.tipX .== "A")
    @test all(df_multi_R.tipY .== "B")

    rates = transition_rates_through_time(toy_tree, [simmap, simmap]; bin = 1.0)
    red_blue = filter(rate -> rate.transition == "red -> blue" && isapprox(rate.mya, 2.0; atol = 1e-8), rates)
    @test length(red_blue) == 1
    @test isapprox(red_blue[1].mean_count, 1.0; atol = 1e-8)
    @test red_blue[1].mean_start_branch_length > 0.0
    @test red_blue[1].mean_rate > 0.0
end

@testset "Friendly simmap_samples wrapper" begin
    mk_tree_path = joinpath(mktempdir(), "toy_friendly_simmap_tree.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = serialize_tree(read_tree(mk_tree_path))
    labels = collect(toy_tree.tip_labels)
    tip_states = Dict(labels[1] => :red, labels[2] => :red, labels[3] => :blue, labels[4] => :blue)

    simmaps = simmap_samples(
        toy_tree;
        tip_states = tip_states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
        nsim = 2,
        rng = MersenneTwister(123),
    )

    @test length(simmaps) == 2
    @test all(simmap.success for simmap in simmaps)
    @test all(simmap.state_labels == ["red", "blue"] for simmap in simmaps)
    @test all(length(simmap.edge_segments) == toy_tree.nedges for simmap in simmaps)

    one = simmap_sample(
        toy_tree;
        tip_states = tip_states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
        rng = MersenneTwister(321),
    )
    @test one.success
    @test one.state_labels == ["red", "blue"]
end

@testset "Prepared simmap sampler and fit-based sampling" begin
    mk_tree_path = joinpath(mktempdir(), "toy_prepared_simmap_tree.tre")
    write(mk_tree_path, "(((A:1,B:1):1,C:1):1,D:3);")
    toy_tree = serialize_tree(read_tree(mk_tree_path))
    labels = collect(toy_tree.tip_labels)
    tip_states = Dict(labels[1] => :red, labels[2] => :red, labels[3] => :blue, labels[4] => :blue)

    fit = fit_mk(
        toy_tree;
        tip_states = tip_states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
    )
    @test fit.success

    from_fit = simmap_samples(
        toy_tree,
        fit;
        tip_states = tip_states,
        nsim = 2,
        rng = MersenneTwister(404),
    )
    @test length(from_fit) == 2
    @test all(simmap.success for simmap in from_fit)
    @test all(simmap.state_labels == ["red", "blue"] for simmap in from_fit)

    encoded = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(toy_tree, encoded, 2)
    sampler = prepare_simmap_sampler(
        toy_tree,
        priors,
        fit.transition_matrix;
        root_prior = fit.root_prior,
        nparams = fit.nparams,
        state_labels = fit.state_labels,
    )
    @test sampler.cache.success
    @test sampler.state_labels == ["red", "blue"]

    prepared = simmap_samples(sampler; nsim = 3, rng = MersenneTwister(505))
    @test length(prepared) == 3
    @test all(simmap.success for simmap in prepared)
    @test all(length(simmap.edge_segments) == toy_tree.nedges for simmap in prepared)

    low_level = simmap_samples(
        toy_tree,
        priors,
        fit.transition_matrix;
        nsim = 2,
        root_prior = fit.root_prior,
        nparams = fit.nparams,
        state_labels = fit.state_labels,
        rng = MersenneTwister(606),
    )
    @test length(low_level) == 2
    @test all(simmap.state_labels == ["red", "blue"] for simmap in low_level)
end

@testset "Conditioned simmap branch respects reachable paths" begin
    Q = [
        -1.0 1.0 0.0;
         0.5 -1.0 0.5;
         0.0 1.0 -1.0;
    ]
    segs = EvoTraits._simulate_conditioned_branch!(
        MersenneTwister(2026),
        Q,
        1.0,
        Int32(1),
        Int32(3),
    )
    @test !isempty(segs)
    @test segs[end].state == 3
    @test any(seg.state == 2 for seg in segs)
    @test !any(seg.state == 3 && i < length(segs) - 1 for (i, seg) in enumerate(segs))
    @test isapprox(sum(seg.length for seg in segs), 1.0; atol = 1e-8)
end

@testset "Full simmap smoke test on real tree" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    big_tree = serialize_tree(read_tree(tree_path))

    states = Int32[mod1(i, 4) for i in 1:big_tree.ntips]
    priors = tip_priors_from_states(big_tree, states, 4)
    Q = [
        -0.30 0.10 0.10 0.10;
         0.10 -0.30 0.10 0.10;
         0.10 0.10 -0.30 0.10;
         0.10 0.10 0.10 -0.30;
    ]

    simmap = simmap_sample(big_tree, priors, Q; root_prior = :flat, rng = MersenneTwister(11))
    @test simmap.success
    @test simmap.nstates == 4
    @test length(simmap.edge_segments) == big_tree.nedges
    @test size(simmap.mapped_edge) == (big_tree.nedges, 4)
end

@testset "Phytools simmap phylip v1.5 read" begin
    project_root = TEST_PROJECT_ROOT
    simmap_path = joinpath(project_root, "validation", "multivariate", "VDH", "data", "VDH_R_mvOUM_simmap.tre")
    parsed = read_simmap(simmap_path; format = :phylip, version = 1.5, rev_order = true)

    tree = parsed.tree
    simmap = parsed.simmap

    @test tree.ntips > 1000
    @test tree.nedges == 2 * tree.ntips - 2
    @test simmap.success
    @test simmap.nstates == 4
    @test Set(simmap.state_labels) == Set(["ang_tracheid", "gym_tracheid", "scalariform", "simple"])
    @test length(simmap.edge_segments) == tree.nedges
    @test size(simmap.mapped_edge) == (tree.nedges, 4)
    summary = summary_simmap(tree, simmap)
    @test summary.ntips == tree.ntips
    @test summary.nedges == tree.nedges
    @test summary.nstates == 4
    @test summary.transition_count >= 0
    @test isapprox(sum(summary.mapped_proportions), 1.0; atol = 1e-8)
    @test occursin("mean total time spent in each state", sprint(show, MIME("text/plain"), summary))
    @test isapprox(sum(simmap.mapped_edge), sum(tree.edge_length); atol = 1e-6)
    for edge in 1:tree.nedges
        @test isapprox(sum(seg.length for seg in simmap.edge_segments[edge]), tree.edge_length[edge]; atol = 1e-6)
    end
    out_path = joinpath(mktempdir(), "roundtrip.simmap")
    text = write_simmap(tree, simmap; file = out_path, format = :phylip, version = 1.5, map_order = :right_to_left)
    @test isfile(out_path)
    @test !isempty(text)
    roundtrip = read_simmap(out_path; format = :phylip, version = 1.5, rev_order = true)
    @test roundtrip.tree.ntips == tree.ntips
    @test roundtrip.simmap.nstates == simmap.nstates
    @test isapprox(sum(roundtrip.simmap.mapped_edge), sum(simmap.mapped_edge); atol = 1e-5)
end

@testset "Simmap keep/drop tips preserves mapped histories" begin
    simmap_path = joinpath(TEST_PROJECT_ROOT, "validation", "multivariate", "VDH", "data", "VDH_R_mvOUM_simmap.tre")
    parsed = read_simmap(simmap_path; format = :phylip, version = 1.5, rev_order = true)
    tree = parsed.tree
    simmap = parsed.simmap

    keep = tree.tip_labels[1:100]
    kept = keep_tip_simmap(tree, simmap, keep)
    @test kept.tree.ntips == 100
    @test Set(kept.tree.tip_labels) == Set(keep)
    @test length(kept.simmap.edge_segments) == kept.tree.nedges
    @test size(kept.simmap.mapped_edge) == (kept.tree.nedges, kept.simmap.nstates)
    @test isapprox(sum(kept.simmap.mapped_edge), sum(kept.tree.edge_length); atol = 1e-6)
    for edge in 1:kept.tree.nedges
        @test isapprox(sum(seg.length for seg in kept.simmap.edge_segments[edge]), kept.tree.edge_length[edge]; atol = 1e-6)
    end

    dropped = drop_tip_simmap(tree, simmap, setdiff(tree.tip_labels, keep))
    @test dropped.tree.ntips == kept.tree.ntips
    @test Set(dropped.tree.tip_labels) == Set(kept.tree.tip_labels)
    @test isapprox(sum(dropped.simmap.mapped_edge), sum(kept.simmap.mapped_edge); atol = 1e-6)

    summary = summary_simmap(kept.tree, kept.simmap)
    @test summary.transition_count >= 0
    @test isapprox(sum(summary.mapped_lengths), sum(kept.tree.edge_length); atol = 1e-6)

    out_path = joinpath(mktempdir(), "kept.simmap")
    write_simmap(kept.tree, kept.simmap; file = out_path, format = :phylip, version = 1.5)
    roundtrip = read_simmap(out_path; format = :phylip, version = 1.5, rev_order = true)
    @test roundtrip.tree.ntips == kept.tree.ntips
    @test roundtrip.simmap.nstates <= kept.simmap.nstates
    @test isapprox(sum(roundtrip.simmap.mapped_edge), sum(kept.simmap.mapped_edge); atol = 1e-5)
end

@testset "Phytools simmap nexus write" begin
    tree_path = joinpath(mktempdir(), "toy_nexus_simmap_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]
    simmap = simmap_sample(tree, priors, Q; root_prior = :flat, rng = MersenneTwister(808))
    out_path = joinpath(mktempdir(), "roundtrip.nex")
    text = write_simmap(tree, simmap; file = out_path, format = :nexus, version = 1.5)
    @test isfile(out_path)
    @test occursin("#NEXUS", text)
    @test occursin("BEGIN TREES;", text)
    @test occursin("TRANSLATE", text)
end

@testset "Phytools 1:1 simmap format roundtrip (phylip/nexus × v1.0/v1.5)" begin
    # Reference format docs:
    #   phytools read.simmap / write.simmap — v1.0 uses "{state,length}" inside branches,
    #   v1.5 uses "[&map={state,length,...}]total_length". Nexus v1.0 uses "BEGIN SMPTREES;",
    #   nexus v1.5 uses "BEGIN TREES;". Both use "TRANSLATE" block before the tree line.
    tree_path = joinpath(mktempdir(), "toy_4fmt_tree.tre")
    write(tree_path, "(((A:1,B:1):1,(C:0.5,D:0.5):0.5):1);")
    tree = serialize_tree(read_tree(tree_path))
    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]
    simmap = simmap_sample(tree, priors, Q; root_prior = :flat, rng = MersenneTwister(808))

    tmpdir = mktempdir()
    cases = [
        (label = "phylip v1.0",  fmt = :phylip, ver = 1.0, block = nothing,  bracket = false),
        (label = "phylip v1.5",  fmt = :phylip, ver = 1.5, block = nothing,  bracket = true),
        (label = "nexus v1.0",   fmt = :nexus,  ver = 1.0, block = "BEGIN SMPTREES;", bracket = false),
        (label = "nexus v1.5",   fmt = :nexus,  ver = 1.5, block = "BEGIN TREES;",    bracket = true),
    ]

    for case in cases
        # ---- WRITE ----
        path = joinpath(tmpdir, "$(case.label).txt")
        text = write_simmap(tree, simmap; file = path, format = case.fmt, version = case.ver, map_order = :right_to_left)
        @test isfile(path)
        @test endswith(text, ";")  # every newick ends with ;
        if case.fmt === :nexus
            @test occursin("#NEXUS", text)
            @test occursin("TRANSLATE", text)
            @test occursin(case.block, text)
            @test occursin("TREE * UNTITLED = [&R]", text)
        end
        if case.bracket
            @test occursin("[&map={", text)
            @test !occursin(":{", text)  # v1.5 must NOT use v1.0 syntax
        else
            @test occursin(":{", text)
            @test !occursin("[&map={", text)  # v1.0 must NOT use v1.5 syntax
        end

        # ---- READ ----
        rt = read_simmap(path)
        @test rt.tree.ntips == tree.ntips
        @test rt.tree.nedges == tree.nedges
        @test rt.simmap.nstates == simmap.nstates
        @test Set(rt.simmap.state_labels) == Set(simmap.state_labels)
        @test isapprox(sum(rt.simmap.mapped_edge), sum(simmap.mapped_edge); atol = 1e-5)
        for e in 1:tree.nedges
            @test isapprox(sum(seg.length for seg in rt.simmap.edge_segments[e]), tree.edge_length[e]; atol = 1e-5)
        end
    end
end

@testset "Simmap nexus v1.5 with &map= inside branches (regression for [^=]* bug)" begin
    # Regression: tree_text = replace(line, r"^.*=\s*" => "") used to greedy-match
    # the LAST '=' in the line, which sat inside "[&map={...}]". Using "[^=]*"
    # confines the match to the first '=' (the TREE name separator).
    tree_path = joinpath(mktempdir(), "toy_bracket_eq_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]
    simmap = simmap_sample(tree, priors, Q; root_prior = :flat, rng = MersenneTwister(11))
    text = write_simmap(tree, simmap; format = :nexus, version = 1.5)
    @test occursin("[&map={", text)
    # Read from in-memory text (not file) to make sure extract+parse path works
    rt = read_simmap(text)
    @test rt.tree.ntips == tree.ntips
    @test rt.tree.nedges == tree.nedges
    @test rt.simmap.nstates == simmap.nstates
    @test isapprox(sum(rt.simmap.mapped_edge), sum(simmap.mapped_edge); atol = 1e-5)
end

@testset "Simmap v1.5 auto-detected from phylip file content" begin
    # When a file uses [&map={...}] syntax, version auto-detect should pick 1.5
    # even if user passes version=1.0.
    tree_path = joinpath(mktempdir(), "toy_autodetect_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    states = Int32[1, 1, 2, 2]
    priors = tip_priors_from_states(tree, states, 2)
    Q = [-0.3 0.3; 0.1 -0.1]
    simmap = simmap_sample(tree, priors, Q; root_prior = :flat, rng = MersenneTwister(22))
    text_v15 = write_simmap(tree, simmap; format = :phylip, version = 1.5)
    rt = read_simmap(text_v15; format = :phylip, version = 1.0)  # asks v1.0 but file is v1.5
    @test rt.tree.ntips == tree.ntips
    @test isapprox(sum(rt.simmap.mapped_edge), sum(simmap.mapped_edge); atol = 1e-5)
end






