using EvoTraits
using Test

function _categorical_mk_test_tree()
    path = joinpath(mktempdir(), "categorical_mk_tree.tre")
    write(path, "(((A:1,B:1):1,C:1):1,D:3);")
    return to_compact_tree(load_newick_tree(path))
end

function _assert_categorical_metadata(res)
    @test res.success
    @test res.nstates == 2
    @test res.state_labels == Any[:red, :blue]
    @test res.state_encoding[:red] == 1
    @test res.state_encoding[:blue] == 2
    @test length(res.ancestral_state_labels) == length(res.ancestral_states)
    @test all(label in res.state_labels for label in res.ancestral_state_labels)
end

@testset "Mk ASR categorical Dict tip states" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(
        labels[4] => :blue,
        labels[3] => :blue,
        labels[2] => :red,
        labels[1] => :red,
    )

    res = asr_mk(
        tree;
        tip_states = tip_states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
    )

    _assert_categorical_metadata(res)
end

@testset "Mk ASR categorical pair tip states" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = [
        labels[4] => :blue,
        labels[3] => :blue,
        labels[2] => :red,
        labels[1] => :red,
    ]

    res = asr_mk(
        tree;
        tip_states = tip_states,
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
    )

    _assert_categorical_metadata(res)
end

@testset "Mk ASR categorical input validation" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(labels[1] => :red, labels[2] => :red, labels[3] => :blue, labels[4] => :blue)
    missing_tip_states = Dict(labels[1] => :red, labels[2] => :red, labels[3] => :blue)

    @test_throws ArgumentError asr_mk(tree; tip_states = missing_tip_states)
    @test_throws ArgumentError asr_mk(tree; tip_states = tip_states, tip_priors = zeros(4, 2))
end

@testset "Mk ASR legacy integer metadata defaults" begin
    tree = _categorical_mk_test_tree()

    res = asr_mk(
        tree,
        2;
        tip_states = Int32[1, 1, 2, 2],
        rate_model = :ER,
        root_prior = :flat,
        max_iterations = 40,
    )

    @test res.success
    @test isempty(res.state_labels)
    @test isempty(res.state_encoding)
    @test isempty(res.ancestral_state_labels)
end

