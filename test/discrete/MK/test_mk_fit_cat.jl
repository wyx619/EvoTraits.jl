using EvoTraits
using Test

function _categorical_mk_test_tree()
    path = joinpath(mktempdir(), "categorical_mk_tree_mkfit.tre")
    write(path, "(((A:1,B:1):1,C:1):1,D:3);")
    return to_compact_tree(load_newick_tree(path))
end

function _assert_mkfit_metadata(res, state_labels)
    @test res.success
    @test res.nstates == length(state_labels)
    @test res.state_labels == state_labels
    for (idx, lbl) in enumerate(state_labels)
        @test res.state_encoding[lbl] == idx
    end
end

@testset "Mk fit categorical Dict tip states" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(
        labels[4] => :blue,
        labels[3] => :blue,
        labels[2] => :red,
        labels[1] => :red,
    )

    res = fit_mk(tree; tip_states = tip_states, rate_model = :ER, root_prior = :flat, max_iterations = 40)
    _assert_mkfit_metadata(res, Any[:red, :blue])
end

@testset "Mk fit categorical pair tip states" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = [
        labels[4] => :blue,
        labels[3] => :blue,
        labels[2] => :red,
        labels[1] => :red,
    ]

    res = fit_mk(tree; tip_states = tip_states, rate_model = :ER, root_prior = :flat, max_iterations = 40)
    _assert_mkfit_metadata(res, Any[:red, :blue])
end

@testset "Mk fit categorical input validation" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(labels[1] => :red, labels[2] => :red, labels[3] => :blue, labels[4] => :blue)
    missing_tip_states = Dict(labels[1] => :red, labels[2] => :red, labels[3] => :blue)

    @test_throws ArgumentError fit_mk(tree; tip_states = missing_tip_states)
    @test_throws ArgumentError fit_mk(tree; tip_states = tip_states, tip_priors = zeros(4, 2))
end

@testset "Mk ordered models require complete state_order" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(labels[1] => :A, labels[2] => :B, labels[3] => :C, labels[4] => :C)

    @test_throws ArgumentError fit_mk(tree; tip_states = tip_states, rate_model = :SRD, max_iterations = 20)
    @test_throws ArgumentError fit_mk(tree; tip_states = tip_states, rate_model = :SUEDE, max_iterations = 20)
    @test_throws ArgumentError fit_mk(tree; tip_states = tip_states, rate_model = :SRD, state_order = [:A, :B], max_iterations = 20)
    @test_throws ArgumentError fit_mk(tree; tip_states = tip_states, rate_model = :SRD, state_order = [:A, :B, :C, :D], max_iterations = 20)

    fit = fit_mk(
        tree;
        tip_states = tip_states,
        rate_model = :SRD,
        state_order = [:A, :B, :C],
        root_prior = :flat,
        max_iterations = 40,
    )
    @test fit.success
    @test fit.state_labels == Any[:A, :B, :C]
    @test fit.state_encoding[:A] == 1
    @test fit.state_encoding[:B] == 2
    @test fit.state_encoding[:C] == 3
    @test fit.transition_matrix[1, 3] == 0.0
    @test fit.transition_matrix[3, 1] == 0.0
end

@testset "Mk fit legacy integer metadata defaults" begin
    tree = _categorical_mk_test_tree()
    res = fit_mk(tree, 2; tip_states = Int32[1, 1, 2, 2], rate_model = :ER, root_prior = :flat, max_iterations = 40)
    @test res.success
    @test isempty(res.state_labels)
    @test isempty(res.state_encoding)
end

@testset "Mk fit show summary contains key markers" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(labels[4] => :blue, labels[3] => :blue, labels[2] => :red, labels[1] => :red)
    res = fit_mk(tree; tip_states = tip_states, rate_model = :ER, root_prior = :flat, max_iterations = 20)
    io = IOBuffer()
    show(io, res)
    out = String(take!(io))
    @test occursin("MkFitResult", out)
    @test occursin("success=true", out)
    @test occursin("states=2", out)
    @test occursin("loglik", out)
    @test occursin("aic", out)
    @test occursin("state_labels", out)
end

@testset "Mk ASR show summary contains key markers" begin
    tree = _categorical_mk_test_tree()
    labels = collect(tree.tip_labels)
    tip_states = Dict(labels[4] => :blue, labels[3] => :blue, labels[2] => :red, labels[1] => :red)
    res = asr_mk(tree; tip_states = tip_states, rate_model = :ER, root_prior = :flat, max_iterations = 20)
    io = IOBuffer()
    show(io, res)
    out = String(take!(io))
    @test occursin("MkASRResult", out)
    @test occursin("success=true", out)
    @test occursin("states=2", out)
    @test occursin("nodes=", out)
    @test occursin("reroot=", out)
    @test occursin("state_labels", out)
end

