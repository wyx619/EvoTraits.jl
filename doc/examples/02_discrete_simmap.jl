include(joinpath(@__DIR__, "common.jl"))
using Test

data = documentation_data()
tree = data.tree
states = Dict(label => (iseven(i) ? "open" : "closed")
              for (i, label) in enumerate(tree.tip_labels))
states[tree.tip_labels[1]] = "open&closed"
states[tree.tip_labels[2]] = "?"
workers = min(8, Threads.nthreads())
fit = fit_corhmm(tree, states;
    model = :ARD, state_order = ["open", "closed"],
    node_states = :marginal, Ntrials = workers, Nthreads = workers,
    max_iterations = 2000, rng = MersenneTwister(73))
@test isfinite(fit.loglik)
@test fit.success
posterior = asr_corhmm(fit; mode = :marginal)
root_row = findfirst(==(Int(tree.root)), posterior.node_ids)
@test root_row !== nothing
@test sum(posterior.observed_likelihoods[root_row, :]) ≈ 1.0
println("Root prior: ", fit.root_prior_probs)
println("Root posterior: ", posterior.observed_likelihoods[root_row, :])

maps = simmap_corhmm(fit; nsim = 2, rng = MersenneTwister(2026))
@test maps.success
mktempdir() do dir
    path = joinpath(dir, "histories.simmap")
    open(path, "w") do io
        for sim in maps.collapsed_samples
            println(io, write_simmap(tree, sim))
        end
    end
    parsed = read_simmap(path)
    @test length(parsed) == 2
    @test all(item -> summary_simmap(item.tree, item.simmap).success, parsed)
end
println("Discrete fitting, root posterior and multi-history SIMMAP I/O verified.")
