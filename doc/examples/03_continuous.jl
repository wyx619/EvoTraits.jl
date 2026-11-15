include(joinpath(@__DIR__, "common.jl"))
using Test

data = documentation_data()
tree, y, sim = data.tree, data.y, data.sim
comparison = fit_compare(tree, y, [:BM1, :BMM, :OUM];
    edge_segments = sim.edge_segments,
    max_iterations = 300, rel_tol = 1e-6)
@test comparison.asr === nothing
@test comparison.best_fit !== nothing
println.(comparison.aic_table)

fit = fit_oum(tree, y, sim; max_iterations = 500,
    root_mean_mode = :regime, root_cov_mode = :fixed)
@test fit.success
asr = estim_node(tree, y, fit, sim)
@test asr.success
table = estim_node_table(tree, asr; simmap = sim)
summary = summarize_node_estimates_by_time(table, 4.0; w = 2.0)
@test haskey(summary, "trait")
@test summary["trait"].start[1] == summary["trait"].end[1] == 0
@test nrow(table) == length(asr.node_ids)
println("Converged: ", fit.converged)
println("Mean diagnostics: ", fit.mean_diagnostics)
println("Windows (excluding whole-tree baseline):")
println(summary["trait"][2:end, :])
