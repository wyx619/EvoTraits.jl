include(joinpath(@__DIR__, "common.jl"))
using Test

data = documentation_data()
tree, X, sim = data.tree, data.X, data.sim
fit = fit_mvoum(tree, X, sim;
    A_decomp = :diagonal, max_iterations = 100,
    polish_iterations = 30, random_seed = 73,
    trait_names = ["trait1", "trait2"])
@test fit.success
@test size(fit.theta) == (2, 2)
@test size(fit.A, 1) == 2
asr = estim_node(tree, X, fit, sim)
@test asr.success
table = estim_node_table(tree, asr, fit)
@test :estimate_trait1 in propertynames(table)
println("Multivariate logLik: ", fit.loglik, "; converged: ", fit.converged)
println("A: ", fit.A[:, :, 1])
println("Sigma: ", fit.Sigma[:, :, 1])
println("theta (regime x trait): ", fit.theta)
