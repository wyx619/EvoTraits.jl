include(joinpath(@__DIR__, "common.jl"))
using Test

data = documentation_data()
tree, y, X = data.tree, data.y, data.X
ages = collect(0.0:0.25:10.0)
values = 18 .+ 3 .* sin.(ages ./ 2)
curve = smooth_env(ages, values; df = 8, scale = true)
envfit = fit_t_env(tree, y, curve;
    model = :EnvExp, max_iterations = 300, quadrature_order = 32)
@test envfit.success
@test isfinite(envfit.loglik)
println("Environmental fit: ", envfit.loglik, "; beta=", envfit.beta)

# beta = 0 must recover BM diffusion at fixed parameters.
constant_env = t_env_LL(tree, y, curve, 0.7, 0.0; model = :EnvExp)
@test isfinite(constant_env.loglik)
pre = shift_precalc(tree; age = 5.0, ntraits = size(X, 2))
shift = fit_mvshift(tree, X; model = :SR, precalc = pre,
    max_iterations = 200)
@test shift.success
@test isfinite(shift.loglik)
println("Shift fit: ", shift.loglik, "; converged: ", shift.converged)
