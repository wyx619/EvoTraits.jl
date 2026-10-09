include("env_smoothing.jl")
include("env_cache.jl")

@testset "Environmental BM" begin
    tree = EvoTraits._parse_load_tree_text("((a:1,b:1):2,(c:2,d:2):1);")
    y = [0.4,-0.7,1.2,0.1]
    curve = EnvCurve([3.0,0.0,1.0],[3.0,0.0,1.0])
    @test curve(0.5) ≈ 0.5
    @test curve(3.0) == 3.0
    @test EnvCurve([0.,3.],[10.,20.];scale=true)(1.5) ≈ 0.5
    @test_throws ArgumentError EnvCurve([0.,0.],[1.,2.])
    @test_throws ArgumentError EnvCurve([0.,1.],[1.,1.];scale=true)
    @test_throws ArgumentError curve(-0.1)
    @test_throws ArgumentError t_env_loglikelihood(tree,y,EnvCurve([0.,2.],[0.,1.]),0.7,0.2)
    @test_throws ArgumentError t_env_loglikelihood(tree,y,t -> NaN,0.7,0.2)
    @test_throws ArgumentError t_env_loglikelihood(tree,y,curve,0.7,0.2;model=:bad)
    for model in (:EnvExp,:EnvLin)
        constant = t_env_loglikelihood(tree,y,t -> 0.5,0.7,model === :EnvExp ? 0.0 : 0.7;model)
        bm = bm1_loglikelihood(tree,y,0.7)
        @test constant.loglik ≈ bm.loglik atol=1e-12
        @test constant.theta ≈ bm.theta atol=1e-12
        @test constant.nparams == 3
        @test constant.aic ≈ bm.aic+2
    end
    for order in (8,16,32), model in (:EnvExp,:EnvLin)
        environment = model === :EnvExp ? curve : EnvCurve([0.,1.,3.],[0.,1/3,1.])
        result = t_env_loglikelihood(tree,y,environment,0.7,0.2;model,quadrature_order=order)
        @test result.success
        expected = Float64[]
        for edge in 1:tree.nedges
            lo = 3-tree.dist_from_root[tree.child_of_edge[edge]]
            hi = lo+tree.edge_length[edge]
            value = model === :EnvExp ? 0.7/0.2*(exp(0.2*hi)-exp(0.2*lo)) :
                0.7*(hi-lo)+(0.2-0.7)/6*(hi^2-lo^2)
            push!(expected,value)
        end
        @test result.edge_variances ≈ expected atol=1e-12
        # Independent dense covariance for this small test only: shared root paths.
        paths = zeros(tree.ntips,tree.nedges)
        for (i,tip) in enumerate(tree.tip_ids)
            node = tip
            while node != tree.root
                edge = findfirst(==(node),tree.child_of_edge)
                paths[i,edge] = 1
                node = tree.parent_of_edge[edge]
            end
        end
        V = paths*Diagonal(expected)*paths'
        for values in (y,[NaN,-0.7,1.2,0.1])
            selected = findall(!isnan,values)
            F = cholesky(Symmetric(V[selected,selected]))
            one = ones(length(selected))
            root = dot(one,F\values[selected])/dot(one,F\one)
            residual = values[selected] .- root
            ll = -0.5*(dot(residual,F\residual)+logdet(F)+length(selected)*log(2pi))
            profile = t_env_loglikelihood(tree,values,environment,0.7,0.2;model,quadrature_order=order)
            @test profile.loglik ≈ ll atol=1e-11
            @test profile.theta ≈ root atol=1e-12
        end
    end
    invalid = t_env_loglikelihood(tree,y,curve,0.7,0.01;model=:EnvLin)
    @test !invalid.success
    @test invalid.loglik == -Inf
    @test !t_env_loglikelihood(tree,y,curve,-1.,0.).success
    calls = Ref(0)
    environment(t) = (calls[] += 1; t/3)
    cache = EvoTraits._env_precalc(tree,environment)
    expected_calls = calls[]
    calls[] = 0
    fit = fit_t_env(tree,y,environment;max_iterations=80)
    @test calls[] == expected_calls
    @test fit.success
    @test fit.f_calls <= 80
    replay = t_env_loglikelihood(tree,y,environment,fit.sigma2,fit.beta)
    @test fit.loglik ≈ replay.loglik atol=1e-12
    budget_limited = fit_t_env(tree,y,environment;max_iterations=1)
    @test budget_limited.success
    @test !budget_limited.converged
    @test budget_limited.f_calls == 1
    @test aic(fit) == fit.aic
    @test aicc(fit) == Inf
    for model in (:EnvExp,:EnvLin)
        fitted = fit_t_env(tree,y,EnvCurve([0.,3.],[0.,1.]);model,max_iterations=500)
        @test fitted.success
        replay = t_env_loglikelihood(tree,y,fitted.environment,fitted.sigma2,fitted.beta;model)
        @test fitted.loglik ≈ replay.loglik atol=1e-12
    end
    @test_throws ArgumentError fit_t_env(tree,y,curve;max_iterations=0)
    @test_throws ArgumentError fit_t_env(tree,y,curve;guess_sigma2=-1.)
    @test_throws ArgumentError fit_t_env(tree,fill(NaN,4),curve)
    @test_throws ArgumentError t_env_loglikelihood(tree,y,curve,0.7,0.;quadrature_breakpoints=[-1.])
    @test_throws ArgumentError t_env_loglikelihood(tree,y,curve,0.7,0.;quadrature_breakpoints=[4.])
    @test_throws ArgumentError t_env_loglikelihood(tree,y,curve,0.7,0.;quadrature_breakpoints=[NaN])
    divided = t_env_loglikelihood(tree,y,curve,0.7,0.2;
        quadrature_breakpoints=[1.5,0.5,1.5])
    undivided = t_env_loglikelihood(tree,y,curve,0.7,0.2)
    @test divided.loglik ≈ undivided.loglik atol=1e-12
    @test divided.quadrature_breakpoints == [0.5,1.5]
    splitfit = fit_t_env(tree,y,t -> t/3;quadrature_breakpoints=[0.5,1.5],max_iterations=80)
    @test splitfit.success
    replay = t_env_loglikelihood(tree,y,splitfit.environment,splitfit.sigma2,splitfit.beta;
        quadrature_breakpoints=splitfit.quadrature_breakpoints)
    @test splitfit.loglik ≈ replay.loglik atol=1e-12
    pulse(t) = exp(-0.5*((t-1.37)/0.005)^2)
    pulse_breaks = collect(range(1.32,1.42;length=41))
    pulsed = t_env_loglikelihood(tree,y,pulse,0.7,1.4;model=:EnvLin,
        quadrature_breakpoints=pulse_breaks)
    normal = EvoTraits.Normal()
    expected = Float64[]
    for edge in 1:tree.nedges
        lo = 3-tree.dist_from_root[tree.child_of_edge[edge]]
        hi = lo+tree.edge_length[edge]
        integral = 0.005sqrt(2pi)*(EvoTraits.cdf(normal,(hi-1.37)/0.005)-
            EvoTraits.cdf(normal,(lo-1.37)/0.005))
        push!(expected,0.7*(hi-lo)+0.7integral)
    end
    @test pulsed.edge_variances ≈ expected atol=1e-12
    @test pulsed.success
end
