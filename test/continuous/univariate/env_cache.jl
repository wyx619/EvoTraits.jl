# Independent per-edge implementation of the pre-sharing quadrature layout.
function reference_env_variances(tree,env,sigma2,beta,model;order=16,breaks=Float64[])
    coefficients = [i/sqrt(4.0*i*i-1) for i in 1:order-1]
    F = eigen(SymTridiagonal(zeros(order),coefficients))
    weights = 2 .* F.vectors[1,:].^2
    height = maximum(tree.dist_from_root[tree.tip_ids])
    result = zeros(tree.nedges)
    for edge in 1:tree.nedges
        age = height-tree.dist_from_root[tree.child_of_edge[edge]]
        young,old = max(age,0.),max(age+tree.edge_length[edge],max(age,0.))
        cuts = sort!(unique!(vcat(young,
            filter(t -> young<t<old,EvoTraits._env_knots(env)),
            filter(t -> young<t<old,breaks),old)))
        for j in 1:length(cuts)-1
            half = (cuts[j+1]-cuts[j])/2
            center = cuts[j]+half
            for k in eachindex(weights)
                value = env(center+half*F.values[k])
                rate = model===:EnvExp ? sigma2*exp(beta*value) : sigma2+(beta-sigma2)*value
                result[edge] += half*weights[k]*rate
            end
        end
    end
    return result
end

@testset "Shared environmental intervals and analytical EnvLin" begin
    tree = EvoTraits._parse_load_tree_text("((a:1,b:1):2,(c:2,d:2):1);")
    zero_tree = EvoTraits._parse_load_tree_text("((a:1,b:1):2,(c:3,d:3):0);")
    x = collect(range(0.,3.;length=81))
    curve = EnvCurve(x,0.5 .+ 0.2sin.(3x))
    spline = smooth_env(x,0.5 .+ 0.2sin.(3x);df=8.)
    callable(t) = 0.5+0.2sin(3t)
    breaks = [0.,0.5,1.,1.5,1.5,2.,2.9,3.]
    for tr in (tree,zero_tree), env in (curve,spline,callable), model in (:EnvExp,:EnvLin)
        cache = EvoTraits._env_precalc(tr,env;model,quadrature_breakpoints=breaks)
        variances = zeros(tr.nedges)
        for beta in (model===:EnvExp ? (-3.,0.,4.) : (1e-15,0.7,2.))
            @test EvoTraits._env_edge_variances!(variances,cache,0.7,beta,model)
            reference = reference_env_variances(tr,env,0.7,beta,model;breaks)
            @test variances ≈ reference atol=1e-12 rtol=1e-12
        end
        if model===:EnvLin && env !== callable
            @test cache.analytic_linear
            @test isempty(cache.values) && isempty(cache.weights) && isempty(cache.interval_ids)
            @test_throws ArgumentError EvoTraits._env_edge_variances!(variances,cache,0.7,0.1,:EnvExp)
        else
            @test !cache.analytic_linear
            @test length(cache.values)==cache.order*length(cache.interval_variances)
            @test length(cache.interval_variances)<length(cache.interval_ids)
        end
    end
    invalid = EnvCurve([0.,1.,3.],[0.,1.5,0.])
    result = t_env_loglikelihood(tree,[0.4,-0.7,1.2,0.1],invalid,0.7,1e-15;model=:EnvLin)
    @test !result.success
    constant = EnvCurve([0.,1.,3.],ones(3))
    cache = EvoTraits._env_precalc(tree,constant;model=:EnvLin)
    variances = zeros(tree.nedges)
    @test EvoTraits._env_edge_variances!(variances,cache,0.7,1e-20,:EnvLin)
    @test variances ≈ 1e-20 .* tree.edge_length rtol=1e-14
end
