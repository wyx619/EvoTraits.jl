@testset "Environmental BM terminal errors" begin
    tree = EvoTraits._parse_load_tree_text("((a:1,b:1):2,(c:2,d:2):1);")
    y = [0.4,-0.7,1.2,0.1]
    environment = EnvCurve([0.,3.],[0.,1.])
    known = [0.,0.2,0.1,0.3]
    nuisance = 0.4
    paths = zeros(tree.ntips,tree.nedges)
    for (i,tip) in enumerate(tree.tip_ids)
        node = tip
        while node != tree.root
            edge = findfirst(==(node),tree.child_of_edge)
            paths[i,edge] = 1
            node = tree.parent_of_node[node]
        end
    end
    for model in (:EnvExp,:EnvLin)
        branch_variances = map(1:tree.nedges) do edge
            young = 3-tree.dist_from_root[tree.child_of_edge[edge]]
            old = young+tree.edge_length[edge]
            model === :EnvExp ? 0.7/(0.2/3)*(exp(0.2old/3)-exp(0.2young/3)) :
                0.7*(old-young)+(0.2-0.7)/6*(old^2-young^2)
        end
        for values in (y,[NaN,-0.7,1.2,0.1]), errors in (zeros(4),known), nu in (0.,nuisance)
            result = t_env_LL(tree,values,environment,0.7,0.2;
                model,error=errors,nuisance=nu)
            covariance = paths*Diagonal(branch_variances)*paths'+Diagonal(errors.^2 .+ nu^2)
            selected = findall(!isnan,values)
            factor = cholesky(Symmetric(covariance[selected,selected]))
            one = ones(length(selected))
            root = dot(one,factor\values[selected])/dot(one,factor\one)
            residual = values[selected].-root
            ll = -0.5*(dot(residual,factor\residual)+logdet(factor)+length(selected)*log(2pi))
            @test result.success
            @test result.loglik ≈ ll atol=1e-11
            @test result.root_state ≈ root atol=1e-12
            @test result.nparams == 4
            @test result.aic ≈ 8-2ll
            @test result.nobs == length(selected)
            @test result.nuisance == nu
            @test result.nuisance_variance == nu^2
            @test result.standard_errors == errors
            for edge in 1:tree.nedges
                tip = findfirst(==(tree.child_of_edge[edge]),tree.tip_ids)
                expected = branch_variances[edge]+(tip === nothing ? 0. : errors[tip]^2+nu^2)
                @test result.edge_variances[edge] ≈ expected atol=1e-12
            end
        end
        plain = t_env_LL(tree,y,environment,0.7,0.2;model)
        boundary = t_env_LL(tree,y,environment,0.7,0.2;model,error=missing)
        @test plain.edge_variances == boundary.edge_variances
        @test plain.loglik == boundary.loglik
        @test plain.nparams == 3
        @test boundary.aic == plain.aic+2
        @test isempty(plain.standard_errors)
        @test plain.nuisance == plain.nuisance_variance == 0.
        @test t_env_LL(tree,y,environment,0.7,0.2;model,error=NaN).loglik == boundary.loglik
        implicit = t_env_LL(tree,y,environment,0.7,0.2;model,nuisance)
        explicit = t_env_LL(tree,y,environment,0.7,0.2;model,error=missing,nuisance)
        @test implicit.edge_variances == explicit.edge_variances
        named = Dict(Symbol(tree.tip_labels[i])=>known[i] for i in 4:-1:1)
        from_named = t_env_LL(tree,y,environment,0.7,0.2;model,error=named,nuisance)
        from_vector = t_env_LL(tree,y,environment,0.7,0.2;model,error=known,nuisance)
        @test from_named.edge_variances == from_vector.edge_variances
        @test from_named.loglik == from_vector.loglik
        fitted = fit_t_env(tree,y,environment;model,error=known,max_iterations=1200,rel_tol=1e-8)
        replay = t_env_LL(tree,y,environment,fitted.sigma2,fitted.beta;
            model,error=known,nuisance=fitted.nuisance)
        @test fitted.success
        @test fitted.nparams == 4
        @test fitted.loglik ≈ replay.loglik atol=1e-12
        @test fitted.edge_variances == replay.edge_variances
        @test fitted.nuisance >= 0
        @test fitted.nuisance_variance == fitted.nuisance^2
        @test occursin("nuisance=",sprint(show,fitted))
    end
    fit = fit_t_env(tree,y,environment;error=missing,max_iterations=1)
    @test fit.success
    @test !fit.converged
    @test fit.f_calls == 1
    @test fit.nparams == 4
    @test fit.nuisance ≈ 0.1
    @test isinf(fit.aicc)
    @test fit.edge_variances == fit_t_env(tree,y,environment;error=NaN,max_iterations=1).edge_variances
    frame = DataFrame(taxon=reverse(tree.tip_labels),height=reverse(y),se=reverse(known))
    from_column = fit_t_env(tree,frame,environment;trait_col=:height,error=:se,max_iterations=1)
    from_vector = fit_t_env(tree,y,environment;error=known,max_iterations=1)
    @test from_column.loglik == from_vector.loglik
    @test from_column.standard_errors == known
    @test from_column.trait_name == "height"
    @test from_column.edge_variances == from_vector.edge_variances
    @test_throws ArgumentError fit_t_env(tree,frame,environment;trait_col=:height,error=:absent)
    larger = EvoTraits._parse_load_tree_text("(((a:1,b:1):1,(c:1,d:1):1):1,((e:1,f:1):1,(g:1,h:1):1):1);")
    for values in (collect(1.:8.),[NaN,2.,3.,4.,5.,6.,7.,8.])
        result = t_env_LL(larger,values,environment,0.7,0.2;error=missing,nuisance)
        @test result.aicc ≈ result.aic+40/(count(!isnan,values)-5)
    end
    for invalid in (-0.1,Inf,[0.1],[-0.1,0.,0.,0.],[Inf,0.,0.,0.],
        [NaN,0.,0.,0.],[missing,0.,0.,0.],fill(floatmax(Float64),4),Dict("a"=>0.1))
        @test_throws ArgumentError fit_t_env(tree,y,environment;error=invalid)
        @test_throws ArgumentError t_env_LL(tree,y,environment,0.7,0.2;error=invalid)
    end
    duplicate = Dict{Any,Float64}("a"=>0.1,:a=>0.2,"b"=>0.1,"c"=>0.1,"d"=>0.1)
    @test_throws ArgumentError fit_t_env(tree,y,environment;error=duplicate)
    for invalid in (-0.1,Inf,NaN,floatmax(Float64))
        @test_throws ArgumentError t_env_LL(tree,y,environment,0.7,0.2;nuisance=invalid)
    end
end
