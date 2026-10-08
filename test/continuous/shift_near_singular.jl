@testset "BM covariance workspace reuse" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    data = [0.2 0.1; -0.5 0.6; 1.4 -0.3; 0.9 1.2]
    context = EvoTraits._mvbm_covariance_workspace(tree, data)
    workspace = EvoTraits._mv_profile_workspace(tree, 2)
    mapped = reshape(copy(tree.edge_length), :, 1)
    saved = nothing
    saved_theta = nothing
    for pass in 1:6
        values = copy(data) .+ pass/10
        pass in (2,4) && (values[1,1] = NaN)
        pass == 3 && (values[2,:] .= NaN)
        sigmas = [[1.0+pass/10 0.1; 0.1 0.8]]
        reference = EvoTraits._mvbmm_profile_validated(tree,values,sigmas,mapped)
        reused = EvoTraits._mvbmm_profile_validated(tree,values,sigmas,mapped;
            workspace,covariance_workspace=context)
        @test reused.success
        @test reused.loglik ≈ reference.loglik atol=1e-12
        @test reused.theta ≈ reference.theta atol=1e-12
        if pass == 1
            saved = reused
            saved_theta = copy(reused.theta)
        else
            @test saved.theta == saved_theta
        end
    end
end

@testset "SR covariance recursion near singular diffusion" begin
    tree=EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    Y=[0.2 0.1;-0.5 0.6;1.4 -0.3;0.9 1.2]
    rotation=[cos(0.7) -sin(0.7);sin(0.7) cos(0.7)]
    for small in (1e-6,1e-10,1e-14)
        S=rotation*Diagonal([small,0.8])*rotation'
        S=(S+S')/2
        S2=[1.2 0.1;0.1 1.]
        fixed=mvshift_loglikelihood(tree,Y,[S,S2];model=:SR,age=2.)
        @test fixed.success
        # Independent tip covariance: early shared times and late shared times.
        early=[2. 2. 0. 0.;2. 2. 0. 0.;0. 0. 2. 2.;0. 0. 2. 2.]
        late=[2. 0. 0. 0.;0. 2. 0. 0.;0. 0. 2. 1.;0. 0. 1. 2.]
        V=kron(S,early)+kron(S2,late)
        W=kron(Matrix{Float64}(I,2,2),ones(4,1))
        F=cholesky(Symmetric(V))
        theta=(W'*(F\W))\(W'*(F\vec(Y)))
        delta=vec(Y)-W*theta
        ll=-0.5*(dot(delta,F\delta)+logdet(F)+8log(2pi))
        @test fixed.loglik ≈ ll atol=1e-10
        @test fixed.theta ≈ theta atol=1e-10
        roundtrip=EvoTraits._shift_unpack_sigma(EvoTraits._shift_pack_sigma(S,:cholesky),2,:cholesky)
        replay=mvshift_loglikelihood(tree,Y,[roundtrip,S2];model=:SR,age=2.)
        @test replay.loglik ≈ ll atol=1e-10
        supplied=[0.1,-0.2]
        fixedtheta=mvshift_loglikelihood(tree,Y,[S,S2];model=:SR,age=2.,theta=supplied)
        residual=vec(Y)-W*supplied
        @test fixedtheta.loglik ≈ -0.5*(dot(residual,F\residual)+logdet(F)+8log(2pi)) atol=1e-10
        for pattern in ([(1,1)],[(1,1),(2,2)],[(1,1),(1,2)],
            [(1,1),(2,1),(3,2),(4,2)],[(1,1),(2,2),(3,1)])
            missingY=copy(Y)
            for (i,j) in pattern
                missingY[i,j]=NaN
            end
            uncertain=mvshift_loglikelihood(tree,missingY,[S,S2];model=:SR,age=2.)
            @test uncertain.success
            observed=findall(!isnan,vec(missingY))
            Wobs=W[observed,:]
            Fobs=cholesky(Symmetric(V[observed,observed]))
            yobs=vec(missingY)[observed]
            mean=(Wobs'*(Fobs\Wobs))\(Wobs'*(Fobs\yobs))
            residual=yobs-Wobs*mean
            expected=-0.5*(dot(residual,Fobs\residual)+logdet(Fobs)+length(observed)*log(2pi))
            @test uncertain.loglik ≈ expected atol=1e-10
            @test uncertain.theta ≈ mean atol=1e-10
            suppliedfit=mvshift_loglikelihood(tree,missingY,[S,S2];model=:SR,age=2.,theta=supplied)
            residual=yobs-Wobs*supplied
            @test suppliedfit.loglik ≈ -0.5*(dot(residual,Fobs\residual)+logdet(Fobs)+length(observed)*log(2pi)) atol=1e-10
        end
    end
    zero=EvoTraits._parse_load_tree_text("(((a:1,b:1):0,c:1):1,(d:1,e:1):1);")
    Yzero=[0.2 NaN;NaN 0.6;1.4 -0.3;NaN NaN;0.9 1.2]
    zerofit=mvshift_loglikelihood(zero,Yzero,[[0.8 0.1;0.1 0.6],[1.2 0.1;0.1 1.]];model=:SR,age=0.5)
    @test zerofit.success
    @test all(isfinite,zerofit.theta)
    # Exactly deterministic contradictory observations do not define a nonsingular density.
    singular=EvoTraits._parse_load_tree_text("((a:0,b:0):2,(c:1,d:1):1);")
    singularY=[0.2 NaN;0.4 NaN;1.4 -0.3;0.9 1.2]
    invalid=mvshift_loglikelihood(singular,singularY,[[0.8 0.1;0.1 0.6],[1.2 0.1;0.1 1.]];model=:SR,age=0.5)
    @test !invalid.success
    @test invalid.loglik==-Inf
end
