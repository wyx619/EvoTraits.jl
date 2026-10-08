@testset "Shift automatic height scaling" begin
    tree=EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    snapshot=deepcopy(tree)
    normalized=deepcopy(tree)
    normalized.edge_length ./=4
    normalized.dist_from_root ./=4
    Y=[0.2 0.1;-0.5 0.6;1.4 -0.3;0.9 1.2]
    Y[2,1]=NaN
    S=[0.8 0.1;0.1 0.8]
    A=[0.4 0.1;0 0.8]
    for model in (:SR,:OUBM,:OUBMi,:BMOU,:BMOUi,:EBOU,:EBOUi,:OUEB,:OUEBi,:EBBM,:EBBMi,:BMEB,:BMEBi,:ER,:RR,:EC,:RC)
        spec=model===:SR ? nothing : EvoTraits._shift_process_spec(model)
        ou=spec!==nothing && spec.ou_period>0
        eb=spec!==nothing && spec.eb_period>0
        independent=spec!==nothing && spec.independent
        sigma=model===:SR ? [S,1.3S] : S
        extra=merge(ou ? (A=A,A_decomp=:upper) : NamedTuple(),eb ? (beta=-0.3,) : NamedTuple(),
            independent ? (sig=1.3S,) : NamedTuple())
        fitextra=merge(ou ? (guess_A=A,A_decomp=:upper) : NamedTuple(),eb ? (guess_beta=-0.3,) : NamedTuple(),
            independent ? (guess_sig=1.3S,) : NamedTuple())
        expected=mvshift_loglikelihood(normalized,Y,sigma;model=model,age=0.5,extra...)
        cache=shift_precalc(tree;age=2.,ntraits=2)
        saved=copy(cache.mapping.mapped_edge)
        for input in ((age=2.,),(mapped_edge=saved,),(precalc=cache,))
            result=mvshift_loglikelihood(tree,Y,sigma;model=model,scale_height=true,input...,extra...)
            @test result.loglik ≈ expected.loglik atol=1e-10
            @test result.theta ≈ expected.theta atol=1e-10
            @test result.height ≈ 1.0
            @test result.age ≈ 0.5
            @test result.source_height==4.0
            @test result.time_scale==0.25
            @test result.scale_height
            @test result.mapped_edge ≈ saved/4
        end
        fitted=fit_mvshift(tree,Y;model=model,scale_height=true,precalc=cache,
            guess_sigma=sigma,max_iterations=1,optimization=:LN_SBPLX,fitextra...)
        manual=fit_mvshift(normalized,Y;model=model,age=0.5,
            guess_sigma=sigma,max_iterations=1,optimization=:LN_SBPLX,fitextra...)
        @test fitted.loglik ≈ manual.loglik atol=1e-10
        @test fitted.sigma ≈ manual.sigma atol=1e-10
        @test cache.mapping.mapped_edge==saved
        # Convert rates to original time units for probability-model equivalence.
        rateextra=merge(ou ? (A=A/4,A_decomp=:upper) : NamedTuple(),
            eb ? (beta=-0.3/4,) : NamedTuple(),independent ? (sig=1.3S/4,) : NamedTuple())
        original=mvshift_loglikelihood(tree,Y,model===:SR ? [S/4,1.3S/4] : S/4;
            model=model,age=2.,rateextra...)
        @test expected.loglik ≈ original.loglik atol=1e-10
        @test expected.aic ≈ original.aic atol=1e-10
    end
    for name in fieldnames(CompactTree)
        @test getfield(tree,name)==getfield(snapshot,name)
    end
    scalar=fit_shift(tree,Y[:,1];model=:EBBM,age=2.,scale_height=true,max_iterations=10)
    manual=fit_shift(normalized,Y[:,1];model=:EBBM,age=0.5,max_iterations=10)
    @test scalar.loglik ≈ manual.loglik atol=1e-10
    @test scalar.height ≈ 1.
    @test_throws ArgumentError fit_shift(tree,Y[:,1];age=2.,mapped_edge=zeros(tree.nedges,2),scale_height=true)
    @test_throws ArgumentError fit_mvshift(tree,Y;precalc=shift_precalc(tree;age=2.),scale_height=true)
end
@testset "Common OU attraction rate" begin
    tree=EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    for p in (1,2,3)
        Y=reshape(collect(range(-0.6,1.3;length=4p)),4,p)
        A=Matrix{Float64}(I,p,p)*0.4
        S=Matrix{Float64}(I,p,p)*0.8
        pars=EvoTraits._shift_pack_A(A,:equaldiagonal)
        @test length(pars)==1
        @test EvoTraits._shift_unpack_A(pars,p,:equaldiagonal) ≈ A atol=1e-12
        for model in (:OUBM,:OUBMi,:BMOU,:BMOUi,:EBOU,:EBOUi,:OUEB,:OUEBi)
            spec=EvoTraits._shift_process_spec(model)
            extra=merge(spec.independent ? (sig=1.3S,) : NamedTuple(),
                spec.eb_period>0 ? (beta=-0.3,) : NamedTuple())
            full=mvshift_loglikelihood(tree,Y,S;model=model,age=2.,A=A,A_decomp=:diagonal,extra...)
            constrained=mvshift_loglikelihood(tree,Y,S;model=model,age=2.,A=A,A_decomp=:equaldiagonal,extra...)
            @test constrained.loglik ≈ full.loglik atol=1e-10
            @test constrained.nparams==full.nparams-p+1
            fitted=fit_mvshift(tree,Y;model=model,age=2.,A_decomp=:equaldiagonal,
                max_iterations=5,(spec.independent ? (guess_sig=1.3S,) : NamedTuple())...)
            @test fitted.success
            @test isdiag(fitted.A)
            @test all(==(fitted.A[1,1]),diag(fitted.A))
        end
    end
    Y=zeros(4,2)
    @test_throws ArgumentError mvshift_loglikelihood(tree,Y,Matrix{Float64}(I,2,2);
        model=:OUBM,age=2.,A=[0.4 0;0 0.8],A_decomp=:equaldiagonal)
    @test_throws ArgumentError mvshift_loglikelihood(tree,Y,Matrix{Float64}(I,2,2);
        model=:OUBM,age=2.,A=[0.4 0.1;0.1 0.4],A_decomp=:equaldiagonal)
end
