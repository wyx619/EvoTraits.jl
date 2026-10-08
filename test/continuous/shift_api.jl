@testset "Public shift API and result contract" begin
    tree=EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    y=[0.2,-0.5,1.4,0.9]
    Y=reshape(y,:,1)
    models=(:SR,:OUBM,:OUBMi,:BMOU,:BMOUi,:EBOU,:EBOUi,:OUEB,:OUEBi,:EBBM,:EBBMi,:BMEB,:BMEBi,:ER,:RR,:EC,:RC)
    for name in models
        spec=name===:SR ? nothing : EvoTraits._shift_process_spec(name)
        ou=spec!==nothing && spec.ou_period>0
        eb=spec!==nothing && spec.eb_period>0
        independent=spec!==nothing && spec.independent
        S=name===:SR ? [fill(0.8,1,1),fill(1.04,1,1)] : fill(0.8,1,1)
        fixedextra=merge(ou ? (A=fill(0.4,1,1),) : NamedTuple(),
            eb ? (beta=-0.3,) : NamedTuple(), independent ? (sig=fill(1.04,1,1),) : NamedTuple())
        fitextra=merge(ou ? (guess_A=fill(0.4,1,1),) : NamedTuple(),
            eb ? (guess_beta=-0.3,) : NamedTuple(),independent ? (guess_sig=fill(1.04,1,1),) : NamedTuple())
        fixed=mvshift_loglikelihood(tree,Y,S;model=name,age=2.,fixedextra...)
        fitted=fit_mvshift(tree,Y;model=name,age=2.,guess_sigma=S,
            optimization=:LN_SBPLX,max_iterations=1,fitextra...)
        @test fixed isa ShiftFitResult
        @test fitted isa ShiftFitResult
        @test fixed.success && fitted.success
        @test !fixed.converged && !fitted.converged
        @test fixed.termination===:FIXED
        @test fitted.optimization===:LN_SBPLX
        replayS=name===:SR ? [fitted.sigma,fitted.sig] : fitted.sigma
        replayextra=merge(ou ? (A=fitted.A,) : NamedTuple(),
            eb ? (beta=fitted.input_beta,) : NamedTuple(),
            independent ? (sig=fitted.sig,) : NamedTuple())
        replay=mvshift_loglikelihood(tree,Y,replayS;model=name,age=2.,replayextra...)
        @test replay.loglik ≈ fitted.loglik atol=1e-10
        @test size(fitted.sigma)==(1,1)
        @test length(fitted.theta)==1
        @test fitted.input_model===name
        @test aic(fitted)==fitted.aic
        @test aicc(fitted)==fitted.aicc
        @test loglik(fitted)==fitted.loglik
        @test nparams(fitted)==fixed.nparams
        @test length(aic_table(:fixed=>fixed,:fit=>fitted))==2
        cache=shift_precalc(tree;age=2.)
        reused=mvshift_loglikelihood(tree,Y,S;model=name,precalc=cache,fixedextra...)
        @test reused.loglik ≈ fixed.loglik atol=1e-10
        uniextra=merge(ou ? (alpha=0.4,) : NamedTuple(), eb ? (beta=-0.3,) : NamedTuple(),
            independent ? (sig=1.04,) : NamedTuple())
        unifitextra=merge(ou ? (guess_alpha=0.4,) : NamedTuple(),eb ? (guess_beta=-0.3,) : NamedTuple(),
            independent ? (guess_sig=1.04,) : NamedTuple())
        scalarS=name===:SR ? [0.8,1.04] : 0.8
        uni=shift_loglikelihood(tree,y,scalarS;model=name,age=2.,uniextra...)
        unifit=fit_shift(tree,y;model=name,age=2.,guess_sigma2=scalarS,
            optimization=:LN_SBPLX,max_iterations=1,unifitextra...)
        @test uni.loglik ≈ fixed.loglik atol=1e-10
        @test unifit.loglik ≈ fitted.loglik atol=1e-10
        @test uni.theta ≈ fixed.theta
        @test occursin("time-shift",sprint(show,fitted))
    end
    missingY=copy(Y);missingY[2,1]=NaN
    @test fit_mvshift(tree,missingY;model=:EBBM,age=2.,max_iterations=10).nobs==3
    @test_throws ArgumentError fit_shift(tree,y;model=:unknown,age=2.)
    @test_throws ArgumentError fit_shift(tree,y;model=:SR,age=2.,guess_alpha=0.4)
    @test_throws ArgumentError mvshift_loglikelihood(tree,Y,fill(0.8,1,1);model=:EBBM,age=2.)
    @test_throws ArgumentError fit_mvshift(tree,Y;model=:EBBM,age=2.,max_iterations=0)
    @test_throws MethodError fit_mvshift(tree,Y;model=:EBBM,age=2.,error=0.1)
end
