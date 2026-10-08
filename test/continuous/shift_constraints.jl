@testset "Shift constrained diffusion" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    data = [0.2 0.1;-0.5 0.6;1.4 -0.3;0.9 1.2]
    A = [0.4 0.1;0 0.8]
    matrices = ((:diagonal,[0.8 0;0 0.6],2),(:equaldiagonal,[0.8 0;0 0.8],1),
        (:equal,[0.8 0.15;0.15 0.8],2))
    for (decomp,S,k) in matrices
        pars = EvoTraits._shift_pack_sigma(S,decomp)
        @test length(pars)==k
        @test EvoTraits._shift_unpack_sigma(pars,2,decomp) ≈ S atol=1e-12
        full = EvoTraits._mvshift_oubm_loglikelihood(tree,data,A,S;age=2.0,A_decomp=:upper)
        constrained = EvoTraits._mvshift_oubm_loglikelihood(tree,data,A,S;
            age=2.0,A_decomp=:upper,Sigma_decomp=decomp)
        @test constrained.loglik ≈ full.loglik atol=1e-10
        @test constrained.nparams==2+3+k
        fitted = EvoTraits._fit_mvshift_oubm(tree,data;age=2.0,A_decomp=:upper,
            Sigma_decomp=decomp,guess_A=A,guess_sigma=S,max_iterations=1)
        @test fitted.success
        @test !fitted.converged
        @test fitted.loglik ≈ constrained.loglik atol=1e-10
        sr = EvoTraits._mvshift_sr_loglikelihood(tree,data,[S,1.3S];age=2.0,Sigma_decomp=decomp)
        @test sr.nparams==2+2k
        fitted_sr = EvoTraits._fit_mvshift_sr(tree,data;age=2.0,Sigma_decomp=decomp,
            guess_sigma=[S,1.3S],optimization=:LN_SBPLX,max_iterations=1)
        @test fitted_sr.success
        @test !fitted_sr.converged
        @test fitted_sr.loglik ≈ sr.loglik atol=1e-10
    end
    @test_throws ArgumentError EvoTraits._shift_sigma([0.8 0.1;0.1 0.8],2,:diagonal)
    @test_throws ArgumentError EvoTraits._shift_sigma([0.8 0;0 0.6],2,:equaldiagonal)
    @test_throws ArgumentError EvoTraits._shift_sigma([0.8 0.1;0.1 0.6],2,:equal)
    @test_throws ArgumentError EvoTraits._shift_sigma_nparams(2,:unknown)
end

@testset "EB time shifts and zero-beta limits" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    data = [0.2 0.1;-0.5 0.6;1.4 -0.3;0.9 1.2]
    A,S = [0.4 0.1;0 0.8],[0.8 0.15;0.15 0.6]
    for model in (:EBOU,:EBOUi,:OUEB,:OUEBi,:EBBM,:EBBMi,:BMEB,:BMEBi)
        spec = EvoTraits._shift_process_spec(model)
        extra = spec.independent ? (sig=1.3S,) : NamedTuple()
        ouextra = spec.ou_period>0 ? (A=A,A_decomp=:upper) : NamedTuple()
        fit = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,model=model,beta=-0.3,extra...,ouextra...)
        @test fit.success
        @test !fit.converged
        fixed = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,model=model,beta=-0.3,
            theta=fit.theta,extra...,ouextra...)
        @test fixed.loglik ≈ fit.loglik atol=1e-10
        zero = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,model=model,beta=0.0,extra...,ouextra...)
        limiting = if spec.ou_period>0
            bm_model = spec.ou_period==1 ? :OUBM : :BMOU
            spec.independent && (bm_model=Symbol(string(bm_model,"i")))
            EvoTraits._mvshift_oubm_loglikelihood(tree,data,A,S;age=2.0,model=bm_model,A_decomp=:upper,extra...)
        else
            early = spec.primary_period==1 ? S : (spec.independent ? 1.3S : S)
            late = spec.primary_period==2 ? S : (spec.independent ? 1.3S : S)
            EvoTraits._mvshift_sr_loglikelihood(tree,data,[early,late];age=2.0)
        end
        @test zero.loglik ≈ limiting.loglik atol=1e-10
        tiny = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,model=model,beta=1e-320,extra...,ouextra...)
        @test tiny.loglik ≈ zero.loglik atol=1e-10
    end
    @test_throws ArgumentError EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,beta=NaN)
    @test_throws ArgumentError EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,beta=-0.3,model=:EBOU)
    failed = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,beta=1000.0,beta_bounds=nothing)
    @test !failed.success
    @test failed.loglik==-Inf
    reset = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,beta=3.0)
    zero = EvoTraits._mvshift_eb_loglikelihood(tree,data,S;age=2.0,beta=0.0)
    @test reset.beta==0.0
    @test reset.input_beta==3.0
    @test reset.loglik ≈ zero.loglik atol=1e-10
end
@testset "EB shift fitting" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    Y = [0.2 0.1; -0.5 0.6; 1.4 -0.3; 0.9 1.2]
    S = [0.8 0.15; 0.15 0.8]
    for model in (:EBOU,:EBOUi,:OUEB,:OUEBi,:EBBM,:EBBMi,:BMEB,:BMEBi)
        spec = EvoTraits._shift_process_spec(model)
        initial = spec.independent ? (guess_sig=1.3S,) : NamedTuple()
        ou = spec.ou_period > 0 ? (guess_A=[0.4 0.1;0 0.8],A_decomp=:upper) : NamedTuple()
        fit = EvoTraits._fit_mvshift_eb(tree,Y;model=model,age=2.0,
            guess_sigma=S,guess_beta=-0.3,Sigma_decomp=:equal,max_iterations=1,initial...,ou...)
        @test fit.success
        @test !fit.converged
        @test fit.termination=="MAXEVAL_REACHED"
        @test fit.loglik ≈ fit.initial_loglik atol=1e-10
        @test fit.beta==-0.3
        fixedextra = spec.independent ? (sig=fit.sig,) : NamedTuple()
        fixed = EvoTraits._mvshift_eb_loglikelihood(tree,Y,fit.sigma;model=model,age=2.0,
            A=fit.A,A_decomp=:upper,Sigma_decomp=:equal,beta=fit.input_beta,fixedextra...)
        @test fit.loglik ≈ fixed.loglik atol=1e-10
        @test fit.nparams==2+(spec.ou_period>0 ? 3 : 0)+2*(spec.independent ? 2 : 1)+1
        uni = EvoTraits._fit_shift_eb(tree,Y[:,1];model=model,age=2.0,
            guess_sigma2=0.8,guess_beta=0.0,max_iterations=1,
            (spec.independent ? (guess_sig=1.04,) : NamedTuple())...,
            (spec.ou_period>0 ? (guess_alpha=0.4,) : NamedTuple())...)
        @test uni.success
        @test uni.beta==0.0
        @test (uni.alpha===nothing)==(spec.ou_period==0)
    end
    @test_throws ArgumentError EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,model=:OUBM)
    @test_throws ArgumentError EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,guess_beta=NaN)
    @test_throws ArgumentError EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,beta_bounds=(2.,-2.))
    @test_throws ArgumentError EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,guess_A=S)
    @test_throws ArgumentError EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,guess_sig=S)
    @test_throws ArgumentError EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,max_iterations=0)
    reset = EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,guess_beta=3.,max_iterations=1)
    @test reset.beta==0.0
    @test reset.input_beta==3.0
    unlimited = EvoTraits._fit_mvshift_eb(tree,Y;age=2.0,guess_beta=3.,beta_bounds=nothing,max_iterations=1)
    @test unlimited.beta==3.0
    @test unlimited.success
end
