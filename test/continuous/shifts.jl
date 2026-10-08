include("shift_constraints.jl")

@testset "Time-shift input, fixed likelihood and fit diagnostics" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    trait = [0.2,-0.5,1.4,0.9]
    cache = EvoTraits._shift_precalc(tree;age=2.0)
    for model in (:OUBM,:OUBMi,:BMOU,:BMOUi)
        independent = model in (:OUBMi,:BMOUi)
        extra = independent ? (sig=1.2,) : NamedTuple()
        profiled = EvoTraits._shift_oubm_loglikelihood(tree,trait,0.5,0.7;model=model,precalc=cache,extra...)
        at_mean = EvoTraits._shift_oubm_loglikelihood(tree,trait,0.5,0.7;
            model=model,age=2.0,theta=profiled.theta,extra...)
        away = EvoTraits._shift_oubm_loglikelihood(tree,trait,0.5,0.7;
            model=model,mapped_edge=cache.mapping.mapped_edge,theta=profiled.theta+0.4,extra...)
        @test profiled.success
        @test at_mean.loglik ≈ profiled.loglik atol=1e-10
        @test away.loglik < profiled.loglik
        @test profiled.nparams == (independent ? 4 : 3)
        fitextra = independent ? (guess_sig=1.2,) : NamedTuple()
        exhausted = EvoTraits._fit_shift_oubm(tree,trait;model=model,precalc=cache,
            guess_alpha=0.5,guess_sigma2=0.7,max_iterations=1,fitextra...)
        @test exhausted.success
        @test !exhausted.converged
        @test exhausted.termination == "MAXEVAL_REACHED"
        @test exhausted.loglik >= profiled.loglik-1e-10
        @test exhausted.alpha > 0
    end
    @test_throws ArgumentError EvoTraits._shift_precalc(tree;age=0.0)
    @test_throws ArgumentError EvoTraits._shift_mapping(tree,2;precalc=cache)
    @test_throws ArgumentError EvoTraits._shift_mapping(tree,1;precalc=cache,age=2.0)
    changed = deepcopy(tree)
    changed.edge_length[1] += 0.1
    @test_throws ArgumentError EvoTraits._shift_mapping(changed,1;precalc=cache)
    @test_throws ArgumentError EvoTraits._fit_shift_oubm(tree,trait;age=2.0,max_iterations=0)
    @test_throws ArgumentError EvoTraits._fit_shift_oubm(tree,trait;age=2.0,guess_alpha=-1.0)
    @test_throws ArgumentError EvoTraits._fit_shift_oubm(tree,trait;age=2.0,guess_sig=1.0)
    @test_throws ArgumentError EvoTraits._fit_shift_oubm(tree,trait;age=2.0,A_decomp=:schur)
end

@testset "General OU shift A with positive real eigenvalues" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    data = [0.2 0.1;-0.5 0.6;1.4 -0.3;0.9 1.2]
    sigma = [0.8 0.15;0.15 0.6]
    A = [-1.0 2.0;-3.0 4.0]
    factor = EvoTraits._shift_ou_factor(A,2,:decomposable_positive)
    @test factor.reliable
    @test !isposdef(Symmetric((A+A')/2))
    pars = EvoTraits._shift_pack_A(A,:decomposable_positive)
    @test length(pars) == 4
    @test EvoTraits._shift_unpack_A(pars,2,:decomposable_positive) ≈ A atol=1e-10
    P,Q = EvoTraits._shift_ou_increment(factor,sigma,1.2)
    E = exp([-A sigma;zeros(2,2) A']*1.2)
    @test P ≈ E[1:2,1:2] atol=1e-10
    @test Q ≈ E[1:2,3:4]*E[1:2,1:2]' atol=1e-10
    for model in (:OUBMi,:BMOUi)
        evaluated = EvoTraits._mvshift_oubm_loglikelihood(tree,data,A,sigma;
            age=2.0,model=model,sig=sigma,A_decomp=:decomposable_positive)
        @test evaluated.success
        @test evaluated.nparams == 12
        fitted = EvoTraits._fit_mvshift_oubm(tree,data;age=2.0,model=model,
            A_decomp=:decomposable_positive,guess_A=A,guess_sigma=sigma,guess_sig=sigma,max_iterations=1)
        @test fitted.success
        @test !fitted.converged
        @test fitted.loglik ≈ evaluated.loglik atol=1e-10
    end
    @test_throws ArgumentError EvoTraits._shift_ou_factor([0.5 -1;1 0.5],2,:decomposable_positive)
    @test_throws ArgumentError EvoTraits._shift_ou_factor([-0.5 0;0 0.9],2,:decomposable_positive)
    @test_throws ArgumentError EvoTraits._shift_unpack_A([1.0],2,:decomposable_positive)
    initialized = EvoTraits._fit_shift_oubm(tree,data[:,1];age=2.0,
        A_decomp=:decomposable_positive,max_iterations=1)
    @test initialized.success
    @test !initialized.converged
    rotation = [cos(0.3) -sin(0.3);sin(0.3) cos(0.3)]
    unreliable = rotation*[6e14 0;0 0.7]*rotation'
    failed = EvoTraits._mvshift_oubm_loglikelihood(tree,data,unreliable,sigma;
        age=2.0,A_decomp=:decomposable_positive)
    @test !failed.success
    @test failed.loglik == -Inf
end

@testset "Triangular OU shift kernel" begin
    tree = EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    data = [0.2 0.1;-0.5 0.6;1.4 -0.3;0.9 1.2]
    sigma = [0.8 0.15;0.15 0.6]
    for decomp in (:upper,:lower)
        A = decomp === :upper ? [0.4 2;0 0.4] : [0.4 0;2 0.4]
        factor = EvoTraits._shift_ou_factor(A,2,decomp)
        P,Q = EvoTraits._shift_ou_increment(factor,sigma,2.0)
        @test P ≈ exp(-A*2) atol=1e-10
        identity = Matrix{Float64}(I,2,2)
        stationary = reshape((kron(identity,A)+kron(A,identity))\vec(sigma),2,2)
        @test Q ≈ stationary-P*stationary*P' atol=1e-10
        pars = EvoTraits._shift_pack_A(A,decomp)
        @test EvoTraits._shift_unpack_A(pars,2,decomp) ≈ A atol=1e-12
        for model in (:OUBMi,:BMOUi)
            evaluated = EvoTraits._mvshift_oubm_loglikelihood(tree,data,A,sigma;
                age=2.0,model=model,sig=sigma,A_decomp=decomp)
            @test evaluated.success
            @test evaluated.nparams == 11
            fitted = EvoTraits._fit_mvshift_oubm(tree,data;age=2.0,model=model,
                A_decomp=decomp,guess_A=A,guess_sigma=sigma,guess_sig=sigma,max_iterations=1)
            @test fitted.success
            @test !fitted.converged
            @test fitted.loglik ≈ evaluated.loglik atol=1e-10
        end
    end
    A = [6.122516728883845e14 1.1658837890624998;0 0.7915449232888906]
    factor = EvoTraits._shift_ou_factor(A,2,:upper)
    P,Q = EvoTraits._shift_ou_increment(factor,sigma,2.0)
    @test P[2,2] ≈ exp(-A[2,2]*2) atol=1e-14
    unreliable = EvoTraits._mvshift_oubm_loglikelihood(tree,data,A,sigma;
        age=2.0,model=:BMOUi,sig=sigma,A_decomp=:upper)
    @test !unreliable.success
    @test unreliable.loglik == -Inf
    @test_throws ArgumentError EvoTraits._shift_ou_factor(A,2,:lower)
    @test_throws ArgumentError EvoTraits._shift_ou_factor(A,2,:cholesky)
end
include("shift_api.jl")
include("shift_scaling.jl")
include("shift_diagnostics.jl")
include("shift_near_singular.jl")
