@testset "Shift local curvature diagnostics" begin
    quadratic=x->sum(abs2,x)
    d=EvoTraits._shift_curvature(quadratic,[0.2,-0.3],1e-3)
    @test d.hessian ≈ 2Matrix{Float64}(I,2,2) atol=1e-8
    @test d.gradient ≈ [0.4,-0.6] atol=1e-8
    coupled=EvoTraits._shift_curvature(x->x[1]^2+3x[1]*x[2]+4x[2]^2,[0.,0.],1e-3)
    @test coupled.hessian ≈ [2. 3.;3. 8.] atol=1e-8
    flat=EvoTraits._shift_curvature(x->x[1]^2,[0.,0.],1e-3)
    @test eigvals(Symmetric(flat.hessian)) ≈ [0.,2.] atol=1e-8
    saddle=EvoTraits._shift_curvature(x->x[1]^2-x[2]^2,[0.,0.],1e-3)
    @test eigvals(Symmetric(saddle.hessian)) ≈ [-2.,2.] atol=1e-8
    tree=EvoTraits._parse_load_tree_text("((a:2,b:2):2,(c:1,d:1):3);")
    y=[0.2,-0.5,1.4,0.9]
    for model in (:SR,:OUBM,:OUBMi,:BMOU,:BMOUi,:EBOU,:EBOUi,:OUEB,:OUEBi,:EBBM,:EBBMi,:BMEB,:BMEBi)
        fit=fit_shift(tree,y;model=model,age=2.,max_iterations=10,optimization=:LN_SBPLX)
        diagnostic=diagnose_shift(tree,y,fit)
        @test diagnostic.status in (:positive_curvature,:nonstationary,:weak_curvature,
            :negative_curvature,:unstable_differences,:invalid_neighborhood,:beta_plateau)
        @test diagnostic.f_calls>0
        @test fit.success
        if hasproperty(diagnostic,:hessian)
            @test issymmetric(diagnostic.hessian)
            @test size(diagnostic.hessian,1)==fit.nparams-1
        end
    end
    plateau=shift_loglikelihood(tree,y,0.8;model=:EBBM,age=2.,beta=3.)
    @test diagnose_shift(tree,y,plateau).status===:beta_plateau
    boundary=shift_loglikelihood(tree,y,0.8;model=:EBBM,age=2.,beta=2.)
    @test diagnose_shift(tree,y,boundary).status===:beta_boundary
    scaled=fit_shift(tree,y;model=:EBBM,age=2.,scale_height=true,max_iterations=10)
    @test diagnose_shift(tree,y,scaled).f_calls>0
    @test_throws ArgumentError diagnose_shift(tree,y .+ [0.,0.,0.,1.],scaled)
    @test_throws ArgumentError diagnose_shift(tree,y,scaled;step=0)
    @test_throws ArgumentError diagnose_shift(tree,y,scaled;tolerance=NaN)
end
