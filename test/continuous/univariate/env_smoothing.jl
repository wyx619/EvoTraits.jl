@testset "Fixed-df environment smoothing" begin
    x = collect(range(0.,3.;length=20))
    y = sin.(x)+0.1cos.(20x)
    curve = smooth_env(x,y;df=6.)
    @test curve.effective_df ≈ 6 atol=1e-7
    @test curve.(x) ≈ curve.smoothed atol=1e-12
    @test_throws ArgumentError curve(-1.)
    @test_throws ArgumentError smooth_env(x,y;df=2.)
    @test_throws ArgumentError smooth_env(x,y;df=20.)
    @test_throws ArgumentError smooth_env(x,y;df=6.,weights=zeros(20))
    @test_throws ArgumentError smooth_env([0.,1.,2.],[1.,2.,3.];df=2.5)
    linear = smooth_env(x,2x .+ 1;df=6.)
    @test linear.smoothed ≈ 2x .+ 1 atol=1e-10
    duplicate = smooth_env(vcat(x,x[3]),vcat(y,y[3]+0.2);df=6.)
    @test duplicate.weights[3] == 2
    @test duplicate.values[3] ≈ y[3]+0.1
    scaled = smooth_env(x,y;df=6.,scale=true)
    @test minimum(scaled.smoothed) == 0
    @test maximum(scaled.smoothed) == 1
    # Independent dense linear-system check is confined to a small test fixture.
    n = length(x)
    h = diff(x)
    Q = zeros(n,n-2)
    for i in 1:n-2
        Q[i,i]=1/h[i];Q[i+1,i]=-1/h[i]-1/h[i+1];Q[i+2,i]=1/h[i+1]
    end
    R = Matrix(SymTridiagonal((h[1:end-1]+h[2:end])/3,h[2:end-1]/6))
    B = R+curve.penalty*Q'*Q
    S = Matrix{Float64}(I,n,n)-curve.penalty*Q*(B\Q')
    @test tr(S) ≈ curve.effective_df atol=1e-9
    @test S*y ≈ curve.smoothed atol=1e-10
    tree = EvoTraits._parse_load_tree_text("((a:1,b:1):2,(c:2,d:2):1);")
    values = [0.4,-0.7,1.2,0.1]
    fixed = t_env_loglikelihood(tree,values,scaled,0.7,0.2)
    @test fixed.success
    @test fixed.loglik ≈ t_env_loglikelihood(tree,values,scaled,0.7,0.2;quadrature_order=32).loglik atol=1e-10
    fit = fit_t_env(tree,values,scaled;max_iterations=80)
    @test fit.success
    @test fit.loglik ≈ t_env_loglikelihood(tree,values,scaled,fit.sigma2,fit.beta).loglik atol=1e-12
    lo,hi = EvoTraits._env_smooth_range(scaled,3.)
    @test lo <= minimum(scaled.smoothed)
    @test hi >= maximum(scaled.smoothed)
    # Dense irregular knots exercise the high-precision preprocessing path.
    dense_x = collect(range(0.,1.;length=1001)).^3
    dense_y = sin.(3dense_x)
    dense = smooth_env(dense_x,dense_y;df=6.)
    converted = smooth_env(100dense_x,dense_y;df=6.)
    @test dense.effective_df ≈ 6 atol=1e-7
    @test dense.smoothed ≈ converted.smoothed atol=1e-9
    @test converted.penalty ≈ 100^3*dense.penalty rtol=1e-8
end
