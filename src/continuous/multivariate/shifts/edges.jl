function _shift_oubm_spec(model::Symbol)
    canonical = model === :ER ? :OUBM : model === :RR ? :OUBMi :
        model === :EC ? :BMOU : model === :RC ? :BMOUi : model
    canonical in (:OUBM, :OUBMi, :BMOU, :BMOUi) ||
        throw(ArgumentError("Unsupported OU/BM shift model: $model"))
    return (model = canonical, ou_period = canonical in (:OUBM, :OUBMi) ? 1 : 2,
        independent = canonical in (:OUBMi, :BMOUi))
end

function _shift_process_spec(model::Symbol)
    if model in (:OUBM, :OUBMi, :BMOU, :BMOUi, :ER, :RR, :EC, :RC)
        spec = _shift_oubm_spec(model)
        return merge(spec, (eb_period = 0, primary_period = spec.ou_period))
    end
    model in (:EBOU, :EBOUi, :OUEB, :OUEBi, :EBBM, :EBBMi, :BMEB, :BMEBi) ||
        throw(ArgumentError("Unsupported shift process model: $model"))
    early_eb = model in (:EBOU, :EBOUi, :EBBM, :EBBMi)
    has_ou = model in (:EBOU, :EBOUi, :OUEB, :OUEBi)
    eb_period = early_eb ? 1 : 2
    other_period = 3 - eb_period
    return (model = model, ou_period = has_ou ? other_period : 0,
        independent = endswith(String(model), "i"), eb_period = eb_period, primary_period = other_period)
end

function _shift_eb_duration(start::Float64, duration::Float64, beta::Float64)
    duration == 0 && return 0.0
    beta == 0 && return duration
    scaled = beta * duration
    integral = scaled == 0 ? duration : duration * (expm1(scaled) / scaled)
    return exp(beta * start) * integral
end

struct ShiftAsymmetricFactor
    A::Matrix{Float64}
end

struct ShiftSchurFactor
    basis::Matrix{Float64}
    triangular::ShiftAsymmetricFactor
    reliable::Bool
end

_shift_A_nparams(p, decomp) = decomp === :decomposable_positive ? p * p :
    decomp === :equaldiagonal ? 1 : decomp === :diagonal ? p : p * (p + 1) ÷ 2

function _shift_ou_factor(A::AbstractMatrix{<:Real}, p::Integer, A_decomp::Symbol)
    size(A) == (p, p) || throw(ArgumentError("A must match trait dimension"))
    all(isfinite, A) || throw(ArgumentError("A must be finite"))
    A_decomp in (:cholesky, :diagonal, :equaldiagonal, :upper, :lower, :decomposable_positive) ||
        throw(ArgumentError("Unsupported shift A_decomp=$A_decomp"))
    if A_decomp === :decomposable_positive
        decomposition = schur(Matrix{Float64}(A))
        T = Matrix{Float64}(decomposition.T)
        istriu(T) || throw(ArgumentError("decomposable_positive requires real eigenvalues"))
        all(>(0), diag(T)) || throw(ArgumentError("decomposable_positive requires positive eigenvalues"))
        # Schur backward error scales with norm(A); preserve relative resolution of slow rates.
        reliable = minimum(diag(T)) / maximum(abs, A) > sqrt(eps(Float64))
        return ShiftSchurFactor(Matrix{Float64}(decomposition.Z), ShiftAsymmetricFactor(T), reliable)
    end
    if A_decomp in (:upper, :lower)
        (A_decomp === :upper ? istriu(A) : istril(A)) ||
            throw(ArgumentError("A_decomp=$A_decomp requires the corresponding triangular matrix"))
        all(>(0), diag(A)) || throw(ArgumentError("Triangular OU A requires a positive diagonal"))
        return ShiftAsymmetricFactor(Matrix{Float64}(A))
    end
    issymmetric(A) || throw(ArgumentError("A_decomp=$A_decomp requires a symmetric A"))
    A_decomp in (:diagonal,:equaldiagonal) && !isdiag(A) &&
        throw(ArgumentError("A_decomp=$A_decomp requires zero off-diagonal entries"))
    A_decomp === :equaldiagonal && !all(==(A[1,1]),diag(A)) &&
        throw(ArgumentError("A_decomp=:equaldiagonal requires equal diagonal rates"))
    factor = _mvou_transition_eigen(A)
    all(>(0), factor.values) || throw(ArgumentError("OU A must be positive definite"))
    return factor
end

function _shift_diffusion(S::AbstractMatrix{<:Real}, p::Integer)
    size(S) == (p, p) || throw(ArgumentError("Diffusion matrix must match trait dimension"))
    all(isfinite, S) && issymmetric(S) ||
        throw(ArgumentError("Diffusion matrix must be finite and symmetric"))
    matrix = Matrix{Float64}(S)
    isposdef(Symmetric(matrix)) || throw(ArgumentError("Diffusion matrix must be positive definite"))
    return matrix
end

function _shift_ou_increment(factor, scatter::Matrix{Float64}, duration::Float64)
    Phi = _mvou_transition_from_eigen(factor, duration)
    p = length(factor.values)
    integrated = similar(scatter)
    for j in 1:p, i in 1:p
        rate = factor.values[i] + factor.values[j]
        integrated[i, j] = scatter[i, j] * (-expm1(-rate * duration) / rate)
    end
    Q = factor.vectors * integrated * factor.vectors'
    return Phi, (Q + Q') / 2
end

function _shift_correct_triangular_transition!(Phi, A, duration)
    p = size(A, 1)
    for i in 1:p
        Phi[i, i] = exp(-A[i, i] * duration)
    end
    upper = istriu(A)
    for k in 1:p-1
        i, j = upper ? (k, k+1) : (k+1, k)
        a, b = A[i, i], A[j, j]
        delta = abs(a - b)
        integral = delta == 0 ? duration * exp(-a * duration) :
            exp(-min(a, b) * duration) * (-expm1(-delta * duration)) / delta
        Phi[i, j] = -A[i, j] * integral
    end
    return Phi
end

function _shift_ou_increment(factor::ShiftAsymmetricFactor, sigma::Matrix{Float64}, duration::Float64)
    A = factor.A
    p = size(A, 1)
    duration == 0 && return Matrix{Float64}(I, p, p), zeros(p, p)
    # Keep the positive exponential block small, then compose exact transitions.
    # This remains valid for Jordan blocks and repeated eigenvalues, without A eigenvectors.
    scale = maximum(abs, A)
    logsize = log2(scale) + log2(Float64(p)) + log2(duration)
    doublings = max(0, ceil(Int, logsize))
    dt = ldexp(duration, -doublings)
    dt > 0 || throw(OverflowError("OU time subdivision underflow"))
    E = exp([-A sigma; zeros(p, p) A'] * dt)
    Phi = E[1:p, 1:p]
    _shift_correct_triangular_transition!(Phi, A, dt)
    Q = E[1:p, p+1:2p] * Phi'
    Q = (Q + Q') / 2
    for iteration in 1:doublings
        Q = Q + Phi * Q * Phi'
        Q = (Q + Q') / 2
        Phi = Phi * Phi
        _shift_correct_triangular_transition!(Phi, A, ldexp(dt, iteration))
    end
    return Phi, Q
end

function _shift_ou_increment(factor::ShiftSchurFactor, sigma::Matrix{Float64}, duration::Float64)
    factor.reliable || throw(PosDefException(0))
    basis = factor.basis
    transformed = basis' * sigma * basis
    transformed = (transformed + transformed') / 2
    Phi, Q = _shift_ou_increment(factor.triangular, transformed, duration)
    transition = basis * Phi * basis'
    covariance = basis * Q * basis'
    return transition, (covariance + covariance') / 2
end

function _shift_oubm_edges(tree::CompactTree, mapping, spec, factor, sigma, sig; beta = 0.0)
    p = size(sigma, 1)
    d = 3p
    transitions = Vector{Matrix{Float64}}(undef, tree.nedges)
    covariances = Vector{Matrix{Float64}}(undef, tree.nedges)
    noise_factors = Vector{Matrix{Float64}}(undef, tree.nedges)
    scatter = factor === nothing || factor isa Union{ShiftAsymmetricFactor, ShiftSchurFactor} ?
        sigma : factor.vectors' * sigma * factor.vectors
    eb_period = hasproperty(spec, :eb_period) ? spec.eb_period : 0
    primary_period = hasproperty(spec, :primary_period) ? spec.primary_period : spec.ou_period
    for edge in 1:tree.nedges
        F = Matrix{Float64}(I, d, d)
        Q = zeros(d, d)
        active = Int[]
        for period in 1:2
            duration = mapping.mapped_edge[edge, period]
            duration == 0 && continue
            block = (period - 1) * p + 1:period * p
            if period == spec.ou_period
                F[block, block], Q[block, block] = _shift_ou_increment(factor, scatter, duration)
            elseif period == eb_period
                parent_time = tree.dist_from_root[tree.parent_of_edge[edge]]
                start = period == 1 ? min(parent_time, mapping.boundary) : max(parent_time - mapping.boundary, 0.0)
                amount = _shift_eb_duration(start, duration, beta)
                isfinite(amount) && amount > 0 || throw(OverflowError("EB variance increment is outside representable range"))
                Q[block, block] = sig * amount
            else
                Q[block, block] = (period == primary_period ? sigma : sig) * duration
            end
            append!(active, block)
        end
        # Only active blocks have noise; the frozen and constant-mean blocks are exactly zero.
        all(isfinite, F) && all(isfinite, Q) || throw(OverflowError("Shift edge operators are not finite"))
        L = zeros(d, length(active))
        if !isempty(active)
            L[active, :] = cholesky(Symmetric(Q[active, active])).L
        end
        transitions[edge], covariances[edge], noise_factors[edge] = F, Q, L
    end
    return (Phi = transitions, Q = covariances, L = noise_factors)
end
