# Shared time mapping; process-specific implementations reuse existing engines.
function _shift_time_map(tree::CompactTree, age::Real)
    _validate_binary_tree(tree)
    times = tree.dist_from_root[tree.tip_ids]
    all(isfinite, tree.edge_length) && all(>=(0), tree.edge_length) ||
        throw(ArgumentError("Shift models require finite nonnegative branch lengths"))
    height = maximum(times)
    isfinite(height) && height > 0 || throw(ArgumentError("Shift models require positive tree height"))
    tolerance = 1e-8 * height
    maximum(times) - minimum(times) <= tolerance ||
        throw(ArgumentError("Shift models require an ultrametric tree"))
    age_value = Float64(age)
    isfinite(age_value) && 0 < age_value < height ||
        throw(ArgumentError("Shift age must be finite and strictly between zero and tree height ($height)"))
    boundary = height - age_value
    mapped = zeros(Float64, tree.nedges, 2)
    for edge in 1:tree.nedges
        start = tree.dist_from_root[tree.parent_of_edge[edge]]
        length = tree.edge_length[edge]
        early = clamp(boundary - start, 0.0, length)
        mapped[edge, 1] = early
        mapped[edge, 2] = length - early
    end
    all(sum(@view mapped[:, r]) > 0 for r in 1:2) ||
        throw(ArgumentError("Both shift periods must contain positive branch duration"))
    return (age = age_value, boundary = boundary, height = height, mapped_edge = mapped)
end

function _shift_time_map(tree::CompactTree, mapped_edge::AbstractMatrix{<:Real})
    size(mapped_edge) == (tree.nedges, 2) ||
        throw(ArgumentError("Shift mapping must have nedges rows and two columns: early, late"))
    mapped = _validate_mapped_edge(tree, mapped_edge)
    early_time = zeros(Float64, tree.nnodes)
    for node in tree.preorder
        tree.is_tip[node] && continue
        for edge in tree.first_child_edge[node]:tree.last_child_edge[node]
            early_time[tree.child_of_edge[edge]] = early_time[node] + mapped[edge, 1]
        end
    end
    boundary = early_time[first(tree.tip_ids)]
    height = maximum(@view tree.dist_from_root[tree.tip_ids])
    mapping = _shift_time_map(tree, height - boundary)
    tolerance = 1e-8 * height
    for edge in 1:tree.nedges, period in 1:2
        abs(mapped[edge, period] - mapping.mapped_edge[edge, period]) <= tolerance ||
            throw(ArgumentError("Shift mapping must represent one common time boundary, with early before late"))
    end
    return mapping
end

struct ShiftPrecalc{T}
    tree_snapshot::CompactTree
    mapping::T
    ntraits::Int
end

"""
    _shift_precalc(tree; age=nothing, mapped_edge=nothing, ntraits=1)

Cache a parameter-independent, two-period time mapping for internal shift
likelihoods. Supply exactly one of `age` or `mapped_edge` (columns early, late).
Reuse checks the tree snapshot and trait dimension; process kernels are not cached.
"""
function _shift_precalc(tree::CompactTree; age = nothing, mapped_edge = nothing, ntraits::Integer = 1)
    ntraits > 0 || throw(ArgumentError("ntraits must be positive"))
    (age === nothing) != (mapped_edge === nothing) ||
        throw(ArgumentError("Specify exactly one of age or mapped_edge"))
    mapping = _shift_time_map(tree, age === nothing ? mapped_edge : age)
    return ShiftPrecalc(deepcopy(tree), mapping, Int(ntraits))
end

function _shift_mapping(tree::CompactTree, ntraits::Integer; age = nothing, mapped_edge = nothing, precalc = nothing)
    if precalc === nothing
        (age === nothing) != (mapped_edge === nothing) ||
            throw(ArgumentError("Specify exactly one of age or mapped_edge"))
        return _shift_time_map(tree, age === nothing ? mapped_edge : age)
    end
    precalc isa ShiftPrecalc || throw(ArgumentError("precalc must be a ShiftPrecalc"))
    age === nothing && mapped_edge === nothing ||
        throw(ArgumentError("precalc cannot be combined with age or mapped_edge"))
    precalc.ntraits == ntraits || throw(ArgumentError("precalc trait dimension does not match data"))
    snapshot = precalc.tree_snapshot
    # CompactTree contains mutable arrays; identity alone cannot detect stale input.
    for name in fieldnames(CompactTree)
        getfield(tree, name) == getfield(snapshot, name) ||
            throw(ArgumentError("precalc tree no longer matches input ($name)"))
    end
    mapping = precalc.mapping
    size(mapping.mapped_edge) == (tree.nedges, 2) || throw(ArgumentError("precalc mapping dimensions changed"))
    isfinite(mapping.age) && 0 < mapping.age < mapping.height &&
        mapping.boundary == mapping.height - mapping.age ||
        throw(ArgumentError("precalc time metadata is invalid"))
    tolerance = 1e-8 * mapping.height
    for edge in 1:tree.nedges
        early = clamp(mapping.boundary - tree.dist_from_root[tree.parent_of_edge[edge]], 0.0, tree.edge_length[edge])
        abs(mapping.mapped_edge[edge, 1] - early) <= tolerance &&
            abs(mapping.mapped_edge[edge, 2] - (tree.edge_length[edge] - early)) <= tolerance ||
            throw(ArgumentError("precalc time mapping changed"))
    end
    return mapping
end

function _shift_sr_result(fit, mapping, nobs::Integer)
    return (
        model = :SR,
        success = fit.success,
        converged = fit.converged,
        loglik = fit.loglik,
        aic = fit.aic,
        aicc = compute_aicc(fit.loglik, fit.nparams, nobs),
        nparams = fit.nparams,
        nobs = Int(nobs),
        age = mapping.age,
        boundary = mapping.boundary,
        height = mapping.height,
        mapped_edge = copy(mapping.mapped_edge),
        fit = fit,
    )
end

include("univariate/shifts/model.jl")
include("multivariate/shifts/edges.jl")
include("multivariate/shifts/parameters.jl")
include("multivariate/shifts/likelihood.jl")
include("multivariate/shifts/model.jl")
include("multivariate/shifts/fit.jl")
include("multivariate/shifts/api.jl")
include("multivariate/shifts/diagnostics.jl")
