@inline function _time_summary_check_input(node_tbl::AbstractDataFrame, n::Real, w::Real)
    n > 0 || throw(ArgumentError("n must be positive"))
    0 <= w < n || throw(ArgumentError("w must satisfy 0 <= w < n"))
    :time_before_present in propertynames(node_tbl) || throw(ArgumentError("node table must contain time_before_present"))
    return nothing
end

function _time_summary_trait_columns(node_tbl::AbstractDataFrame)
    cols = Symbol.(names(node_tbl))
    if :estimate in cols
        return [("trait", :estimate)]
    end
    pairs = Tuple{String, Symbol}[]
    for col in cols
        s = String(col)
        startswith(s, "estimate_") || continue
        suffix = s[10:end]
        isempty(suffix) && continue
        push!(pairs, (suffix, col))
    end
    isempty(pairs) && throw(ArgumentError("node table does not contain estimate columns"))
    return pairs
end

@inline function _time_summary_window_starts(max_age::Real, n::Real, w::Real)
    step = Float64(w)
    width = Float64(n)
    if step == 0.0
        nbins = max(1, ceil(Int, Float64(max_age) / width))
        return collect(0.0:width:((nbins - 1) * width))
    end
    last_start = max(0.0, Float64(max_age) - width)
    nwin = floor(Int, last_start / step) + 1
    return [step * i for i in 0:(nwin - 1)]
end

function _time_summary_stats(values::AbstractVector{<:Real})
    x = Float64[]
    sizehint!(x, length(values))
    @inbounds for v in values
        fv = Float64(v)
        isfinite(fv) && push!(x, fv)
    end
    isempty(x) && return (n_nodes = 0, mean = NaN, median = NaN, q2_5 = NaN, q97_5 = NaN)
    sort!(x)
    return (
        n_nodes = length(x),
        mean = Statistics.mean(x),
        median = Statistics.median(x),
        q2_5 = Statistics.quantile(x, 0.025),
        q97_5 = Statistics.quantile(x, 0.975),
    )
end

function _time_summary_one_trait(
    times::AbstractVector{<:Real},
    estimates::AbstractVector{<:Real},
    n::Float64,
    w::Float64,
)
    max_age = maximum(times)
    starts = _time_summary_window_starts(max_age, n, w)
    nbins = length(starts)

    start_vals = Vector{Float64}(undef, nbins + 1)
    end_vals = Vector{Float64}(undef, nbins + 1)
    n_nodes = Vector{Int}(undef, nbins + 1)
    mean_vals = Vector{Float64}(undef, nbins + 1)
    median_vals = Vector{Float64}(undef, nbins + 1)
    q2_5_vals = Vector{Float64}(undef, nbins + 1)
    q97_5_vals = Vector{Float64}(undef, nbins + 1)

    global_stats = _time_summary_stats(estimates)
    start_vals[1] = 0.0
    end_vals[1] = 0.0
    n_nodes[1] = global_stats.n_nodes
    mean_vals[1] = global_stats.mean
    median_vals[1] = global_stats.median
    q2_5_vals[1] = global_stats.q2_5
    q97_5_vals[1] = global_stats.q97_5

    for i in 1:nbins
        lo = starts[i]
        hi = lo + n

        vals = Float64[]
        @inbounds for j in eachindex(times)
            t = Float64(times[j])
            isfinite(t) || continue
            if i == nbins
                (lo <= t <= hi) || continue
            else
                (lo <= t < hi) || continue
            end
            push!(vals, Float64(estimates[j]))
        end

        stats = _time_summary_stats(vals)
        idx = i + 1
        start_vals[idx] = lo
        end_vals[idx] = hi
        n_nodes[idx] = stats.n_nodes
        mean_vals[idx] = stats.mean
        median_vals[idx] = stats.median
        q2_5_vals[idx] = stats.q2_5
        q97_5_vals[idx] = stats.q97_5
    end

    return DataFrame(
        :start => start_vals,
        :end => end_vals,
        :n_nodes => n_nodes,
        :mean => mean_vals,
        :median => median_vals,
        :q2_5 => q2_5_vals,
        :q97_5 => q97_5_vals,
    )
end

function _time_summary_regime(
    times::AbstractVector{<:Real},
    regime_ids,
    regime_labels,
    n::Float64,
    w::Float64,
)
    max_age = maximum(times)
    starts = _time_summary_window_starts(max_age, n, w)
    rows_start = Float64[]; rows_end = Float64[]; rows_regime_id = Int[]; rows_regime = String[]; rows_n = Int[]; rows_prop = Float64[]

    valid_ids = Int[]
    valid_labels = String[]
    seen = Set{Int}()
    for i in eachindex(regime_ids)
        rid = Int(regime_ids[i])
        rid > 0 || continue
        rid in seen && continue
        push!(seen, rid)
        push!(valid_ids, rid)
        push!(valid_labels, String(regime_labels[i]))
    end
    perm = sortperm(valid_ids)
    valid_ids = valid_ids[perm]
    valid_labels = valid_labels[perm]
    regime_to_slot = Dict{Int, Int}(valid_ids[i] => i for i in eachindex(valid_ids))

    for i in eachindex(valid_ids)
        rid = valid_ids[i]
        label = valid_labels[i]
        count = 0
        denom = 0
        @inbounds for j in eachindex(times)
            t = Float64(times[j])
            isfinite(t) || continue
            denom += 1
            Int(regime_ids[j]) == rid && (count += 1)
        end
        push!(rows_start, 0.0)
        push!(rows_end, 0.0)
        push!(rows_regime_id, rid)
        push!(rows_regime, label)
        push!(rows_n, count)
        push!(rows_prop, denom == 0 ? NaN : count / denom)
    end

    for win in eachindex(starts)
        lo = starts[win]
        hi = lo + n
        denom = 0
        counts = zeros(Int, length(valid_ids))
        @inbounds for j in eachindex(times)
            t = Float64(times[j])
            isfinite(t) || continue
            in_window = win == length(starts) ? (lo <= t <= hi) : (lo <= t < hi)
            in_window || continue
            denom += 1
            rid = Int(regime_ids[j])
            idx = get(regime_to_slot, rid, 0)
            idx == 0 || (counts[idx] += 1)
        end
        for i in eachindex(valid_ids)
            push!(rows_start, lo)
            push!(rows_end, hi)
            push!(rows_regime_id, valid_ids[i])
            push!(rows_regime, valid_labels[i])
            push!(rows_n, counts[i])
            push!(rows_prop, denom == 0 ? NaN : counts[i] / denom)
        end
    end

    return DataFrame(
        :start => rows_start,
        :end => rows_end,
        :regime_id => rows_regime_id,
        :regime => rows_regime,
        :n_nodes => rows_n,
        :proportion => rows_prop,
    )
end

"""
    summarize_node_estimates_by_time(node_tbl, n; w = 0.0)

Summarize `estim_node_table(...)` output by time bins on `time_before_present`.
`n` is the window width. `w` is the sliding step size. When `w = 0`, windows
default to non-overlapping bins of width `n`: `[0,n), [n,2n), ...`. When
`0 < w < n`, windows slide by `w`: `[0,n), [w,w+n), [2w,2w+n), ...`. The
first row is an extra baseline row with `start = end = 0`, summarizing all
node estimates regardless of time. Returns one summary `DataFrame` per trait.
"""
function summarize_node_estimates_by_time(
    node_tbl::AbstractDataFrame,
    n::Real;
    w::Real = 0.0,
)
    _time_summary_check_input(node_tbl, n, w)
    trait_cols = _time_summary_trait_columns(node_tbl)
    times = Float64.(node_tbl[!, :time_before_present])
    out = Dict{String, DataFrame}()
    for (trait, col) in trait_cols
        out[trait] = _time_summary_one_trait(times, node_tbl[!, col], Float64(n), Float64(w))
    end
    cols = Symbol.(names(node_tbl))
    if :regime_id in cols && :regime in cols
        out["_regime_summary"] = _time_summary_regime(
            times,
            node_tbl[!, :regime_id],
            node_tbl[!, :regime],
            Float64(n),
            Float64(w),
        )
    end
    return out
end
