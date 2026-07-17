_continuous_default_regime_names(k::Integer) = ["regime$(i)" for i in 1:Int(k)]

function _continuous_checked_trait_name(trait_name)
    trait_name === nothing && return "trait"
    return String(trait_name)
end

function _continuous_checked_regime_names(regime_names, nregimes::Integer)
    n = Int(nregimes)
    if regime_names === nothing
        return _continuous_default_regime_names(n)
    end
    out = String.(collect(regime_names))
    length(out) == n || throw(ArgumentError("regime_names length must be $n"))
    return out
end

function _normalize_trait_columns(df::AbstractDataFrame, trait_cols)
    if trait_cols === nothing
        return Symbol.(names(df))
    elseif trait_cols isa Symbol || trait_cols isa AbstractString
        return [Symbol(trait_cols)]
    else
        return Symbol.(trait_cols)
    end
end

function _check_dataframe_columns(df::AbstractDataFrame, cols::AbstractVector{Symbol})
    for col in cols
        hasproperty(df, col) || throw(ArgumentError("data does not contain column `$col`"))
    end
    return nothing
end

function _is_missing_string(x::AbstractString)
    s = lowercase(strip(String(x)))
    return isempty(s) || s in ("na", "nan", "missing", "null")
end

function _continuous_missing_string(x)
    x isa AbstractString || return false
    s = lowercase(strip(String(x)))
    return isempty(s) || s in ("na", "nan", "missing", "null")
end

function _aligned_trait_value(val, allow_missing::Bool)
    if ismissing(val)
        allow_missing || throw(ArgumentError("single-trait alignment does not allow missing values"))
        return NaN
    elseif val isa AbstractString
        _is_missing_string(val) && begin
            allow_missing || throw(ArgumentError("single-trait alignment does not allow missing values"))
            return NaN
        end
        parsed = tryparse(Float64, strip(String(val)))
        parsed === nothing && throw(ArgumentError("trait data contains a non-numeric value: `$val`"))
        return parsed
    else
        return Float64(val)
    end
end

function _aligned_dataframe_rows(tree::CompactTree, df::AbstractDataFrame, taxon_col)
    if taxon_col === nothing
        DataFrames.nrow(df) == tree.ntips ||
            throw(ArgumentError("data must have $(tree.ntips) rows when taxon_col is not supplied; rows are assumed to match tree.tip_labels order"))
        return collect(1:tree.ntips)
    end

    taxon = Symbol(taxon_col)
    _check_dataframe_columns(df, [taxon])
    row_by_tip = Dict{String, Int}()
    for row in 1:DataFrames.nrow(df)
        label = string(df[row, taxon])
        haskey(row_by_tip, label) && throw(ArgumentError("duplicate taxon label `$label` in data"))
        row_by_tip[label] = row
    end

    rows = Vector{Int}(undef, tree.ntips)
    missing_tips = String[]
    @inbounds for (i, label) in enumerate(tree.tip_labels)
        row = get(row_by_tip, label, 0)
        if row == 0
            push!(missing_tips, label)
        else
            rows[i] = row
        end
    end
    if !isempty(missing_tips)
        example = join(first(missing_tips, min(length(missing_tips), 5)), ", ")
        throw(ArgumentError("data is missing $(length(missing_tips)) tree tips, e.g. $example"))
    end
    return rows
end

"""
    align_traits_to_tree(tree, data; taxon_col=nothing, trait_cols=nothing)

Return continuous trait values reordered to match `tree.tip_labels`.

For `DataFrame` input, pass `taxon_col` when rows are not already in tree tip
order. Extra rows not present in the tree are ignored. Missing tree tips and
duplicate taxon labels are rejected. For a single trait, missing values are
rejected. For multiple traits, missing values are converted to `NaN`.

If one trait column is selected, the result is a `Vector{Float64}`; otherwise it
is a `Matrix{Float64}` with rows in `tree.tip_labels` order.
"""
function align_traits_to_tree(
    tree::CompactTree,
    data::AbstractDataFrame;
    taxon_col::Union{Nothing, Symbol, AbstractString} = nothing,
    trait_cols = nothing,
)
    df = DataFrames.DataFrame(data)
    cols = _normalize_trait_columns(df, trait_cols)
    if taxon_col !== nothing
        cols = [col for col in cols if col != Symbol(taxon_col)]
    end
    isempty(cols) && throw(ArgumentError("trait_cols selects no trait columns"))
    _check_dataframe_columns(df, cols)

    rows = _aligned_dataframe_rows(tree, df, taxon_col)
    values = Matrix(df[rows, cols])
    out = Matrix{Float64}(undef, size(values, 1), size(values, 2))
    allow_missing = size(values, 2) > 1
    @inbounds for j in axes(values, 2), i in axes(values, 1)
        out[i, j] = _aligned_trait_value(values[i, j], allow_missing)
    end
    any(isinf, out) && throw(ArgumentError("trait data contains infinite values"))
    if size(out, 2) == 1
        any(isnan, out) && throw(ArgumentError("single-trait alignment does not allow missing values"))
    else
        @inbounds for i in axes(out, 1)
            all(isnan, @view(out[i, :])) && throw(ArgumentError("trait row $i contains no observed values"))
        end
        @inbounds for j in axes(out, 2)
            all(isnan, @view(out[:, j])) && throw(ArgumentError("trait column $j contains no observed values"))
        end
    end
    return size(out, 2) == 1 ? vec(out) : out
end

function align_traits_to_tree(tree::CompactTree, data::AbstractVector{<:Real}; kwargs...)
    isempty(kwargs) || throw(ArgumentError("keyword arguments are only supported for DataFrame input"))
    length(data) == tree.ntips || throw(ArgumentError("trait length must match tree.ntips"))
    out = Float64.(data)
    all(isfinite, out) || throw(ArgumentError("trait data contains non-finite values"))
    return out
end

function align_traits_to_tree(tree::CompactTree, data::AbstractMatrix{<:Real}; kwargs...)
    isempty(kwargs) || throw(ArgumentError("keyword arguments are only supported for DataFrame input"))
    size(data, 1) == tree.ntips || throw(ArgumentError("trait matrix row count must match tree.ntips"))
    out = Matrix{Float64}(data)
    any(isinf, out) && throw(ArgumentError("trait data contains infinite values"))
    @inbounds for i in axes(out, 1)
        all(isnan, @view(out[i, :])) && throw(ArgumentError("trait row $i contains no observed values"))
    end
    @inbounds for j in axes(out, 2)
        all(isnan, @view(out[:, j])) && throw(ArgumentError("trait column $j contains no observed values"))
    end
    return out
end

function _continuous_dataframe_trait_column(df::AbstractDataFrame; trait_col = nothing)
    cols = Symbol.(names(df))
    length(cols) >= 2 || throw(ArgumentError("DataFrame input must have taxon labels in the first column and one numeric trait column"))
    if trait_col !== nothing
        trait = Symbol(trait_col)
        trait in cols[2:end] || throw(ArgumentError("trait column `$trait` was not found after the taxon column"))
        return cols[1], trait
    end
    trait_cols = Symbol[]
    for col in cols[2:end]
        values = df[!, col]
        usable = true
        has_observed = false
        for val in values
            if ismissing(val) || _continuous_missing_string(val)
                continue
            elseif val isa Real
                has_observed = true
            elseif val isa AbstractString && tryparse(Float64, strip(String(val))) !== nothing
                has_observed = true
            else
                usable = false
                break
            end
        end
        usable && has_observed && push!(trait_cols, col)
    end
    isempty(trait_cols) && throw(ArgumentError("DataFrame input contains no numeric trait column after the first taxon column"))
    length(trait_cols) == 1 || throw(ArgumentError("univariate fits found multiple numeric trait columns ($(join(String.(trait_cols), ", "))); pass trait_col to choose one, or use a multivariate model"))
    return cols[1], only(trait_cols)
end

function _continuous_align_dataframe_trait(tree::CompactTree, data::AbstractDataFrame; trait_col = nothing)
    df = DataFrames.DataFrame(data)
    taxon_col, trait = _continuous_dataframe_trait_column(df; trait_col = trait_col)
    rows = _aligned_dataframe_rows(tree, df, taxon_col)
    y = Vector{Float64}(undef, tree.ntips)
    @inbounds for (i, row) in enumerate(rows)
        val = df[row, trait]
        if ismissing(val) || _continuous_missing_string(val)
            y[i] = NaN
        elseif val isa AbstractString
            parsed = tryparse(Float64, strip(String(val)))
            parsed === nothing && throw(ArgumentError("trait data contains a non-numeric value: `$val`"))
            y[i] = parsed
        else
            y[i] = Float64(val)
        end
    end
    any(isinf, y) && throw(ArgumentError("trait data contains infinite values"))
    all(isnan, y) && throw(ArgumentError("trait column contains no observed values"))
    return Vector{Float64}(y), String(trait)
end

_mv_default_trait_names(p::Integer) = ["trait$(i)" for i in 1:Int(p)]
_mv_default_regime_names(k::Integer) = ["regime$(i)" for i in 1:Int(k)]

function _mv_checked_names(names, n::Integer, what::AbstractString)
    if names === nothing
        return what == "trait" ? _mv_default_trait_names(n) : _mv_default_regime_names(n)
    end
    out = String.(collect(names))
    length(out) == Int(n) || throw(ArgumentError("$(what)_names length must be $(n)"))
    return out
end

function _mv_missing_string(x)
    x isa AbstractString || return false
    s = lowercase(strip(String(x)))
    return isempty(s) || s in ("na", "nan", "missing", "null")
end

function _mv_dataframe_trait_columns(df::AbstractDataFrame)
    cols = Symbol.(names(df))
    length(cols) >= 2 || throw(ArgumentError("DataFrame input must have taxon labels in the first column and at least one trait column"))
    trait_cols = Symbol[]
    for col in cols[2:end]
        values = df[!, col]
        usable = true
        has_observed = false
        for val in values
            if ismissing(val) || _mv_missing_string(val)
                continue
            elseif val isa Real
                has_observed = true
            elseif val isa AbstractString && tryparse(Float64, strip(String(val))) !== nothing
                has_observed = true
            else
                usable = false
                break
            end
        end
        usable && has_observed && push!(trait_cols, col)
    end
    isempty(trait_cols) && throw(ArgumentError("DataFrame input contains no numeric trait columns after the first taxon column"))
    return cols[1], trait_cols
end

function _mv_align_dataframe_traits(tree::CompactTree, data::AbstractDataFrame)
    df = DataFrames.DataFrame(data)
    taxon_col, trait_cols = _mv_dataframe_trait_columns(df)
    X = align_traits_to_tree(tree, df; taxon_col = taxon_col, trait_cols = trait_cols)
    X isa AbstractMatrix || throw(ArgumentError("multivariate fits require at least two numeric trait columns"))
    return Matrix{Float64}(X), String.(trait_cols)
end

function _continuous_regime_names_from_simmap(simmap, nregimes::Integer)
    labels = getproperty(simmap, :state_labels)
    return _continuous_checked_regime_names(isempty(labels) ? nothing : labels, nregimes)
end
