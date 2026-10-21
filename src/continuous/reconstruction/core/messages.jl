@inline function _gaussian_product(mean1::Float64, var1::Float64, mean2::Float64, var2::Float64)
    if isinf(var1)
        return (mean = mean2, var = var2)
    elseif isinf(var2)
        return (mean = mean1, var = var1)
    elseif var1 == 0.0 && var2 == 0.0
        abs(mean1 - mean2) <= 1e-8 || return (mean = NaN, var = NaN)
        return (mean = 0.5 * (mean1 + mean2), var = 0.0)
    elseif var1 == 0.0
        return (mean = mean1, var = 0.0)
    elseif var2 == 0.0
        return (mean = mean2, var = 0.0)
    end

    invvar = 1.0 / var1 + 1.0 / var2
    var = 1.0 / invvar
    mean = var * (mean1 / var1 + mean2 / var2)
    return (mean = mean, var = var)
end

@inline function _information_to_gaussian(precision::Float64, linear::Float64)
    precision >= 0.0 || return (mean = NaN, var = NaN)
    precision == 0.0 && return (mean = 0.0, var = Inf)
    mean = linear / precision
    isfinite(mean) || return (mean = NaN, var = NaN)
    return (mean = mean, var = 1.0 / precision)
end

@inline function _edge_information_to_parent(
    child_precision::Float64,
    child_linear::Float64,
    a::Float64,
    b::Float64,
    v::Float64,
)
    (isfinite(a) && a >= 0.0 && isfinite(v) && v >= 0.0 && child_precision >= 0.0) ||
        return (success = false, precision = 0.0, linear = 0.0)
    denominator = muladd(v, child_precision, 1.0)
    isfinite(denominator) && denominator > 0.0 ||
        return (success = false, precision = 0.0, linear = 0.0)
    precision = (a * (child_precision / denominator)) * a
    linear = a * (child_linear - b * child_precision) / denominator
    return (success = true, precision = precision, linear = linear)
end

@inline function _edge_message_to_parent(child_mean::Float64, child_var::Float64, a::Float64, b::Float64, v::Float64)
    (isfinite(a) && a >= 0.0 && isfinite(v) && v >= 0.0) || return (mean = NaN, var = NaN)
    a == 0.0 && return (mean = 0.0, var = Inf)
    total_var = child_var + v
    parent_var = total_var / (a * a)
    isfinite(parent_var) || return (mean = 0.0, var = Inf)
    return (mean = (child_mean - b) / a, var = parent_var)
end

@inline function _edge_predict_to_child(parent_mean::Float64, parent_var::Float64, a::Float64, b::Float64, v::Float64)
    (isfinite(a) && a >= 0.0 && isfinite(v) && v >= 0.0) || return (mean = NaN, var = NaN)
    a == 0.0 && return (mean = b, var = v)
    if isinf(parent_var)
        return (mean = 0.0, var = Inf)
    end
    return (mean = a * parent_mean + b, var = a * a * parent_var + v)
end

@inline function _scalar_observation_info_to_parent(y::Float64, a::Float64, b::Float64, v::Float64)
    (isfinite(a) && a >= 0.0 && isfinite(v) && v >= 0.0) || return (success = false, precision = 0.0, linear = 0.0, logconst = -Inf)
    isnan(y) && return (success = true, precision = 0.0, linear = 0.0, logconst = 0.0)
    vv = max(v, 1e-12)
    resid = y - b
    invv = 1.0 / vv
    return (
        success = true,
        precision = (a * invv) * a,
        linear = a * resid * invv,
        logconst = -0.5 * (resid * resid * invv + log(vv) + log(2 * pi)),
    )
end

@inline function _scalar_info_to_parent(
    child_precision::Float64,
    child_linear::Float64,
    child_logconst::Float64,
    a::Float64,
    b::Float64,
    v::Float64,
)
    (isfinite(a) && a >= 0.0 && isfinite(v) && v >= 0.0 && child_precision >= 0.0) ||
        return (success = false, precision = 0.0, linear = 0.0, logconst = -Inf)
    vv = max(v, 1e-12)
    denominator = muladd(vv, child_precision, 1.0)
    (isfinite(denominator) && denominator > 0.0) ||
        return (success = false, precision = 0.0, linear = 0.0, logconst = -Inf)
    # Avoid subtracting nearly equal precisions or Gaussian normalizers.
    scaled_precision = child_precision / denominator
    scaled_linear = child_linear / denominator
    precision = (a * scaled_precision) * a
    linear = a * muladd(-b, scaled_precision, scaled_linear)
    correction = 0.5 * (vv * scaled_linear) * child_linear +
                 b * scaled_linear - 0.5 * (b * scaled_precision) * b
    logconst = child_logconst - 0.5 * log1p(vv * child_precision) + correction
    return (success = isfinite(logconst) && isfinite(linear), precision = precision, linear = linear, logconst = logconst)
end

@inline function _scalar_root_info_loglik(
    precision::Float64,
    linear::Float64,
    logconst::Float64,
    root_prior_mean::Float64,
    root_prior_var::Float64,
    profile_root::Bool,
)
    if profile_root || isinf(root_prior_var)
        precision > 0.0 || return (success = false, loglik = -Inf, root_state = NaN)
        root_state = linear / precision
        loglik = logconst + 0.5 * linear * root_state
        return (success = isfinite(loglik), loglik = loglik, root_state = root_state)
    elseif root_prior_var <= 1e-12
        root_state = root_prior_mean
        loglik = logconst - 0.5 * precision * root_state * root_state + linear * root_state
        return (success = isfinite(loglik), loglik = loglik, root_state = root_state)
    else
        denominator = muladd(root_prior_var, precision, 1.0)
        isfinite(denominator) && denominator > 0.0 ||
            return (success = false, loglik = -Inf, root_state = NaN)
        scaled_precision = precision / denominator
        scaled_linear = linear / denominator
        root_state = root_prior_mean / denominator + root_prior_var * scaled_linear
        correction = 0.5 * (root_prior_var * scaled_linear) * linear +
                     root_prior_mean * scaled_linear -
                     0.5 * (root_prior_mean * scaled_precision) * root_prior_mean
        loglik = logconst - 0.5 * log1p(root_prior_var * precision) + correction
        return (success = isfinite(loglik), loglik = loglik, root_state = root_state)
    end
end
