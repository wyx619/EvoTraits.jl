"""
    mk_loglikelihood(tree, tip_priors, Q; root_prior=:likelihoods, root_prior_probs=nothing, nparams=nothing)

Compute the Mk log-likelihood on a preprocessed `CompactTree` using a pruning pass.
This layer only exposes likelihood and AIC bookkeeping.
"""
function mk_loglikelihood(
    tree::CompactTree,
    tip_priors::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real};
    root_prior::Symbol = :likelihoods,
    root_prior_probs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    nparams::Union{Nothing, Integer} = nothing,
)
    cache = mk_pruning_cache(
        tree,
        tip_priors,
        Q;
        root_prior = root_prior,
        root_prior_probs = root_prior_probs,
        nparams = nparams,
    )
    return MkLikelihoodResult(
        success = cache.success,
        loglik = cache.loglik,
        aic = cache.aic,
        nstates = cache.nstates,
        nparams = cache.nparams,
        root_prior = cache.root_prior,
        scaling_shift = cache.scaling_shift,
    )
end
