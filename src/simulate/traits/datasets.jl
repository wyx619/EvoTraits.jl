"""
    simulate_mvbm1_dataset(n_tips, Sigma; kwargs...)

Generate a complete BM1 simulation bundle with tree, tip labels, trait names,
and simulated traits.
"""
function simulate_mvbm1_dataset(
    n_tips::Integer,
    Sigma::AbstractMatrix;
    tree_height::Real = 1.0,
    tip_prefix::AbstractString = "t",
    trait_names = nothing,
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    simtree = simulate_yule_simtree(
        n_tips;
        tree_height = tree_height,
        tip_prefix = tip_prefix,
        rng = rng,
    )
    tree = to_compact_tree(simtree)
    newick = to_newick(simtree)
    traits = simulate_trait(:BM1, tree, Sigma; root_state = root_state, rng = rng)
    p = size(Sigma, 1)
    names = trait_names === nothing ? ["trait$(i)" for i in 1:p] : collect(String, trait_names)
    length(names) == p || throw(ArgumentError("trait_names length must match Sigma size"))
    return (
        newick = newick,
        tree = tree,
        tip_labels = copy(tree.tip_labels),
        trait_names = names,
        traits = traits,
    )
end

"""
    simulate_trait(model, tree, args...; kwargs...)

Dispatch to one of the supported trait simulation routines. Both canonical
model names (`:BM1`, `:BMM`, `:EB`, `:OU1`, `:OUM`, `:OUMV`, `:OUMA`, `:OUMVA`)
and legacy multivariate names (`:mvBM1`, `:mvBMM`, `:mvEB`, `:mvOU1`, `:mvOUM`,
`:mvOUMV`, `:mvOUMA`, `:mvOUMVA`) are accepted.
"""
function simulate_trait(model::Symbol, tree::CompactTree, args...; kwargs...)
    if model === :BM1 || model === :mvBM1
        return simulate_mvbm1(tree, args...; kwargs...)
    elseif model === :BMM || model === :mvBMM
        return simulate_mvbmm(tree, args...; kwargs...)
    elseif model === :EB || model === :mvEB
        return simulate_mveb(tree, args...; kwargs...)
    elseif model === :OU1 || model === :mvOU1
        return simulate_mvou1(tree, args...; kwargs...)
    elseif model === :OUM || model === :mvOUM
        return simulate_mvoum(tree, args...; kwargs...)
    elseif model === :OUMVA || model === :mvOUMVA
        return simulate_mvoumva(tree, args...; kwargs...)
    end
    throw(ArgumentError(
        "Unsupported trait simulation model $model. Supported models are " *
        ":BM1, :BMM, :EB, :OU1, :OUM, :OUMVA and legacy :mv* aliases."
    ))
end

"""
    simulate_mv_data(model, tree, args...; kwargs...)

Legacy alias for `simulate_trait`.
"""
function simulate_mv_data(model::Symbol, tree::CompactTree, args...; kwargs...)
    return simulate_trait(model, tree, args...; kwargs...)
end
