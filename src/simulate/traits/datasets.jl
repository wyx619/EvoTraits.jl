"""
    sim_mvbm1_dataset(n_tips, Sigma; kwargs...)

Generate a complete BM1 simulation bundle with tree, tip labels, trait names,
and simulated traits.
"""
function sim_mvbm1_dataset(
    n_tips::Integer,
    Sigma::AbstractMatrix;
    tree_height::Real = 1.0,
    tip_prefix::AbstractString = "t",
    trait_names = nothing,
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    simtree = sim_yule_simtree(
        n_tips;
        tree_height = tree_height,
        tip_prefix = tip_prefix,
        rng = rng,
    )
    tree = serialize_tree(simtree)
    newick = to_newick(simtree)
    traits = sim_trait(:BM1, tree, Sigma; root_state = root_state, rng = rng)
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
    sim_trait(model, tree, args...; kwargs...)

Dispatch to one of the supported trait simulation routines. Both canonical
model names (`:BM1`, `:BMM`, `:EB`, `:OU1`, `:OUM`, `:OUMV`, `:OUMA`, `:OUMVA`)
and legacy multivariate names (`:mvBM1`, `:mvBMM`, `:mvEB`, `:mvOU1`, `:mvOUM`,
`:mvOUMV`, `:mvOUMA`, `:mvOUMVA`) are accepted.
"""
function sim_trait(model::Symbol, tree::CompactTree, args...; kwargs...)
    if model === :BM1 || model === :mvBM1
        return sim_mvbm1(tree, args...; kwargs...)
    elseif model === :BMM || model === :mvBMM
        return sim_mvbmm(tree, args...; kwargs...)
    elseif model === :EB || model === :mvEB
        return sim_mveb(tree, args...; kwargs...)
    elseif model === :OU1 || model === :mvOU1
        return sim_mvou1(tree, args...; kwargs...)
    elseif model === :OUM || model === :mvOUM
        return sim_mvoum(tree, args...; kwargs...)
    elseif model === :OUMVA || model === :mvOUMVA
        return sim_mvoumva(tree, args...; kwargs...)
    end
    throw(ArgumentError(
        "Unsupported trait simulation model $model. Supported models are " *
        ":BM1, :BMM, :EB, :OU1, :OUM, :OUMVA and legacy :mv* aliases."
    ))
end

"""
    sim_mv_data(model, tree, args...; kwargs...)

Legacy alias for `sim_trait`.
"""
function sim_mv_data(model::Symbol, tree::CompactTree, args...; kwargs...)
    return sim_trait(model, tree, args...; kwargs...)
end
