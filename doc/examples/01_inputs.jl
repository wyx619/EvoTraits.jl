include(joinpath(@__DIR__, "common.jl"))
using Test

data = documentation_data()
tree, y = data.tree, data.y
traits = DataFrame(taxon = reverse(tree.tip_labels), value = reverse(y))
aligned = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:value])
@test aligned == y

mktempdir() do dir
    path = joinpath(dir, "tree.nwk")
    write_tree(path, tree)
    loaded = load_tree(path)
    @test Set(loaded.tip_labels) == Set(tree.tip_labels)
    @test loaded.ntips == tree.ntips
end
subset = keep_tip(tree, data.sim, tree.tip_labels[1:12])
@test subset.tree.ntips == 12
@test summary_simmap(subset.tree, subset.simmap).success
println("Input alignment, tree I/O and paired pruning verified.")
