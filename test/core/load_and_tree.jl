@testset "EvoTraits loads" begin
    @test set_engine_blas_threads!(1) == 1
end

@testset "Tree I/O and preprocessing" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    @test isfile(tree_path)
    tree = load_newick_tree(tree_path)
    compact = to_compact_tree(tree)

    @test compact.ntips > 1000
    @test compact.nnodes == 2 * compact.ntips - 1
    @test compact.nedges == compact.nnodes - 1
    @test length(compact.postorder) == compact.nnodes
    @test length(compact.preorder) == compact.nnodes
    @test length(compact.postorder_internal) == compact.ntips - 1
    @test sum(compact.is_tip) == compact.ntips
    @test compact.dist_from_root[compact.root] == 0.0
    @test length(compact.tip_labels) == compact.ntips
end





