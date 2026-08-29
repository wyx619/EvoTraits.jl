@testset "EvoTraits loads" begin
    @test set_engine_blas_threads!(1) == 1
end

@testset "Newick reader helpers and validation" begin
    temp_dir = mktempdir()
    unicode_path = joinpath(temp_dir, "系统发育树.tre")
    write(unicode_path, "((A:1,B:1)95:1,C:2);")

    @test isdefined(EvoTraits, :_read_newick_text)
    @test isdefined(EvoTraits, :_strip_internal_support_labels)
    @test isdefined(EvoTraits, :_parse_newick_text)
    @test EvoTraits._read_newick_text(unicode_path) == "((A:1,B:1)95:1,C:2);"
    @test EvoTraits._strip_internal_support_labels("((A:1,B:1)95:1,C:2);") == "((A:1,B:1):1,C:2);"
    @test read_tree(unicode_path) !== nothing
    @test_throws ArgumentError read_tree(joinpath(temp_dir, "missing.tre"))

    parsed = read_tree(unicode_path)
    compact_roundtrip = serialize_tree(parsed)
    output_path = joinpath(temp_dir, "written.tre")
    written = write_tree(output_path, compact_roundtrip)
    @test isfile(output_path)
    @test endswith(strip(written), ";")
    @test serialize_tree(read_tree(output_path)).tip_labels == compact_roundtrip.tip_labels

    missing_semicolon = joinpath(temp_dir, "missing_semicolon.tre")
    write(missing_semicolon, "(A:1,B:1)")
    @test_throws ArgumentError read_tree(missing_semicolon)
end

@testset "Tree I/O and preprocessing" begin
    project_root = TEST_PROJECT_ROOT
    tree_path = joinpath(project_root, "validation", "seed_H", "种子植物高度_ultrametric.tre")
    @test isfile(tree_path)
    tree = read_tree(tree_path)
    compact = serialize_tree(tree)

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





