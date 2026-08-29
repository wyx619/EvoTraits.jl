@testset "phyloref prototype" begin
    tree_path = joinpath(mktempdir(), "phyloref_tree.tre")
    write(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")
    tree = serialize_tree(read_tree(tree_path))

    map = build_phyloref(tree)
    @test isa(map, PhyloRef)
    @test length(map.ape_node_ids) == tree.nnodes
    @test length(map.evotraits_node_ids_from_ape) == tree.nnodes
    @test length(map.ape_cladewise_edge_ranks) == tree.nedges
    @test length(map.ape_postorder_edge_ranks) == tree.nedges

    node_table = phyloref_node_table(tree; map = map)
    @test nrow(node_table) == tree.nnodes
    @test Set(names(node_table)) == Set(["evotraits_node_id", "ape_node_id", "is_tip", "label", "tipX", "tipY"])

    edge_table = phyloref_edge_table(tree; map = map)
    @test nrow(edge_table) == tree.nedges
    @test Set(names(edge_table)) == Set([
        "evotraits_edge_id",
        "evotraits_parent_node_id",
        "evotraits_child_node_id",
        "ape_parent_node_id",
        "ape_child_node_id",
        "ape_cladewise_edge_rank",
        "ape_postorder_edge_rank",
        "branch_length",
        "tipX",
        "tipY",
        "descendant_signature",
    ])

    @test node_table.ape_node_id[tree.tipname_to_id["A"]] == 1
    @test node_table.ape_node_id[tree.tipname_to_id["B"]] == 2
    @test node_table.ape_node_id[tree.tipname_to_id["C"]] == 3
    @test node_table.ape_node_id[tree.tipname_to_id["D"]] == 4
    @test maximum(node_table.ape_node_id) == tree.nnodes
    @test node_table.ape_node_id[Int(tree.root)] == tree.ntips + 1
    @test map.evotraits_node_ids_from_ape[node_table.ape_node_id[Int(tree.root)]] == Int(tree.root)

    a_id = Int(tree.tipname_to_id["A"])
    @test EvoTraits.phylo_node_anchor(tree, a_id; map = map) == ("A", "A")

    signatures = EvoTraits.phylo_edge_signatures(tree, collect(1:tree.nedges); map = map)
    @test length(signatures) == tree.nedges
    @test length(unique(signatures)) == tree.nedges
    @test EvoTraits.phylo_edges_from_signatures(tree, signatures; map = map) == collect(1:tree.nedges)

    sigset = Set(edge_table.descendant_signature)
    @test "A" in sigset
    @test "B" in sigset
    @test "C" in sigset
    @test "D" in sigset
    @test "A|B" in sigset
    @test "C|D" in sigset

    ab_row = only(findall(==("A|B"), edge_table.descendant_signature))
    @test edge_table.tipX[ab_row] == "A"
    @test edge_table.tipY[ab_row] == "B"

    cd_row = only(findall(==("C|D"), edge_table.descendant_signature))
    @test edge_table.tipX[cd_row] == "C"
    @test edge_table.tipY[cd_row] == "D"

    @test edge_table.ape_cladewise_edge_rank == [1, 4, 2, 3, 5, 6]
    @test edge_table.ape_postorder_edge_rank == [5, 6, 3, 4, 1, 2]

    @test EvoTraits.R_node_id(tree, Int(tree.root); map = map) == tree.ntips + 1
    @test EvoTraits.evotraits_node_id(tree, tree.ntips + 1; map = map) == Int(tree.root)
    @test [EvoTraits.R_node_id(tree, node; map = map) for node in [Int(tree.root), a_id]] == [tree.ntips + 1, 1]
    @test [EvoTraits.evotraits_node_id(tree, node; map = map) for node in [tree.ntips + 1, 1]] == [Int(tree.root), a_id]
    @test EvoTraits.R_edge_id_cladewise(tree, 1; map = map) == 1
    @test EvoTraits.R_edge_id_cladewise(tree, 2; map = map) == 4
    @test EvoTraits.R_edge_id_postorder(tree, 5; map = map) == 1
    @test EvoTraits.evotraits_edge_id_from_R_cladewise(tree, 4; map = map) == 2
    @test EvoTraits.evotraits_edge_id_from_R_postorder(tree, 1; map = map) == 5
    @test [EvoTraits.R_edge_id_cladewise(tree, edge; map = map) for edge in [1, 2, 5]] == [1, 4, 5]
    @test [EvoTraits.R_edge_id_postorder(tree, edge; map = map) for edge in [1, 2, 5]] == [5, 6, 1]
    @test [EvoTraits.evotraits_edge_id_from_R_cladewise(tree, edge; map = map) for edge in [1, 4, 5]] == [1, 2, 5]
    @test [EvoTraits.evotraits_edge_id_from_R_postorder(tree, edge; map = map) for edge in [5, 6, 1]] == [1, 2, 5]

    @test EvoTraits.R_edge_matrix(tree; order = :cladewise, map = map) == [
        5 6;
        6 1;
        6 2;
        5 7;
        7 3;
        7 4
    ]
    @test EvoTraits.R_edge_matrix(tree; order = :postorder, map = map) == [
        7 3;
        7 4;
        6 1;
        6 2;
        5 6;
        5 7
    ]

    node_df = DataFrame(node_id = [Int(tree.root), a_id])
    node_attached = EvoTraits.attach_R_node_map(node_df, tree; map = map)
    @test names(node_attached) == ["node_id", "R_node_id", "R_is_tip", "R_label", "R_tipX", "R_tipY"]
    @test node_attached.R_node_id == [tree.ntips + 1, 1]
    @test node_attached.R_tipX == ["A", "A"]
    @test node_attached.R_tipY == ["C", "A"]

    edge_df = DataFrame(edge_id = [1, 2, 5])
    edge_attached = EvoTraits.attach_R_edge_map(edge_df, tree; order = :postorder, map = map)
    @test names(edge_attached) == ["edge_id", "R_edge_id", "R_parent_node_id", "R_child_node_id", "branch_length", "tipX", "tipY", "descendant_signature"]
    @test edge_attached.R_edge_id == [5, 6, 1]
    @test edge_attached.tipX == ["A", "C", "C"]
    @test edge_attached.tipY == ["B", "D", "C"]
end

@testset "phyloref ape ordering on asymmetric tree" begin
    tree_path = joinpath(mktempdir(), "phyloref_tree2.tre")
    write(tree_path, "(((A:1,B:1):1,C:1):1,(D:1,E:1):1);")
    tree = serialize_tree(read_tree(tree_path))
    map = build_phyloref(tree)
    node_table = phyloref_node_table(tree; map = map)
    edge_table = phyloref_edge_table(tree; map = map)

    @test node_table.ape_node_id[Int(tree.root)] == tree.ntips + 1
    @test sort(node_table.ape_node_id[.!node_table.is_tip]) == collect((tree.ntips + 1):tree.nnodes)

    @test edge_table.ape_cladewise_edge_rank == [1, 6, 2, 5, 3, 4, 7, 8]
    @test edge_table.ape_postorder_edge_rank == [7, 8, 5, 6, 3, 4, 1, 2]

    root_children = findall(==(Int(tree.root)), edge_table.evotraits_parent_node_id)
    @test sort(edge_table.ape_cladewise_edge_rank[root_children]) == [1, 6]
    @test sort(edge_table.ape_postorder_edge_rank[root_children]) == [7, 8]

    @test EvoTraits.R_edge_matrix(tree; order = :cladewise, map = map) == [
        6 7;
        7 8;
        8 1;
        8 2;
        7 3;
        6 9;
        9 4;
        9 5
    ]
    @test EvoTraits.R_edge_matrix(tree; order = :postorder, map = map) == [
        9 4;
        9 5;
        8 1;
        8 2;
        7 8;
        7 3;
        6 7;
        6 9
    ]
end



