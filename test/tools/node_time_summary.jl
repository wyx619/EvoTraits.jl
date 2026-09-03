@testset "node time summary tool" begin
    tbl_uni = DataFrame(
        node_id = 1:5,
        time_before_present = [1.0, 2.0, 6.0, 7.0, 12.0],
        estimate = [10.0, 20.0, 30.0, 40.0, 50.0],
    )
    res_uni = summarize_node_estimates_by_time(tbl_uni, 5.0)
    @test haskey(res_uni, "trait")
    df_uni = res_uni["trait"]
    @test nrow(df_uni) == 4
    @test df_uni.start == [0.0, 0.0, 5.0, 10.0]
    @test df_uni.end == [0.0, 5.0, 10.0, 15.0]
    @test df_uni.n_nodes == [5, 2, 2, 1]
    @test df_uni.mean == [30.0, 15.0, 35.0, 50.0]
    @test df_uni.median == [30.0, 15.0, 35.0, 50.0]

    res_smooth = summarize_node_estimates_by_time(tbl_uni, 5.0; w = 2.0)
    df_smooth = res_smooth["trait"]
    @test df_smooth.start == [0.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    @test df_smooth.end == [0.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    @test df_smooth.n_nodes == [5, 2, 2, 2, 2, 1]
    @test df_smooth.mean == [30.0, 15.0, 25.0, 35.0, 35.0, 50.0]

    tbl_mv = DataFrame(
        node_id = 1:4,
        time_before_present = [1.0, 4.0, 6.0, 9.0],
        estimate_1 = [1.0, 2.0, 3.0, 4.0],
        estimate_2 = [10.0, 20.0, 30.0, 40.0],
    )
    res_mv = summarize_node_estimates_by_time(tbl_mv, 5.0)
    @test Set(keys(res_mv)) == Set(["1", "2"])
    @test res_mv["1"].mean == [2.5, 1.5, 3.5]
    @test res_mv["2"].mean == [25.0, 15.0, 35.0]

    tbl_named = DataFrame(
        node_id = 1:4,
        time_before_present = [0.5, 2.5, 5.5, 7.5],
        estimate_lnH = [2.0, 4.0, 6.0, 8.0],
        estimate_lnVD = [1.0, 3.0, 5.0, 7.0],
    )
    res_named = summarize_node_estimates_by_time(tbl_named, 5.0)
    @test Set(keys(res_named)) == Set(["lnH", "lnVD"])
    @test res_named["lnH"].mean == [5.0, 3.0, 7.0]
    @test res_named["lnVD"].median == [4.0, 2.0, 6.0]

    tbl_regime = DataFrame(
        node_id = 1:5,
        time_before_present = [1.0, 2.0, 6.0, 7.0, 12.0],
        estimate_lnH = [10.0, 20.0, 30.0, 40.0, 50.0],
        regime_id = [1, 1, 2, 2, 1],
        regime = ["A", "A", "B", "B", "A"],
    )
    res_regime = summarize_node_estimates_by_time(tbl_regime, 5.0)
    @test haskey(res_regime, "lnH")
    @test haskey(res_regime, "_regime_summary")
    regime_df = res_regime["_regime_summary"]
    @test nrow(regime_df) == 8
    @test regime_df.start == [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 10.0, 10.0]
    @test regime_df.end == [0.0, 0.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0]
    @test regime_df.regime == ["A", "B", "A", "B", "A", "B", "A", "B"]
    @test regime_df.n_nodes == [3, 2, 2, 0, 0, 2, 1, 0]
    @test regime_df.proportion == [0.6, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]

    @test_throws ArgumentError summarize_node_estimates_by_time(tbl_uni, 0.0)
    @test_throws ArgumentError summarize_node_estimates_by_time(tbl_uni, 5.0; w = 5.0)
    @test_throws ArgumentError summarize_node_estimates_by_time(DataFrame(node_id = 1:3, estimate = [1.0, 2.0, 3.0]), 5.0)
end
