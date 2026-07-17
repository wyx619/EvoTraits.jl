@testset "Real multivariate small-data validation" begin
    asset_root = joinpath(TEST_PROJECT_ROOT, "validation", "multivariate")
    parsed = read_simmap(joinpath(asset_root, "VDH", "data", "VDH_R_mvOUM_simmap.tre"); format = :phylip, version = 1.5, rev_order = true)

    rows = split.(readlines(joinpath(asset_root, "VD_H_4_states_data.csv"))[2:end], ',')
    labels = replace.(replace.(getindex.(rows, 1), "\"" => ""), " " => "_")
    trait_rows = Vector{Float64}[]
    kept_labels = String[]
    for i in eachindex(rows)
        if rows[i][3] != "NA" && rows[i][4] != "NA"
            push!(kept_labels, labels[i])
            push!(trait_rows, parse.(Float64, rows[i][3:4]))
        end
    end
    trait_map = Dict(kept_labels[i] => trait_rows[i] for i in eachindex(kept_labels))

    keep = [lab for lab in kept_labels if lab in Set(parsed.tree.tip_labels)][1:24]
    kept = keep_tip_simmap(parsed.tree, parsed.simmap, keep)
    trait = reduce(vcat, [trait_map[lab]' for lab in kept.tree.tip_labels])

    @test kept.tree.ntips == 24
    @test size(trait) == (24, 2)
    @test kept.simmap.nstates == 4

    bm_fit = fit_mvbm1(kept.tree, trait; max_iterations = 40, rel_tol = 1e-6)
    @test bm_fit.success
    @test bm_fit.model == :mvBM1
    bm_asr = estim_node(kept.tree, trait, bm_fit)
    @test bm_asr.success
    @test size(bm_asr.estimates) == (kept.tree.ntips - 1, 2)

    missing_trait = copy(trait)
    missing_trait[2, 1] = NaN
    missing_trait[7, :] .= NaN
    missing_fit = fit_mvbm1(kept.tree, missing_trait; guess_sigma = bm_fit.sigma, max_iterations = 20, rel_tol = 1e-6)
    @test missing_fit.success
    missing_asr = estim_node(kept.tree, missing_trait, missing_fit)
    @test missing_asr.success

    bmm_fit = fit_mvbmm(kept.tree, trait, kept.simmap.mapped_edge; max_iterations = 40, rel_tol = 1e-6)
    @test bmm_fit.success
    @test bmm_fit.model == :mvBMM
    bmm_branch = estim_branch_for_simmap(kept.tree, trait, bmm_fit; edge_segments = kept.simmap.edge_segments)
    @test bmm_branch.success
    @test length(bmm_branch.edge_ids) == sum(length, kept.simmap.edge_segments)
    @test size(bmm_branch.start_estimates, 2) == 2
end

