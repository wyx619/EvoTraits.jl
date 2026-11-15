# Stochastic character mapping

[中文](simmap_zh.md) · [Index](README.md) · [Discrete models](discrete.md) · [Multivariate models](multivariate.md)

SIMMAP samples branchwise histories conditional on a fitted discrete model. Each map is one possible history drawn from that fitted model.

```julia
using Random
maps = simmap_corhmm(fit; nsim = 100, rng = MersenneTwister(2026))
maps.success
sim = maps.collapsed_samples[1]
summary_simmap(fit.tree, sim)
events = transition_events_table(fit.tree, sim)
```

Here `fit` is a successful `fit_corhmm` result. `maps.samples` retains hidden states; `collapsed_samples` maps them to observed states. Keep the map paired with the tree on which it was generated. `write_simmap(tree, sim; file="history.simmap")` writes one history; `read_simmap` returns the parsed tree and map.

```julia
subset = keep_tip(fit.tree, sim, ["A", "B", "C"])
subset.tree
subset.simmap
```

Use the returned tree together with the returned map for downstream fitting. The older `keep_tip_simmap` and `drop_tip_simmap` names remain available. `transition_times` gives events, while `transition_rates_through_time` summarizes transitions divided by source-state branch exposure.

## A map connects discrete and continuous models

A discrete fit estimates rates and node-state distributions. A SIMMAP additionally samples **where along each branch** state changes occurred. OUM-type models use these segments to propagate the continuous process through successive regimes. Node marginal probabilities carry no information about where changes occur within a branch.

`SimmapSample` contains `edge_segments`, `mapped_edge`, `nstates`, `state_labels` and `root_state`. Segment durations must sum to the corresponding branch length. State IDs index the map's labels. Use `summary_simmap(tree, sim)` before fitting after an import or transformation.

## Saving histories without retaining 100 maps

`simmap_corhmm(fit; nsim=100)` returns all sampled maps in memory. For incremental output, use a persistent RNG and sample one at a time:

```julia
using Random
mkpath("simmaps")
rng = MersenneTwister(2026)
for i in 1:100
    draw = simmap_corhmm(fit; nsim = 1, rng = rng)
    draw.success || error("Mapping failed at draw $i")
    sim = only(draw.collapsed_samples)
    filename = lpad(string(i), 3, '0') * ".simmap"
    write_simmap(fit.tree, sim; file = joinpath("simmaps", filename))
    println("Saved map $i/100")
end
```

Do not re-create the same seeded RNG in every iteration: that repeats the same random stream.

To store 100 histories in one text file, open it once and write one mapped Newick per line:

```julia
open("histories.simmap", "w") do io
    for sim in maps.collapsed_samples
        println(io, write_simmap(fit.tree, sim))
    end
end
```

`read_simmap("history.simmap")` returns a named tuple with `tree`, `simmap` and `newick` for a single history; a multi-history source returns a vector of these tuples. The default format is auto-detected on reading. State order and segment direction matter when exchanging files; preserve the writer/reader conventions and check branch totals after a round trip.

## Using the same history for fitting and ASR

```julia
parsed = read_simmap("history.simmap")
tree2, sim2 = parsed.tree, parsed.simmap
# Realign trait rows to tree2.tip_labels before fitting.
fit2 = fit_oum(tree2, y2, sim2)
nodes2 = estim_node(tree2, y2, fit2, sim2)
```

Here `y2` must be constructed for the imported tree. Do not mix the original tree's edge arrays with a newly parsed tree merely because their tip sets match.

For hidden-rate fits, `samples` uses hidden states and `collapsed_samples` uses observed states. Choose the layer matching the continuous regime hypothesis. Summarize results over multiple maps to assess sensitivity to mapped histories.

See [the discrete-to-SIMMAP example](examples/02_discrete_simmap.jl) and [comparison workflows](workflows.md).
