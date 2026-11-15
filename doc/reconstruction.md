# Ancestral reconstruction

[中文](reconstruction_zh.md) · [Index](README.md) · [Continuous models](continuous.md)

Use `estim_node(tree, trait, fit; ...)` for continuous node reconstruction and `asr_corhmm(fit; mode=...)` for discrete reconstruction. Regime-aware continuous reconstruction requires the same `edge_segments` or SIMMAP used for the fit.

```julia
fit = fit_ou1(tree, y)
node_asr = estim_node(tree, y, fit)
node_table = estim_node_table(tree, node_asr)

fit2 = fit_oum(tree, y, sim)
node_asr2 = estim_node(tree, y, fit2, sim)
branch_asr = estim_branch_for_simmap(tree, y, fit2, sim)
branch_table = estim_branch_table(tree, branch_asr)
```

Tables preserve internal node or branch identity and time coordinates. `summarize_node_estimates_by_time(table, width; w=step)` produces sliding-window summaries. A summary is conditional on the fitted model, root treatment and mapped history; it is not an independent observation.

`width` is the window width and `w` the step; `w=0` selects non-overlapping windows. The first row of each trait summary is an overall baseline across nodes, not a time-zero estimate. Within-window quantiles summarize reconstructed node estimates, not confidence intervals for an evolutionary trajectory.

For discrete fits, marginal ASR gives probabilities at each node, joint ASR gives one joint state configuration, scaled follows the scaled likelihood convention, and none skips reconstruction. Keep marginal probabilities when downstream analyses need uncertainty.

## An end-to-end extraction

```julia
fit = fit_oum(tree, y, sim; max_iterations = 1000)
node_asr = estim_node(tree, y, fit, sim)
node_tbl = estim_node_table(tree, node_asr; simmap = sim)
windows = summarize_node_estimates_by_time(node_tbl, 4.0; w = 2.0)
trait_windows = windows["trait"]
# Remove the first, whole-tree baseline row for a temporal plot.
time_windows = trait_windows[2:end, :]
```

The map must be the one used for fitting. A table with `regime` labels identifies the sampled history, not a corHMM marginal posterior. To use marginal weights, join the discrete marginal probabilities by node ID from the **same tree**, retaining one probability column per state.

## Fields, scales and uncertainty

| ASR field | Interpretation |
| --- | --- |
| `node_ids` | Internal node identifiers |
| `estimates` | Conditional ancestral estimates |
| `variances`, `se` | Conditional uncertainty, given the fitted model |
| `time_from_root` | Time elapsed since the root |
| `time_before_present` | Node age |
| `tip_ids`, `tip_labels` | Tip identity associated with the result |

Node output is not automatically a tip-and-internal-node table. Check the requested table and identifiers before joining external annotations. For multivariate reconstruction, `estim_node_table(tree, asr, fit)` uses fitted trait names in columns such as `estimate_lnH`, `variance_lnH` and `se_lnH`.

If the fitted response is ln(H), `exp(estimate)` is a back-transformed point estimate and the conditional lognormal median. Under a Gaussian log-scale approximation, the conditional mean is `exp(estimate + variance/2)`. Neither is automatically an unbiased population mean. ASR variance does not include all uncertainty in fitted parameters, root assumptions or SIMMAP sampling.

## Sliding windows

The positional argument of `summarize_node_estimates_by_time(table, n; w=0)` is the **window width**, not a requested number of nodes. All time quantities use the tree's time unit.

| Setting | Meaning |
| --- | --- |
| `n=4.0, w=2.0` | Width 4, successive starts 2 units apart |
| `n=4.0, w=0.0` | Non-overlapping windows of width 4 |
| Valid sliding step | `0 < w < n` |
| `n_nodes` | Finite estimates contributing to each window |
| `mean`, `median` | Summaries of reconstructed node point estimates |
| `q2_5`, `q97_5` | Quantiles across those point estimates |

A univariate table returns its trait summary under `"trait"`. Multivariate tables return one entry per trait suffix. Tables with regime columns additionally return `"_regime_summary"`. The first baseline row has `start=end=0` and summarizes all nodes; exclude it from a time curve. Empty windows have no finite estimate. Node density varies through time, and related nodes are not independent observations. The quantile band is therefore not a confidence interval around an evolutionary trajectory.

For a time plot, use window midpoints `(start + end)/2` and reverse the horizontal axis if present day should be on the right. A plotting smoother is a display choice, not a replacement for the model.

## Branch output and limitations

`estim_branch_for_simmap` reconstructs values for supported mapped univariate models, including OUM, OUMV, OUMA, OUMVA and EBM. `estim_branch_table` reports segment time coordinates and estimates; an edge can appear more than once because it contains multiple segments. Do not assume one row per edge.

Environmental and shift fits do not currently expose this ASR workflow. See [environment and shifts](environment_shifts.md), [troubleshooting](troubleshooting.md) and [the executable ASR example](examples/03_continuous.jl).
