# Troubleshooting

[中文](troubleshooting_zh.md) · [Index](README.md) · [Fitting](fitting.md)

| Symptom | Check | Next action |
| --- | --- | --- |
| Missing tree taxon during alignment | Exact names, synonyms, whitespace | Resolve names or prune explicitly; realign |
| Duplicate taxon error | Multiple source rows per species | Aggregate under a documented rule |
| Unexpected trait column | Numeric metadata or several numeric traits | Specify trait_col, or align selected columns first |
| All-missing matrix row/column | At least one observation required | Revise the analysis set; do not insert zeros |
| Invalid map or regime count | Tree identity, edge totals, labels | Read/prune tree and map as a pair |
| Newick parse failure | Final semicolon, malformed tree | Repair source syntax; do not silently concatenate trees |
| Budget exhausted | Function calls and optimizer stopping reason | Increase budget; compare supported alternative starts |
| Finite logLik but no convergence | Model-specific validity versus optimizer status | Keep diagnostic status; do not call it converged |
| Implausible OU optima | Mean-design rank and regime exposure | Inspect mean_diagnostics and mapped history |
| Repeated ASR estimates | α × branch length and proximity to θ | Test OU memory loss and identifiability |
| Negative θ on log data | Trait scale | Negative log optima are allowed |
| Extreme EnvExp β | Environment scaling, coverage, smoothing | Check rate curve and quadrature sensitivity |
| Very slow batch | Compilation, parallel nesting, memory | Separate setup from fitting; benchmark one case |
| Unexpected time-zero summary | Extra whole-tree baseline | Exclude the first summary row before plotting |

## "The root probability disagrees with my expectation"

Check whether you read `root_prior_probs` or the marginal reconstruction. Locate the root by ID and inspect all state probabilities, preserving the declared state order. See [discrete root extraction](discrete.md).

## "I used eight threads; why only one trial?"

Julia threads are available workers, not an automatic replicate count. corHMM defaults to one trial and one requested worker. Set both explicitly. Continuous families have their own candidate scheduling, so do not pass invented thread keywords to them.

## "The best AIC proves this classification is right"

AIC evaluates a defined likelihood and candidate set. Different discrete coding schemes change observations. Continuous models conditional on different SIMMAP histories need sensitivity assessment. See [comparison rules](workflows.md).

## "Can I mix old root names with the current API?"

Use current public names: mean `:regime` / `:stationary`, covariance `:fixed` / `:random`, where supported. OUMV/OUMA/OUMVA support only `:regime + :fixed`. Independent free root-mean estimation is not exposed. Do not use removed legacy combinations from old validation scripts.

## Reporting a reproducible problem

Include package and Julia versions, exact function call, a small tree, aligned observations, the matching SIMMAP when applicable, a fixed seed and the complete error message. Include convergence information and fixed-parameter comparisons if the concern is a likelihood discrepancy. Avoid reporting only a screenshot or an AIC value.
