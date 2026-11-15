# Preparing inputs

[中文](inputs_zh.md) · [Index](README.md) · [Next: fitting](fitting.md)

## Minimum data contract

| Input | Required structure | Ordering |
| --- | --- | --- |
| Tree | A dated/branch-length Newick tree, loaded as CompactTree | Tree defines node and edge identities |
| Single trait | Numeric vector of length ntips | tree.tip_labels |
| Multiple traits | Numeric matrix, ntips rows | Tips in rows, traits in declared column order |
| Discrete observations | Taxon-to-state mapping or supported table | Explicit state_order controls state indices |
| Regime history | SIMMAP paired to that tree | Its original edge indices and state labels |
| Environmental history | Ages before present and environmental values | Covers present through tree root age |

Row order matters: permuting rows changes the association between observations and tips.

## Constructing an analysis subset

```julia
using EvoTraits, DataFrames
tree = load_tree("tree.nwk")
traits = DataFrame(taxon = ["A", "B", "C"],
                   height = [1.2, 2.0, 0.8])
keep = intersect(tree.tip_labels, String.(traits.taxon))
length(keep) >= 3 || error("Too few shared taxa")
tree = keep_tip(tree, keep)
y = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:height])
```

The species names are exact identifiers. Resolve synonyms, duplicate records and whitespace before alignment. If a source has multiple records per species, choose and document an aggregation rule before fitting.

The example assumes those named taxa are present. For a fully self-contained demonstration use [01_inputs.jl](examples/01_inputs.jl).

## Missing-data policy by input route

| Route | Policy |
| --- | --- |
| Discrete states | Explicit missing values or absent taxon records allow all states |
| align_traits_to_tree, one selected trait | Requires finite values; no missing/NaN |
| align_traits_to_tree, multiple traits | missing becomes NaN; at least one observation per row and column |
| Continuous DataFrame fit wrappers | Align by first taxon column; missing may become NaN for supported likelihoods |
| Direct vectors/matrices | Must obey the chosen model's missing-data contract |

The alignment helper's single-column rule is stricter than some fit wrappers. If a model accepts unobserved single-trait tips, construct and check the aligned numeric vector deliberately. Do not replace missing observations by zero. Rows with all continuous traits missing cannot be passed through the standard multivariate alignment helper.

Keep a discrete-only species for discrete fitting if useful, then prune the tree and sampled SIMMAP together to the continuous analysis set. Re-align continuous observations after pruning.

## DataFrame shortcuts

For single-trait fit wrappers, the first column is the taxon identifier. Use `trait_col=:height` to avoid ambiguity when there are multiple numeric columns. For multivariate fit wrappers, numeric columns after the first are candidate traits; explicit alignment gives clearer control when metadata are also numeric.

CSV is a convenient storage format. Read a file into a table, then pass aligned arrays to model functions:

```julia
using CSV, DataFrames
traits = CSV.read("traits.csv", DataFrame)
y = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:height])
```

Taxon identifiers should remain strings. Inspect inferred types after import, particularly when missing values are encoded as text.

## Transformations and reproducibility

Apply ln only to strictly positive measurements. Decide whether a reported quantity is raw height, ln(height), a standard deviation or a variance before import. Branch-length rescaling changes rate units. Save the species list and order, transformations, source filenames, state order, root settings and random seeds with every analysis.

See [trees](trees.md), [API conventions](api.md) and [common problems](troubleshooting.md).
