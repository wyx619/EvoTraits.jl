# Trees and alignment

[中文](trees_zh.md) · [Index](README.md) · [Home](../README.md)

## Reading and writing

`load_tree(path)` reads a Newick tree directly into `CompactTree`. `read_tree(path)` returns a NewickTree node; call `serialize_tree` to obtain a CompactTree. `write_tree(path, tree)` writes a tree, and `to_newick(tree)` returns Newick text.

```julia
using EvoTraits
tree = load_tree("tree.nwk")
tree.ntips
tree.tip_labels
write_tree("copy.nwk", tree)
```

## Alignment

A trait vector or matrix must follow `tree.tip_labels`; rows are tips and matrix columns are traits. For a DataFrame with a taxon column:

```julia
using DataFrames
data = DataFrame(taxon = tree.tip_labels, height = collect(1.0:tree.ntips))
y = align_traits_to_tree(tree, data; taxon_col = :taxon, trait_cols = [:height])
```

The alignment validates duplicate labels and missing tree tips. Extra data rows are ignored. Missing-value behavior depends on trait dimension; see [API conventions](api.md).

## Pruning and identity

`keep_tip(tree, tips)` and `drop_tip(tree, tips)` return pruned trees. Their three-argument counterparts `keep_tip(tree, simmap, tips)` and `drop_tip(tree, simmap, tips)` return a `(tree, simmap)` pair while preserving mapped branch histories. See [SIMMAP](simmap.md). Node and edge identifiers are tree-specific; use `build_phyloref`, `phyloref_node_table`, and `phyloref_edge_table` when mapping identities to R/ape order.

## Choosing a reader

| Operation | Return value | Typical use |
| --- | --- | --- |
| `load_tree(path)` | `CompactTree` | Direct input for evolutionary models |
| `read_tree(path)` | `NewickTree.Node` | Work with the node-based representation |
| `serialize_tree(node)` | `CompactTree` | Convert an already parsed node tree |
| `write_tree(path, tree)` | Writes Newick | Export topology and branch lengths |

The file must exist and contain a terminated Newick tree, including the final semicolon. Unicode paths are supported. The node-based reader retries parsing after removing internal support/bootstrap labels if the initial parse fails. These functions are single-tree interfaces; do not pass a collection of bootstrap trees as though it were one tree.

Branch lengths define the time unit. A tree in Ma produces attraction rates in Ma⁻¹ and diffusion rates in trait-unit²/Ma. Dates are not inferred from taxon names. Check the model's requirements before fitting: the OU, EB and shift workflows described here require an ultrametric tree, and supported pruning workflows expect bifurcating topology.

## Tip positions are not node identifiers

`tree.tip_labels[i]` names row `i` of a trait vector or matrix. The node identifier of that tip is `tree.tip_ids[i]`. Internal node IDs, tip positions and ape/R node numbers must not be substituted for each other.

Useful fields are:

| Field | Meaning |
| --- | --- |
| `ntips`, `nnodes`, `nedges` | Sizes of the tree |
| `root` | Root node ID |
| `parent_of_edge[e]`, `child_of_edge[e]` | Endpoints of edge `e` |
| `edge_length[e]` | Length of that edge |
| `parent_of_node[v]` | Parent of node `v` |
| `dist_from_root[v]` | Time from the root to node `v` |
| `tip_labels`, `tip_ids` | Tip labels and their node IDs |

For an ultrametric tree, let `height = maximum(tree.dist_from_root)`; the age before present of node `v` is `height - tree.dist_from_root[v]`. Build a reference with `build_phyloref(tree)` when exporting node identities. Use `phyloref_node_table` and `phyloref_edge_table` to inspect the mapping.

## Aligning a table explicitly

```julia
using DataFrames
# This table is deliberately not in tree order.
traits = DataFrame(taxon = reverse(tree.tip_labels),
                   height = collect(1.0:tree.ntips))
y = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:height])
@assert length(y) == tree.ntips
```

Explicit taxon matching rejects duplicate names and absent tree tips; unrelated extra rows are ignored. Without `taxon_col`, the helper assumes the rows are already aligned and requires exactly `tree.ntips` rows. Always select trait columns explicitly when a table also contains IDs, coordinates or categorical codes.

For a partial overlap, select taxa first, prune the tree, and then align the observations to the **new** tree order. For mapped trees, prune tree and SIMMAP together; pruning the tree alone invalidates the old edge indexing. See [Inputs](inputs.md) for missingness and [SIMMAP](simmap.md) for paired pruning.
