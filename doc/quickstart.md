# Getting started

[中文](quickstart_zh.md) · [Index](README.md) · [Home](../README.md)

This example creates its own tree. The vector is constructed in the tree's tip order.

```julia
using EvoTraits, Random

tree = mktempdir() do dir
    path = joinpath(dir, "tree.nwk")
    write(path, "((A:1,B:1):1,(C:1,D:1):1);")
    load_tree(path)
end
values = Dict("A" => 1.2, "B" => 1.5, "C" => 1.1, "D" => 1.8)
y = [values[label] for label in tree.tip_labels]

fit = fit_ou1(tree, y; max_iterations = 1000)
fit.success, fit.converged, fit.loglik, fit.aic
nodes = estim_node_table(tree, estim_node(tree, y, fit))

comparison = fit_compare(tree, y, [:BM1, :OU1, :EB])
comparison.best_name
comparison.aic_table
```

ASR is separate from comparison by default. Use `fit_compare_estim` or `fit_compare(...; reconstruct=true)` when needed; see [Workflows](workflows.md). Inspect convergence and parameter diagnostics before interpreting any result; see [API conventions](api.md).

For a taxon-keyed DataFrame, align before fitting:

```julia
using DataFrames
data = DataFrame(taxon = tree.tip_labels, trait = y)
y_aligned = align_traits_to_tree(tree, data; taxon_col = :taxon, trait_cols = [:trait])
```

CSV files can be loaded separately with `CSV.read(path, DataFrame)`. See [Trees](trees.md) for alignment checks.

## Install in an analysis environment

Use a dedicated Julia project so that package versions are recorded independently of other analyses:

```julia
using Pkg
Pkg.activate("evotraits-analysis")
Pkg.add(url = "https://github.com/wyx619/EvoTraits.jl")
```

If you already have a local checkout, use `Pkg.develop(path="/absolute/path/to/EvoTraits.jl")` instead. Start the analysis with `julia --project=evotraits-analysis -t 8`. A local development environment tracks source edits; preserve a release or commit identifier for a published analysis.

## Replace the demonstration data

The four-tip example above teaches the call sequence; such a tiny dataset is not a reliable basis for estimating a rich OU model. For your own analysis:

1. Read the tree with `load_tree`.
2. Read and clean the species and trait table.
3. Select the intended species subset and align to the tree.
4. Choose the response scale and candidate biological models.
5. Fit, inspect diagnostics, compare, and only then reconstruct.

For a mapped regime model, first fit a discrete model and generate a matching SIMMAP. The complete path is [input preparation](inputs.md) → [discrete models](discrete.md) → [SIMMAP](simmap.md) → [continuous models](continuous.md) → [reconstruction](reconstruction.md).

## Run a complete small example

From the EvoTraits checkout, with its dependencies instantiated:

```text
julia --project -t 8 doc/examples/01_inputs.jl
julia --project -t 8 doc/examples/02_discrete_simmap.jl
julia --project -t 8 doc/examples/03_continuous.jl
```

Each script creates its own data, prints relevant results and checks basic output contracts. Files used for round-trip tests are created in temporary directories. No private empirical dataset is needed. See the [example catalogue](examples/README.md) for multivariate and environmental examples.

## What should be saved?

Keep the cleaned input table, final tree, trait order, root assumptions, random seed, optimizer settings, model comparison and diagnostics. For regime fits also keep the exact mapped history. A node table alone is insufficient to reproduce the fit. See [fitting and export](fitting.md) before starting a large run.
