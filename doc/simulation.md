# Simulation

[中文](simulation_zh.md) · [Index](README.md) · [Trees](trees.md)

## Simulate a tree and traits

```julia
using EvoTraits, Random

rng = MersenneTwister(42)
tree = sim_yule_tree(30; birth_rate = 1.0, tree_height = 10.0, rng = rng)
Sigma = [0.8 0.2; 0.2 0.6]
X = sim_mvbm1(tree, Sigma; rng = rng)
size(X) == (tree.ntips, 2)
```

`sim_yule_tree` and `sim_birth_death_tree` return `CompactTree`; their `*_simtree` counterparts return `SimulatedTree`. `sim_ultrametric_newick` returns Newick text. Supply an explicit RNG to reproduce simulations.

`sim_mvbm1`, `sim_mvbmm`, `sim_mvou1`, `sim_mvoum`, `sim_mvoumva` and `sim_mveb` generate trait matrices. Regime-aware functions need branch segments and matching parameter dimensions. `sim_trait` offers a model-symbol entry point, and `sim_mvbm1_dataset` returns a tree, labels and simulated data together.

## Reproducibility and units

Use an explicit `MersenneTwister(seed)` and keep the tree-generation settings. The example's height of 10 is in the chosen simulation time unit; it becomes Ma only if that unit is deliberately assigned. Reusing one RNG advances its stream, while constructing a new RNG with the same seed repeats the stream.

`sim_mvbm1` expects a diffusion **covariance** matrix. Positive diagonal values alone do not make a matrix positive definite. Trait columns in the output follow parameter order, and rows follow the returned tree's tip order.

## Simulating mapped OU traits

After preparing branch segments on a tree:

```julia
A = [0.9 0.1; 0.1 1.0]
Sigma = [0.7 0.15; 0.15 0.5]
theta_regimes = [[0.0, 0.2], [0.8, -0.1]]
X = sim_mvoum(tree, sim.edge_segments, A, Sigma, theta_regimes;
    rng = MersenneTwister(503))
@assert size(X) == (tree.ntips, 2)
```

Here `theta_regimes` is a vector of trait vectors in regime order. Do not confuse that input form with the fitted result's regime-by-trait theta matrix. Segment state IDs must refer to the same regime order as the parameter vectors.

[common.jl](examples/common.jl) constructs a synthetic mapped history for small interface demonstrations. For a posterior history, use [corHMM followed by SIMMAP](simmap.md).

## Choosing a simulation function

| Goal | Function family |
| --- | --- |
| Pure-birth / birth-death trees | `sim_yule_tree`, `sim_birth_death_tree` |
| Node-oriented simulated tree object | Corresponding `*_simtree` functions |
| Ultrametric Newick text | `sim_ultrametric_newick` |
| Shared or mapped BM traits | `sim_mvbm1`, `sim_mvbmm` |
| Shared or mapped OU traits | `sim_mvou1`, `sim_mvoum`, `sim_mvoumva` |
| Early-burst traits | `sim_mveb` |
| Packaged BM simulation dataset | `sim_mvbm1_dataset` |

Simulation and fitting have different option sets. Check the function-specific help for available simulators and root parameters. Fixed seeds make examples repeatable.
