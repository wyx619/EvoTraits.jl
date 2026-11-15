# 模拟

[English](simulation.md) · [文档目录](README_zh.md) · [树操作](trees_zh.md)

## 模拟树和性状

```julia
using EvoTraits, Random

rng = MersenneTwister(42)
tree = sim_yule_tree(30; birth_rate = 1.0, tree_height = 10.0, rng = rng)
Sigma = [0.8 0.2; 0.2 0.6]
X = sim_mvbm1(tree, Sigma; rng = rng)
size(X) == (tree.ntips, 2)
```

`sim_yule_tree` 和 `sim_birth_death_tree` 返回 `CompactTree`；对应的 `*_simtree` 函数返回 `SimulatedTree`。`sim_ultrametric_newick` 返回 Newick 文本。显式传入 RNG 有利于复现模拟。

`sim_mvbm1`、`sim_mvbmm`、`sim_mvou1`、`sim_mvoum`、`sim_mvoumva` 和 `sim_mveb` 生成性状矩阵。regime 模型需要枝段与对应维度的参数。`sim_trait` 提供模型符号入口，`sim_mvbm1_dataset` 一并返回树、标签和模拟数据。

## 可重复性与单位

使用显式 `MersenneTwister(seed)` 并保留树生成设置。示例树高 10 采用所选模拟时间单位，只有主动定义为 Ma 才具有百万年含义。复用同一个 RNG 会推进随机流，重复创建同一种子的 RNG 则会重复该流。

`sim_mvbm1` 接收扩散**协方差**矩阵。仅对角元素为正不能保证矩阵正定。输出列按参数性状顺序，行按返回树的 tip 顺序。

## 模拟映射 OU 性状

在树上准备好枝段后：

```julia
A = [0.9 0.1; 0.1 1.0]
Sigma = [0.7 0.15; 0.15 0.5]
theta_regimes = [[0.0, 0.2], [0.8, -0.1]]
X = sim_mvoum(tree, sim.edge_segments, A, Sigma, theta_regimes;
    rng = MersenneTwister(503))
@assert size(X) == (tree.ntips, 2)
```

这里 `theta_regimes` 是按 regime 排列的性状向量集合，不同于拟合结果"行为 regime、列为性状"的 theta 矩阵。枝段状态编号必须与参数向量的 regime 顺序一致。

[common.jl](examples/common.jl) 为小型接口演示构造合成映射。需要后验历史时，应使用 [corHMM 后接 SIMMAP](simmap_zh.md)。

## 如何选择模拟函数

| 目标 | 函数家族 |
| --- | --- |
| 纯出生或出生死亡树 | `sim_yule_tree`、`sim_birth_death_tree` |
| 节点式模拟树对象 | 对应的 `*_simtree` |
| 超度量 Newick 文本 | `sim_ultrametric_newick` |
| 共享或映射 BM 性状 | `sim_mvbm1`、`sim_mvbmm` |
| 共享或映射 OU 性状 | `sim_mvou1`、`sim_mvoum`、`sim_mvoumva` |
| 早期爆发性状 | `sim_mveb` |
| 打包 BM 模拟数据集 | `sim_mvbm1_dataset` |

模拟与拟合的选项不同。查看具体函数帮助，确认可用的模拟器与根参数。固定种子便于复现示例。
