# 随机性状映射

[English](simmap.md) · [文档目录](README_zh.md) · [离散模型](discrete_zh.md) · [多变量模型](multivariate_zh.md)

SIMMAP 在已拟合离散模型的条件下抽样分支历史。每张图是从拟合模型抽样得到的一条可能历史。

```julia
using Random
maps = simmap_corhmm(fit; nsim = 100, rng = MersenneTwister(2026))
maps.success
sim = maps.collapsed_samples[1]
summary_simmap(fit.tree, sim)
events = transition_events_table(fit.tree, sim)
```

这里的 `fit` 是成功的 `fit_corhmm` 结果。`maps.samples` 保留隐藏状态，`collapsed_samples` 将其映射到观测状态。映射必须与生成它的树配对。`write_simmap(tree, sim; file="history.simmap")` 写入一张历史，`read_simmap` 返回解析后的树与映射。

```julia
subset = keep_tip(fit.tree, sim, ["A", "B", "C"])
subset.tree
subset.simmap
```

下游拟合应使用返回的树和映射。旧名称 `keep_tip_simmap`、`drop_tip_simmap` 仍可用。`transition_times` 提取事件，`transition_rates_through_time` 汇总单位来源状态分支暴露量的转换速率。

## SIMMAP 为什么是离散与连续模型的桥梁

离散拟合得到转换速率与节点状态分布；SIMMAP 还抽样**状态在每条枝的什么位置发生转换**。OUM 类模型利用这些枝段，依次传播不同 regime 下的连续过程。节点边际概率不包含枝内转换位置。

`SimmapSample` 保存 `edge_segments`、`mapped_edge`、`nstates`、`state_labels` 和 `root_state`。每条边的枝段时长之和必须等于该边枝长；状态编号对应映射标签的顺序。导入或变换后，建议先执行 `summary_simmap(tree, sim)`。

## 逐个保存，不在内存保留 100 张映射

`simmap_corhmm(fit; nsim=100)` 会将所有映射返回到内存。需要增量保存时，保持同一个随机数生成器，每次仅抽一张：

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

不要每轮重建同一种子的 RNG，否则会重复相同随机流。

若要将 100 张映射保存在同一个文本文件中，可逐行写出 mapped Newick：

```julia
open("histories.simmap", "w") do io
    for sim in maps.collapsed_samples
        println(io, write_simmap(fit.tree, sim))
    end
end
```

单张映射的 `read_simmap("history.simmap")` 返回含 `tree`、`simmap`、`newick` 的命名元组；多张映射返回这些命名元组的向量。读取默认自动识别格式。跨软件交换时须注意状态顺序和枝段方向，往返读取后应核对枝长总和。

## 拟合与 ASR 必须使用同一历史

```julia
parsed = read_simmap("history.simmap")
tree2, sim2 = parsed.tree, parsed.simmap
# 拟合前按 tree2.tip_labels 重新构造性状行顺序。
fit2 = fit_oum(tree2, y2, sim2)
nodes2 = estim_node(tree2, y2, fit2, sim2)
```

这里的 `y2` 必须对应导入后的树。即使 tip 集合相同，也不能混用原树的边数组与重新解析的树。

隐藏速率模型的 `samples` 使用隐藏状态，`collapsed_samples` 使用观测状态，应按连续 regime 假设选择层级。多张映射可用于评估历史抽样敏感性。

参见[离散到 SIMMAP 示例](examples/02_discrete_simmap.jl)及[模型比较](workflows_zh.md)。
