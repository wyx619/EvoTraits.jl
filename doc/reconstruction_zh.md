# 祖先重建

[English](reconstruction.md) · [文档目录](README_zh.md) · [连续模型](continuous_zh.md)

连续性状使用 `estim_node(tree, trait, fit; ...)`，离散性状使用 `asr_corhmm(fit; mode=...)`。regime 连续重建必须使用拟合时相同的 `edge_segments` 或 SIMMAP。

```julia
fit = fit_ou1(tree, y)
node_asr = estim_node(tree, y, fit)
node_table = estim_node_table(tree, node_asr)

fit2 = fit_oum(tree, y, sim)
node_asr2 = estim_node(tree, y, fit2, sim)
branch_asr = estim_branch_for_simmap(tree, y, fit2, sim)
branch_table = estim_branch_table(tree, branch_asr)
```

表格保留内部节点或分支身份及时间坐标。 `summarize_node_estimates_by_time(table, width; w=step)` 生成滑动窗口汇总。汇总结果依赖拟合模型、根处理和 regime 历史，不能当作独立观测。

`width` 是窗口宽度，`w` 是步长；`w=0` 表示不重叠窗口。每个性状汇总表的第一行是全部节点的总体基线，并非时间零点的估计。窗口内分位数描述重建节点估计值的分布，并不是演化轨迹的置信区间。

离散拟合中，marginal ASR 给出各节点状态概率，joint ASR 给出一组联合状态，scaled 遵循缩放似然约定，none 跳过重建。下游分析需要不确定性时应保留 marginal 概率。

## 从拟合到节点表与时间窗口

```julia
fit = fit_oum(tree, y, sim; max_iterations = 1000)
node_asr = estim_node(tree, y, fit, sim)
node_tbl = estim_node_table(tree, node_asr; simmap = sim)
windows = summarize_node_estimates_by_time(node_tbl, 4.0; w = 2.0)
trait_windows = windows["trait"]
# 时间绘图时去掉首行的全树基准汇总。
time_windows = trait_windows[2:end, :]
```

映射必须与拟合时一致。表中的 `regime` 标签来自抽样历史，不是 corHMM 的边际后验。需要后验加权时，应将**同一棵树**的离散边际概率按节点编号合并，保留各状态的概率列。

## 字段、尺度与不确定性

| ASR 字段 | 含义 |
| --- | --- |
| `node_ids` | 内部节点编号 |
| `estimates` | 条件祖先估计 |
| `variances`、`se` | 给定拟合模型的条件不确定性 |
| `time_from_root` | 根以来的时间 |
| `time_before_present` | 节点距今年龄 |
| `tip_ids`、`tip_labels` | 结果关联的 tip 身份 |

节点输出不自动等于“tip 加全部内部节点”的合表。合并注释前应检查实际节点集合。多变量的 `estim_node_table(tree, asr, fit)` 使用拟合性状名生成 `estimate_lnH`、`variance_lnH`、`se_lnH` 等列。

若响应为 ln(H)，`exp(estimate)` 是反变换点估计，也是条件对数正态中位数。采用对数尺度高斯近似时，条件均值为 `exp(estimate + variance/2)`；两者都不自动等于无偏总体均值。ASR 方差不包含全部参数估计、根假设及 SIMMAP 抽样不确定性。

## 滑动时间窗口

`summarize_node_estimates_by_time(table, n; w=0)` 的位置参数是**窗口宽度**，不是每窗节点数量。时间单位与树一致。

| 设置或字段 | 含义 |
| --- | --- |
| `n=4.0, w=2.0` | 宽度 4，相邻窗口起点相差 2 |
| `n=4.0, w=0.0` | 宽度 4 的不重叠窗口 |
| 合法滑动步长 | `0 < w < n` |
| `n_nodes` | 窗内参与计算的有限估计数 |
| `mean`、`median` | 节点重建点估计的汇总 |
| `q2_5`、`q97_5` | 这些点估计之间的分位数 |

单变量表以 `"trait"` 为键返回性状汇总，多变量按性状列后缀返回；带 regime 列时另有 `"_regime_summary"`。首行 `start=end=0` 汇总全部节点，不能作为时间零点；绘图应去掉。空窗口没有有限估计。节点密度随年代变化，亲缘节点也并非独立观测，因此分位数带不是演化轨迹的置信区间。

时间曲线可使用窗口中点 `(start + end)/2`，并反转横轴使现在位于右侧。绘图平滑只是展示选择，不能替代进化模型。

## 枝段输出与边界

`estim_branch_for_simmap` 支持 OUM、OUMV、OUMA、OUMVA、EBM 等映射单变量模型；`estim_branch_table` 给出枝段的时间坐标和估计。同一条边含多个枝段时会出现多行，不能假定一边一行。

环境及时间转换拟合目前不提供这套 ASR。参见[环境与转换](environment_shifts_zh.md)、[常见问题](troubleshooting_zh.md)和[可执行 ASR 示例](examples/03_continuous.jl)。
