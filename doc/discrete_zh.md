# 离散模型

[English](discrete.md) · [文档目录](README_zh.md) · [SIMMAP](simmap_zh.md)

`fit_mk` 拟合 Mk 模型；`fit_corhmm` 支持观测状态、隐藏速率、模糊观测、自定义速率结构及祖先状态重建。

```julia
using EvoTraits, Random
tree = load_tree("tree.nwk")
states = Dict("A" => "herb", "B" => "herb&tree",
              "C" => "tree", "D" => "?")
fit = fit_corhmm(tree, states; model = :ARD,
    state_order = ["herb", "tree"], rate_cat = 1,
    node_states = :marginal, Ntrials = 4, Nthreads = 4,
    max_iterations = 2000, rng = MersenneTwister(42))
fit.success
fit.transition_matrix
fit.asr
```

示例要求树含这四个 tip。`herb&tree` 是模糊或多态观测，不会新增一个马尔可夫状态；`?` 允许所有状态。`state_order` 固定 Q 行列和 ASR 列的含义。`node_states` 可选 `:marginal`、`:joint`、`:scaled` 或 `:none`；也可在拟合后调用 `asr_corhmm(fit; mode=:marginal)`。默认 `root_prior=:yang`；根先验与根边际后验是不同量。

`Ntrials` 和 `Nthreads` 控制拟合试验和并行工作线程，启动 Julia 时需提供足够线程。进行[性状历史抽样](simmap_zh.md)前应检查拟合诊断和收敛。

## 模型选择与结果

`model` 选择 `:ER`、`:SYM`、`:ARD`、`:SUEDE`、`:SRD` 等速率约束；有序模型的生物学含义取决于 `state_order`。`rate_cat` 是隐藏速率类别数。标准约束无法表达研究假设时，可使用 `rate_mat` 指定自定义参数索引结构。

`fit.transition_matrix` 是拟合 Q 矩阵；`fit.index_matrix` 描述速率参数共享方式，`fit.rates` 保存拟合速率。`fit.root_prior_probs` 是似然计算所用根先验，`fit.asr` 是所请求的重建结果。`fit.diagnostics` 包含缺失和多态 tip 数量、固定节点等设置。除 `fit.success` 外还应检查 `fit.fit.converged`。

`tip_fog`、`fixed_nodes` 和 `lewis_asc_bias` 是专门选项；使用前应通过 Julia 帮助模式核对输入假设。多列 character 分析中，状态组合及是否允许同时转换必须与生物学假设一致。

## 速率模型与状态顺序

观测状态数为 `k`、仅一个速率类别时：

| 模型 | 自由速率数 | 约束 |
| --- | --- | --- |
| `:ER` | 1 | 所有允许转换共用速率 |
| `:SYM` | k(k−1)/2 | 每对状态正反向速率相同 |
| `:ARD` | k(k−1) | 每个有向转换单独估计 |
| `:SUEDE` | 有序链为 2 | 正向共享一个、反向共享另一个速率 |
| `:SRD` | 2(k−1) | 相邻状态各方向分别估计 |

有序链由 `state_order` 定义。例如 SRD 的 `["herb", "shrub", "tree"]` 不允许 herb↔tree 直接转换。颠倒顺序保留邻接关系，只改变参数对应标签。

## 多态、缺失物种与隐藏速率

`"herb&tree"` 在 herb、tree 两列的观测似然均为 1，其他列为 0。它不是第四种状态。缺失 tip 允许所有状态；状态映射中没有记录的树上物种按缺失处理。

`rate_cat > 1` 时，潜在状态空间由观测状态与隐藏速率类别组成。隐藏类别之间转换时保持观测状态不变。Q 和隐藏 ASR 对应扩展状态空间；应通过 `hidden_to_observed` 及观测层 ASR 汇总隐藏维度。

## 主要拟合参数

| 关键词 | 默认值 | 含义 |
| --- | --- | --- |
| `model` | `:ARD` | 速率共享模型 |
| `rate_cat` | `1` | 隐藏速率类别数 |
| `state_order` | `nothing` | 显式指定观测标签顺序 |
| `rate_mat` | `nothing` | 自定义速率**索引**矩阵 |
| `root_prior` | `:yang` | 似然计算的根权重 |
| `node_states` | `:marginal` | ASR 方法；可用 `:none` 跳过 |
| `Ntrials`、`Nthreads` | `1`、`1` | 拟合次数与请求的并行工作数 |
| `max_iterations` | `1_000_000` | 优化预算 |
| `lower_bound`、`upper_bound` | `1e-9`、`100.0` | 速率边界 |
| `rel_tol` | `sqrt(eps(Float64))` | 相对停止容差 |
| `rng` | 默认随机数生成器 | 初始点的可重复性 |

以八线程启动 Julia 不会改变默认的 `Ntrials=1`。需要八次并行拟合时，显式设置 `Ntrials=8, Nthreads=8`。结合多起点似然及参数边界判断结果。

根权重支持 `:yang`、`:flat`、`:maddfitz` 或数值向量。`:yang` 根据 Q 推导权重，`:flat` 为等权，`:maddfitz` 使用似然相关加权；它们都不是根节点重建概率。

## 提取根后验与 Q

```julia
Q = fit.transition_matrix
# Q[i,j] 是从状态 i 到状态 j 的转换速率；每行和为零。
fit.index_matrix
fit.rates
fit.root_prior_probs

asr = asr_corhmm(fit; mode = :marginal)
i = findfirst(==(Int(fit.tree.root)), asr.node_ids)
root_posterior = asr.observed_likelihoods[i, :]
collect(zip(fit.observed_labels, root_posterior))
```

通过节点编号定位根，不假定第一行就是根。Marginal 对其他节点状态积分；joint 给出整树的最佳联合状态配置。

## 自定义假设与多列 character

`rate_mat` 填写参数编号，不填写 Q 的数值。正整数表示自由速率，相同编号表示共享参数；非对角的零、负数、缺失及非有限值禁止转换。对角项忽略，建议使用从 1 开始的连续编号。

两状态非对称模型示例：

```julia
index = [0 1; 2 0]
custom = fit_corhmm(tree, states;
    state_order = ["herb", "tree"], rate_mat = index,
    Ntrials = 4, Nthreads = 4, max_iterations = 2000)
```

多列 character 以状态组合建模，默认只允许一次改变一个 character。独立模型将不同背景下相应速率设为相等；依赖模型释放这些等式约束。同时改变两个 character 需要明确的自定义结构，没有通用的 `dual=true` 拟合参数。

`tip_fog` 表示观测不确定性，`fixed_nodes` 固定指定祖先状态，`lewis_asc_bias` 对仅抽取变异性状进行条件校正。这些都会改变统计模型。完整输入约定可通过 `?fit_corhmm` 查看。

## Mk 接口另有约定

`fit_mk` 使用 `rate_model`，不是 corHMM 的 `model`。显式状态数接口接受 `tip_states` 或 `tip_priors`，后者可表达观测似然。不能将 corHMM 的所有参数直接复制到 Mk 调用。参见[可运行离散示例](examples/02_discrete_simmap.jl)。

## 观测不确定性与固定节点

| 选项 | 可用形式 | 含义 |
| --- | --- | --- |
| `tip_fog=nothing` | 默认 | 不添加观测错误模型 |
| `tip_fog=0.01` | [0,0.5) 标量 | 固定公共 fog 值 |
| `tip_fog=[0.01,0.02]` | 每个观测状态一个概率 | 固定状态特异 fog |
| `tip_fog=:estimate` | 符号 | 估计一个共享 fog 参数 |
| `tip_fog=[1,2]` | 正整数分组编号 | 按组估计 fog 参数 |
| `fixed_nodes=false` | 默认 | 不固定内部节点 |
| `fixed_nodes=Dict(node_id => "herb")` | 内部节点编号或标签为键 | 固定指定祖先观测状态 |
| `fixed_nodes=true` | 布尔值 | 从内部节点标签读取可识别状态 |
| `lewis_asc_bias=true` | 布尔值 | 对仅抽取变异 character 的条件似然校正 |

概率向量与整数分组向量含义不同，应保留元素类型。Fog 修改观测似然，不添加新马尔可夫状态；完全缺失仍不提供状态信息。多于两个状态时，除每项合法外，还必须为允许的观测状态保留正权重。

固定节点的键依赖具体树，不能将 tip 当作内部固定节点。只有外部依据充分时才使用约束。

## 两个二元 character 的明确索引结构

组合状态顺序固定为 `["00", "10", "01", "11"]` 时，以下分别表示独立、依赖且仅单 character 改变、依赖且允许 dual：

```julia
independent = [0 1 3 0;
               2 0 0 3;
               4 0 0 1;
               0 4 2 0]
dependent = [0 1 2 0;
             3 0 0 4;
             5 0 0 6;
             0 7 8 0]
dual = [0 1 2 9;
        3 0 10 4;
        5 11 0 6;
        12 7 8 0]
```

三者分别有 4、8、12 个速率参数。构造采用同一标签的组合 character 映射，将各矩阵传给 `rate_mat`，并保持相同 `state_order`。比较必须使用相同物种和观测编码。只有一个分量缺失时，列出兼容组合；例如第一分量未知、第二分量为 0，可编码为 `"00&10"`。两个分量都缺失时允许全部四态。

独立与 dual 假设需通过 `rate_mat` 明确指定。
