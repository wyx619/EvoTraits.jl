# 模型比较工作流

[English](workflows.md) · [文档目录](README_zh.md) · [连续模型](continuous_zh.md)

`fit_compare` 同时接受向量和矩阵，返回各模型拟合、AIC 表、最优模型名称和拟合结果。默认不进行 ASR。

```julia
result = fit_compare(tree, y, [:BM1, :OU1, :EB])
result = fit_compare(tree, X, [:mvBM1, :mvOU1]; reconstruct = true)
```

候选模型需要不同设置时使用 `ContinuousModelCandidate`：

```julia
candidates = [
    ContinuousModelCandidate(name=:plain, model=:BM1),
    ContinuousModelCandidate(name=:env, model=:EnvLin,
        kwargs=(environment=environment, quadrature_order=32)),
]
comparison = fit_compare(tree, y, candidates)
```

`fit_compare_estim` 明确请求单变量或多变量 ASR。`fit_compare_multivariate` 仍作为兼容名称保留，默认不做 ASR；需要时传入 `reconstruct=true`。环境模型不能进入 ASR 工作流。

AIC 只会在有限且成功的拟合中进行排序，但有限 logLik 不等于数值收敛。报告时应同时记录模型、观测、参数数量、logLik、AIC、收敛状态和诊断。

## 完整的 regime 模型比较

已有对齐的 `tree`、`y`、`sim` 时：

```julia
comparison = fit_compare(tree, y, [:BM1, :BMM, :OUM];
    edge_segments = sim.edge_segments,
    max_iterations = 1000, rel_tol = 1e-6)
for row in comparison.aic_table
    println(row)
end
for (name, fit) in comparison.fits
    println(name, " success=", fit.success,
            " converged=", fit.converged)
end
selected = comparison.best_fit
```

有 regime 依赖的候选必须提供映射，即使其他候选不依赖它。工作流可从枝段推导各 regime 时长；缺少必要映射会报输入错误，不会默认为单一 regime。

向量输入使用 `:BM1, :BMM, :OU1, :OUM, :EB` 等单变量符号；矩阵输入使用 `:mvBM1, :mvBMM, :mvOU1, :mvOUM, :mvOUMVA, :mvOUBM` 等受支持的多变量符号。矩阵派发不会自动重命名全部模型符号。

## 为候选分别设置参数

```julia
candidates = [
    ContinuousModelCandidate(name = :bm, model = :BM1),
    ContinuousModelCandidate(name = :oum, model = :OUM,
        kwargs = (max_iterations = 2000,
                  root_mean_mode = :regime,
                  root_cov_mode = :fixed)),
]
comparison = fit_compare(tree, y, candidates;
    edge_segments = sim.edge_segments)
```

候选自己的关键词覆盖工作流公共设置。名称用于标识输出，应保持唯一。导出结果时同时保留候选定义。

## 返回结构与选择规则

| 成员 | 内容 |
| --- | --- |
| `fits` | 按名称索引的拟合结果字典 |
| `aic_table` | 已排序的命名元组向量 |
| `best_name`、`best_fit` | 首选候选名称与结果 |
| `asr` | 默认 `nothing`，明确请求时才重建 |

AIC 表包含 `name, model, aic, delta_aic, loglik, nparams, success`，不会包含全部收敛诊断。选择规则基于成功且 AIC 有限的结果，因此作出科学解释前还应逐个检查收敛。

`aic_table(:first => fit1, :second => fit2)` 可直接比较已有结果。AIC 为 `2k - 2logLik`，在有效候选集合中越小越优。AICc 还依赖观测数量定义，无法推断时应提供 `nobs`。多变量实现可能按已观测性状单元而非物种数计数，这是一种实现约定，并不证明这些单元是独立生物学重复。

## 什么样的 AIC 比较才有效

保持响应变量、变换、保留物种及缺失模式一致。根假设与参数约束属于候选定义，必须报告。不同离散编码对应不同观测空间，二状态的离散 AIC 更小不能证明两类生活型比三类在生物学上更合理。

不同 SIMMAP 下的连续拟合以不同历史为条件，重复映射可评估历史抽样敏感性；AIC 均值或跨映射最小 AIC 均不是积分模型似然。Regime 模型获得 AIC 支持，也不等于证明离散状态对连续性状有因果效应。

## ASR 必须明确请求

`fit_compare_estim` 或 `reconstruct=true` 请求重建。含环境候选的比较不能请求 ASR，即使最终可能由其他候选胜出。需要重建时，可另行将受支持模型的结果及对应树、映射传给 [ASR 接口](reconstruction_zh.md)。

导出时可用 `DataFrame(comparison.aic_table)` 转换表格，再通过 `CSV.write` 写出。参见[可追溯拟合](fitting_zh.md)。
