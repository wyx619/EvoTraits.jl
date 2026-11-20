# 拟合与诊断

[English](fitting.md) · [目录](README_zh.md) · [模型比较](workflows_zh.md)

## 推荐的拟合顺序

1. 检查物种名、性状顺序、缺失、时间单位与映射枝长。
2. 在最终分析数据集上拟合简单基线，通常为 BM1。
3. 明确根假设与矩阵约束，再拟合生物学候选模型。
4. 检查数值有效性、优化停止状态、边界和可识别性。
5. 在相同观测下比较可用模型，再对支持的首选模型重建。
6. 保存设置和结果后，再扩展到多张映射或多棵树。

[小型可运行示例](examples/README_zh.md)展示接口用法。

## 正确理解状态标记

`success` 是各模型定义的有效性标记；`converged` 描述优化停止状态，corHMM 中位于 `fit.fit.converged`。

采用两阶段优化的单变量拟合中，只有最终获选起点的粗搜索和精调都报告收敛时，才返回 `converged=true`。拟合结果有效但未达到此条件时，仍可返回 `success=true`、`converged=false`。收敛不保证全局最优。

固定参数似然返回给定参数下的 logLik，不含优化状态。排查实现差异时，应在完全相同参数下比较固定参数 logLik。

## 预算不是统一的迭代次数

基于 NLopt 的预算可能限制函数评估次数，L-BFGS 则使用自己的迭代和调用计数。应记录 `f_calls`、`iterations`、可用的停止信息和实测时间。

触及预算时，增加对应预算，在接口支持时更换初始点，并比较最终似然。

## 线程与内存

可用 `julia --project -t 8 analysis.jl` 启动，再检查 `Threads.nthreads()`。EvoTraits 加载时将 BLAS 设为单线程；加载后可用 `set_engine_blas_threads!(n)` 修改这一进程级设置。

corHMM 还需设置 `Ntrials` 和 `Nthreads`，仅启动八线程进程不会请求八次拟合。部分连续模型会内部并行候选拟合，无需用户线程参数。

独立候选需要私有工作区，更多线程可能提高峰值内存。在内部并行拟合之外再启动外层并行任务，可能使资源超额订阅。先测一张映射或一棵树，再决定批处理策略。

## 最小结果导出

以下针对单变量连续拟合结果 `fit`：

```julia
using CSV, DataFrames
mkpath("results")
summary = DataFrame(
    model = [string(fit.model)],
    success = [fit.success],
    converged = [fit.converged],
    loglik = [fit.loglik],
    aic = [fit.aic],
    nparams = [fit.nparams],
)
CSV.write("results/fit_summary.csv", summary)
```

还应导出参数、regime 与性状标签、根假设、诊断及精确分析输入。可使用 Julia 的 `versioninfo()` 记录环境，并保留分析环境的 Project/Manifest。使用本包进行分析不要求序列化拟合对象。

## 解释前检查

核对参数尺度是否符合性状及时间单位、吸引半衰期是否可估计、regime 是否有足够映射时长，以及协方差是否接近奇异。OU 均值设计的 `mean_diagnostics` 必须独立于优化状态检查。

接口支持时，使用不同起点复核关键拟合。无法解决的数值或可识别性问题应如实报告。参见[常见问题](troubleshooting_zh.md)。
