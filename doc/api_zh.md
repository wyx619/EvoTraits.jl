# API 约定

[English](api.md) · [文档目录](README_zh.md) · [入门指南](quickstart_zh.md)

## 核心输入

`CompactTree` 保存拓扑、枝长、遍历顺序和 tip 标签。向量长度为 `tree.ntips`，矩阵行数为 `tree.ntips`。多变量缺失观测使用 `NaN`，每行和每列至少应有一个观测值。离散缺失状态可写为 `?`、`NA`、`missing` 或 `NaN`；多态状态使用 `state1&state2`。

## 命名与固定似然

拟合函数使用 `fit_...`；固定参数似然使用 `..._LL`，例如 `ou1_LL`、`mvoum_LL` 和 `corhmm_LL`。拟合结果通常保留 `loglik`、`aic`、`nparams`、`success` 和 `converged`。只有观测和假设一致时，`aic` 才可比较。

## 根与 regime

根设置是模型参数，不是输出标签。OU1/OUM 默认 `root_mean_mode=:regime`、`root_cov_mode=:fixed`，相应接口支持 `:stationary` 和 `:random`。如果提供 SIMMAP，regime 名称来自其状态标签。regime 历史必须与性状使用同一棵树和同一分支对应关系。

## 线程

使用 `-t N` 启动 Julia。EvoTraits 加载时将 BLAS 设为单线程；`set_engine_blas_threads!(n)` 可改变整个进程设置。Julia 层线程与 BLAS 线程是两类资源。

## 兼容性

`read_tree` 和 `load_tree` 的返回值不同。`transition_events_table` 是当前表格接口，`transition_events_DataFrame` 是兼容别名。`keep_tip(tree, simmap, tips)` 是统一裁剪入口，`keep_tip_simmap` 仍可用。

## 公开接口索引

| 任务 | 入口 | 主要返回约定 |
| --- | --- | --- |
| 读取计算树 | `load_tree`、`serialize_tree` | CompactTree |
| 裁树或映射树 | `keep_tip`、`drop_tip` | 树，或树与映射配对 |
| 离散拟合 | `fit_mk`、`fit_corhmm` | 速率、似然及诊断 |
| 离散重建 | `asr_corhmm` | 指定方法的状态重建 |
| 历史抽样 | `simmap_corhmm` | 隐藏及观测映射集合 |
| 单性状拟合 | `fit_bm1`、`fit_ou1`、`fit_oum*`、`fit_eb*` | 单一或多 regime 连续结果 |
| 多性状拟合 | `fit_mvbm1`、`fit_mvoum*`、`fit_mvoubm` | 矩阵参数拟合结果 |
| 候选比较 | `fit_compare`、`fit_compare_estim` | 命名结果、AIC 行、首选及可选 ASR |
| 连续重建 | `estim_node`、`estim_node_table` | ASR 对象或表格 |
| 环境或时间效应 | `fit_t_env`、`fit_shift`、`fit_mvshift` | 家族专用拟合结果 |

星号仅表示函数家族，不是可直接调用的函数名；具体名称见各模型指南。

## 查询签名与帮助

```julia
using EvoTraits
methods(fit_oum)
methods(fit_mvoum)
@doc fit_corhmm
@doc fit_t_env
@doc fit_mvshift
```

Julia REPL 可输入 `?fit_oum` 查看帮助。多重派发按参数类型选择方法。

## 固定参数似然与拟合结果

以 `_LL` 结尾的函数对给定过程参数求值；部分接口会按其约定剖面估计均值等参数。位置参数顺序依模型而异，例如 `oum_LL` 与 `mvoum_LL` 的标量、矩阵参数不能互换。跨实现验证前应先检查签名。

改变性状单位、枝长单位、根假设或 regime 顺序后，旧参数的统计含义也会改变。

## 结果类型差异

corHMM 的优化信息在 `fit.fit` 下，连续拟合通常直接提供 `converged`。多变量 OU 的扩散字段为 `Sigma`，多变量 BM 为 `sigma`；单变量多 regime 最适值为 `theta_regimes`，多变量为 `theta`。时间转换结果即使只有一个性状，也使用矩阵参数。

可用 `propertynames(result)` 查看字段，再结合相应指南解释；通用导出脚本不能假设所有模型字段完全相同。

## 输出与持久化

建模不强制使用 DataFrame，对齐数值数组即可。表格 helper 返回 DataFrame，便于查看及导出。本包没有自动统一结果归档或检查点格式。应用若使用 Julia 的 Serialization，应在应用环境自行声明该依赖，并保留匹配的软件版本。普通表格导出流程见[拟合与导出](fitting_zh.md)。

任务相关细节见[输入约定](inputs_zh.md)、[模型比较](workflows_zh.md)与[常见问题](troubleshooting_zh.md)。
