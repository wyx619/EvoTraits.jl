# 单变量连续模型

[English](continuous.md) · [文档目录](README_zh.md) · [工作流](workflows_zh.md)

| 模型 | 拟合函数 | 假设 |
| --- | --- | --- |
| BM1 | `fit_bm1` | 单一布朗扩散速率 |
| BMM | `fit_bmm` | regime 特异扩散速率 |
| OU1 | `fit_ou1` | 单一 OU 最适值 |
| OUM | `fit_oum` | regime 特异最适值 |
| OUMV / OUMA / OUMVA | `fit_oumv` / `fit_ouma` / `fit_oumva` | 扩散 / 吸引 / 两者随 regime 变化 |
| EB / EBM | `fit_eb` / `fit_ebm` | 随时间变化的速率 / regime 特异 EB |

输入为与树对齐的性状向量；regime 模型还需分支映射。固定参数似然函数如 `ou1_LL`、`oum_LL`。

```julia
fit = fit_ou1(tree, y; max_iterations = 1000)
fit.loglik, fit.aic, fit.converged
asr = estim_node(tree, y, fit)
```

OU1/OUM 默认根均值 `:regime`、根协方差 `:fixed`，也支持均值 `:stationary` 和协方差 `:random`。公开 OUMV/OUMA/OUMVA 接口使用 `:regime + :fixed`。根假设影响似然与重建；解释 regime 最适值前检查 `mean_diagnostics`。详见[重建](reconstruction_zh.md)和 [API 约定](api_zh.md)。

## OU 模型之间究竟改变了什么

单变量 OU 过程可写为：

```text
dY = alpha_r * (theta_r - Y) dt + sqrt(sigma2_r) dW
```

当前 regime `r` 由映射枝段决定。

| 模型 | 吸引速率 alpha | 扩散方差速率 sigma2 | 最适值 theta |
| --- | --- | --- | --- |
| OU1 | 全局共享 | 全局共享 | 一个 |
| OUM | 全局共享 | 全局共享 | 每个 regime 独立 |
| OUMV | 全局共享 | 每个 regime 独立 | 每个 regime 独立 |
| OUMA | 每个 regime 独立 | 全局共享 | 每个 regime 独立 |
| OUMVA | 每个 regime 独立 | 每个 regime 独立 | 每个 regime 独立 |

`sigma2` 是扩散方差速率，不是其平方根。`alpha` 的单位为时间倒数，半衰期为 `log(2)/alpha`。`theta` 与性状同单位，对数性状的 theta 可以为负。

## 拟合带映射的模型

```julia
oum = fit_oum(tree, y, sim;
    max_iterations = 1000, rel_tol = 1e-6,
    root_mean_mode = :regime, root_cov_mode = :fixed)
oumva = fit_oumva(tree, y, sim;
    max_iterations = 2000, rel_tol = 1e-6)
oumva.theta_regimes
oumva.alpha_regimes
oumva.sigma2
oumva.regime_names
```

`tree`、`y`、`sim` 须按 [SIMMAP 指南](simmap_zh.md)保持一致。单变量公开接口不提供多变量的 `n_starts` 或 `polish_iterations` 参数；粗搜索与精调由内部实现管理。增加 `max_iterations` 可增加搜索预算。

## 根均值与根协方差是两件事

| 模型 | 根均值选项 | 根协方差选项 |
| --- | --- | --- |
| OU1 / OUM | `:regime`、`:stationary` | `:fixed`、`:random` |
| OUMV / OUMA / OUMVA | `:regime` | `:fixed` |

`:regime` 将根均值设为根 regime 的最适值；`:stationary` 使用归一化的 regime 设计权重。`:fixed` 不添加根过程方差，`:random` 添加平稳根方差。当前不提供独立估计 theta0 的模式，不应继续使用已移除的 `:free_theta0` 等旧名称。

固定参数似然、拟合与 ASR 应采用相同根设定。更换根假设可能同时改变似然与祖先重建值。

## 如何读取结果

| 字段 | 含义 |
| --- | --- |
| `loglik`、`aic`、`nparams` | 似然、AIC、拟合参数数量 |
| `success`、`converged` | 结果有效性与优化收敛，应同时检查 |
| `iterations`、`f_calls` | 优化器计数，不能相互等同或直接代表耗时 |
| `theta_regimes` | 多 regime 结果的最适值 |
| `alpha`、`alpha_regimes` | 共享或各 regime 的吸引速率 |
| `sigma2` | 按模型返回共享速率或速率向量 |
| `mean_diagnostics` | 均值设计的秩与条件数 |
| `root_mean_mode`、`root_cov_mode` | 实际使用的根假设 |

单 regime 结果用 `theta` 保存最适值。`mean_diagnostics` 包含 `rank`、`ncoefficients`、`singular_values`、`identifiable`、`condition_number`。

当长枝的 `alpha * length` 很大时，过程会失去祖先状态记忆，相似重建值符合数学预期。结合最适值、枝长和可识别性诊断判断。

## BM 与 EB 的假设

BM1 为单一扩散速率；BMM 允许不同 regime 具有不同速率。EB 的扩散速率随根以来的时间指数变化；负时间系数表示演化过程中扩散减慢。EBM 进一步引入 regime 结构。

比较模型时必须保持观测一致。原始高度转为 ln(height) 会改变响应尺度和似然，不能把原始高度 AIC 与对数高度 AIC 当作仅更换模型来比较。参见[工作流](workflows_zh.md)、[拟合诊断](fitting_zh.md)及[映射 OU 示例](examples/03_continuous.jl)。
