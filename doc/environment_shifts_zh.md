# 环境与时间转换模型

[English](environment_shifts.md) · [文档目录](README_zh.md) · [工作流](workflows_zh.md)

## 环境速率模型

`fit_t_env` 是单变量模型，描述扩散速率如何随环境曲线变化。`EnvExp` 和 `EnvLin` 描述的是速率依赖，不是性状均值或 OU 最适值的因果效应。

```julia
environment = EnvCurve([0.0, 10.0], [15.0, 20.0])
fit = fit_t_env(tree, y, environment; model=:EnvExp,
    quadrature_order=16, max_iterations=2000)
```

可用 `smooth_env(history.age, history.value; df=10, scale=true)` 显式平滑。环境历史必须覆盖树的时间范围；对于尖锐脉冲应检查求积阶数和断点。使用 `fit_compare` 与 BM1、EB 比较；环境模型没有 ASR 接口。

若 `E(t)` 为输入环境值，EnvExp 的速率为 `sigma2 * exp(beta * E(t))`；EnvLin 为 `sigma2 * (1 - E(t)) + beta * E(t)`，并要求速率为正。因此 EnvLin 的 `beta` 是环境值为 1 时的速率，不是加性斜率。环境缩放会改变参数含义。EnvExp 的 `beta=0` 和 EnvLin 的 `beta=sigma2` 均退化为 BM。

已知 tip 标准误可传给 `fit_t_env`；`error=missing` 会估计一个共同 nuisance 标准差。

## 时间转换模型

`fit_shift` 和 `fit_mvshift` 检验指定的早期/晚期过程转换时间。模型包括 SR 以及 BM、OU、EB 的组合。

```julia
shift = fit_mvshift(tree, X; model=:OUBMi, age=20.0)
shift.loglik, shift.aic, shift.converged
```

转换时间由分析者给定，函数不会自动搜索。时间单位必须与树枝长一致。当前时间转换模型不提供 ASR。

## 准备环境历史

环境时间是**距今年龄**，不是根以来的时间。树高 100 Ma 时，曲线必须完整覆盖 0–100 Ma；程序不会外推缺失的古老环境。

```julia
ages = collect(0.0:1.0:100.0)
temperature = 18 .+ 3 .* sin.(ages ./ 12)
linear = EnvCurve(ages, temperature)  # 分段线性
smooth = smooth_env(ages, temperature; df = 10, scale = false)
smooth(25.0)
```

`EnvCurve` 要求至少两个不同的有限非负年龄与有限环境值，并排序年龄。`smooth_env` 要求至少六个不同年龄、有限正权重，以及 `2 < df < 不同年龄数`。有效自由度控制曲线灵活程度，不是原始观测数或插值点数。两个接口都默认 `scale=false`；开启后将环境范围缩放至 [0,1]，常数环境不能如此缩放。

尽可能独立于响应性状选择平滑程度，并开展敏感性检查。尝试大量 df 后按最小模型 AIC 选择，并不会自动计入平滑选择的复杂度。

## 系数含义与正值约束

```text
EnvExp: v(t) = sigma2 * exp(beta * E(t))
EnvLin: v(t) = sigma2 * (1 - E(t)) + beta * E(t)
```

EnvExp 的正 beta 表示 E 增大时扩散加快，负 beta 表示减慢。EnvLin 的 sigma2 与 beta 分别是 E=0、E=1 的速率。若直接输入温度，它们对应 0°C 和 1°C；缩放到 [0,1] 后则对应环境两端。使用的环境历史内，EnvLin 速率必须保持正值。

这些模型解释**扩散速率**变化，不解释性状均值的方向变化。除非主动把温度导数作为 E 输入，否则模型不拟合温度变化速度。

## 求积与观测误差

`quadrature_order=16` 是默认值，不保证适合每种曲线。强指数效应或短暂脉冲应比较 8、16、32、64 点，并用 `quadrature_breakpoints` 标记相关年龄。在将差异解释为生物学效应前，应同时检查逐枝方差与固定参数 logLik。

`error=nothing` 不添加公共观测误差项；`error=missing` 或 `NaN` 估计公共 nuisance 标准差。已知标准误可通过向量、物种字典或相应 DataFrame 列输入，注意是标准差而非方差。提供已知标准误时也会启用 nuisance 估计，末端方差添加已知 SE² 与 nuisance²。结果中的 `nuisance` 与 `nuisance_variance` 分别位于标准差、方差尺度。

应保存 `environment`、`edge_variances`、`quadrature_order`、`quadrature_breakpoints`、参数与收敛诊断。参见[可执行环境示例](examples/05_environment_shifts.jl)。

## 时间转换模型详解

| 家族 | 早期过程 → 晚期过程 |
| --- | --- |
| SR | 布朗速率转换 |
| OUBM / BMOU | OU → BM / BM → OU |
| EBOU / OUEB | EB → OU / OU → EB |
| EBBM / BMEB | EB → BM / BM → EB |

除 SR 外各家族还有 `i` 版本，如 `:OUBMi`，允许两个时期使用独立扩散协方差参数。边界只能选择一种输入方式：`age`、兼容的两时期 `mapped_edge`，或 `shift_precalc` 的预计算结果。年龄须严格位于零与树高之间；树发生变化后不能复用旧预计算。

```julia
height = maximum(tree.dist_from_root)
pre = shift_precalc(tree; age = height / 2, ntraits = size(X, 2))
shift = fit_mvshift(tree, X; model = :OUBMi, precalc = pre,
    A_decomp = :diagonal, Sigma_decomp = :diagonal,
    max_iterations = 1000)
```

`fit_shift` 接受向量，返回同一矩阵式结果家族的单性状结果。`Sigma_decomp=:cholesky` 为完整正定扩散；`:diagonal` 为对角扩散；`:equaldiagonal` 的对角方差相等且非对角为零；`:equal` 的对角方差相等但允许相关。Equal 不代表所有元素相等。

默认 `scale_height=false` 保留原时间单位；设为 true 时在归一化树高下拟合，解释前须检查 `source_height`、`height`、`time_scale` 和 `age`。不应未经换算就把归一化速率当作原时间单位速率。

EB 的 `beta_bounds` 默认为 (-2,2)，范围外输入可映射为零效应；应核查 `input_beta`、`beta` 与诊断，不能把平坦参数方向直接解释为 BM 的强证据。

`diagnose_shift(tree, traits, fit)` 不重新拟合，而是进行数值曲率诊断。有限差分 Hessian 需要大量似然调用，可能较耗时；它不提供通用参数置信区间或全局最优保证。此处时间转换模型要求超度量树，不支持观测误差建模，也不提供 ASR。
