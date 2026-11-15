# 多变量模型

[English](multivariate.md) · [文档目录](README_zh.md) · [SIMMAP](simmap_zh.md)

多变量矩阵每行对应一个树 tip，每列对应一个性状。可用模型包括 `fit_mvbm1`、`fit_mvbmm`、`fit_mveb`、`fit_mvou1`、`fit_mvoum`、`fit_mvoumv`、`fit_mvouma`、`fit_mvoumva` 和 `fit_mvoubm`。

```julia
# tree 和 sim 必须来自同一张 SIMMAP
X = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:lnH, :lnVD])
fit = fit_mvoumva(tree, X, sim;
    trait_names = ["lnH", "lnVD"], A_decomp = :diagonal)
fit.loglik, fit.aic, fit.converged
```

OUM 家族使用分支 regime 映射。OUMV 允许扩散随 regime 变化，OUMA 允许吸引随 regime 变化，OUMVA 允许两者变化。吸引矩阵为对角阵时，若扩散仍相关，性状**并非完全独立**。OU1/OUM 根选项见[连续模型](continuous_zh.md)；多变量 OUMV/OUMA/OUMVA 使用 regime 根均值和固定根协方差。

`fit_mvoubm` 将 OU 响应性状的最适值建模为 BM 预测变量的函数；`kY` 指定前几列为响应。需要仔细检查参数可识别性、收敛和缺失输入。候选模型比较见[工作流](workflows_zh.md)。

## 输入维度与部分缺失

输入为 `ntips × p` 矩阵，每列一个性状。性状名称、参数矩阵和后续 ASR 必须采用相同列顺序。部分缺失用 `NaN`，每个保留 tip 和每个性状列至少有一个观测；无穷值不合法。

```julia
X = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:lnH, :lnVD])
@assert size(X, 1) == tree.ntips
@assert size(X, 2) == 2
```

多变量 DataFrame 拟合包装器从第一列 taxon 后选择数值列。需要精确控制性状列时，先显式对齐，再传入矩阵。不能假定所有拟合包装器都接受 `trait_cols`。

## A、Sigma 与 theta

OU 均值过程为 `dY = A * (theta_r - Y) dt + noise`。`A` 描述吸引及性状间均值耦合，`Sigma` 是扩散协方差速率。

| 模型 | A | Sigma | theta |
| --- | --- | --- | --- |
| mvOU1 | 共享 | 共享 | 一个向量 |
| mvOUM | 共享 | 共享 | 各 regime 独立 |
| mvOUMV | 共享 | 各 regime 独立 | 各 regime 独立 |
| mvOUMA | 各 regime 独立 | 共享 | 各 regime 独立 |
| mvOUMVA | 各 regime 独立 | 各 regime 独立 | 各 regime 独立 |

OU 结果的 `A` 和 `Sigma` 为三维分块数组：共享参数只有一块，独立参数每个 regime 一块。`theta` 的**行为 regime、列为性状**。BM 结果使用小写 `sigma`，不能直接照搬 OU 字段名。

```julia
fit.A[:, :, 1]
fit.Sigma[:, :, 1]
fit.theta
fit.trait_names
fit.regime_names
```

## 吸引矩阵参数化

| `A_decomp` | 含义 |
| --- | --- |
| `:cholesky` | 对称正定吸引矩阵 |
| `:diagonal` | 正值对角吸引速率 |
| `:schur` | 支持的两性状 Schur 参数化 |
| `:decomposable_positive` | 正实特征谱的可分解一般矩阵 |

对角 `A` 移除了均值耦合，但若 `Sigma` 仍有非对角项，扩散仍然相关，因此不能仅据此称性状独立演化。更改单位会改变矩阵元素；标准化有助于改善数值条件，但解释参数时必须交代尺度。

## 不同函数的优化选项并不完全相同

`fit_mvoum` 常用控制包括 `optimization=:SBPLX_L_BFGS`、`max_iterations=300`、`polish_iterations=100`、`rel_tol=1e-5`、`n_starts=0`、`random_seed=nothing`。`n_starts` 表示在结构化初始候选之外额外添加的随机候选数，零不代表没有初始化策略。并非整个 OU 家族都暴露全部参数；例如未在签名提供 `polish_iterations` 的函数不能直接传入它。

OU1/OUM 的根组合见[单变量指南](continuous_zh.md)。OUMV/OUMA/OUMVA 提供 `:regime + :fixed`。进行受控比较时须保持根设置、参数化与性状尺度一致。

## OU 响应与 BM 预测性状：mvOUBM

`fit_mvoubm` 将前 `kY` 列作为 OU 响应 Y，其余列作为 BM 预测性状 X。响应最适值通过 `B` 依赖于预测性状，并含 regime 截距。因此列顺序是模型定义的一部分。

```julia
# 第一列为响应，第二列为 BM 预测性状。
coupled = fit_mvoubm(tree, X, sim;
    kY = 1, A_decomp = :diagonal, max_iterations = 300,
    trait_names = ["response", "predictor"])
coupled.B
coupled.vX0
coupled.theta
```

该函数有专门的根处理与参数化，不继承普通 OU 的 `root_mean_mode` / `root_cov_mode` 控制。其吸引矩阵选项还包含 `:decomposable_real`、`:upper_triangular`、`:lower_triangular`，不能未经核对就用于普通 mvOUM。同样不能假定 mvOUBM 支持普通 mvOUM 的多起点关键词。

参见[可执行多变量示例](examples/04_multivariate.jl)与[拟合诊断](fitting_zh.md)。
