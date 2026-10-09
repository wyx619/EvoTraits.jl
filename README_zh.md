# EvoTraits

**在 Julia 中高效拟合与分析性状演化模型。**

[English](README.md)

EvoTraits 是面向离散与连续性状的 Julia 系统发育比较分析软件包，以似然方法为基础，在统一计算框架中提供演化模型拟合、随机性状映射、祖先状态重建与模型比较。

项目支持谱系间演化异质性、多性状演化动态、对演化预测变量的适应响应、演化过程的时间变化，以及演化速率对环境历史的依赖等研究方向。其主要特点是将计算效率、明确的统计假设与结构化的分析结果相结合。

## 核心特点

**离散与连续推断的整合。** 离散状态模型和 SIMMAP 历史为依赖演化状态的连续性状模型提供基础。共享的树结构与一致的结果接口支持各分析阶段之间的衔接，并保持节点及分支的对应关系。

**高效的系统发育似然计算。** 单变量和多变量模型共用树计算基建，通过剪枝递推计算似然，无需构造观测维度的全树稠密协方差矩阵。支持的多起点拟合路径利用 Julia 线程并行优化候选解。

**灵活的演化假设表达。** 模型允许演化状态之间具有不同的最适值、吸引强度和扩散速率，支持适应最适值对演化预测变量的依赖，以及早期与晚期演化过程的差异。环境历史进一步支持检验演化速率的变化。候选模型可指定独立设定，在同一工作流中比较不同参数化方案。

**支持可重复分析的结构化结果。** 拟合对象保存参数估计、似然、信息准则和收敛状态；祖先重建表保留节点身份与时间坐标，为祖先性状变化和时间趋势等后续分析提供一致的数据基础。

EvoTraits 通过 Julia 原生实现与可复用的计算基建，为成熟的 R 系统发育比较方法生态提供补充。项目开发借鉴 corHMM、OUwie、mvMORPH、mvSLOUCH 和 RPANDA 的统计方法与参考实现，正常运行不调用这些软件包，也不声明与其全部功能完全等价。

## 研究应用

| 生物学问题 | 对应能力 |
|---|---|
| 性状演化更符合随机漂变、早期爆发，还是向最适值回归？ | BM、EB 与 OU 模型 |
| 不同演化状态是否具有不同的最适值、吸引强度或扩散速率？ | OUM、OUMV、OUMA、OUMVA 及其多变量版本 |
| 另一个不断演化的性状，是否参与塑造响应性状的适应最适值？ | `fit_mvoubm` 多变量 OU-BM 适应模型 |
| 早期与晚期的演化过程是否不同？ | 单变量与多变量时间转换模型 |
| 性状演化速率是否随环境历史变化？ | EnvExp、EnvLin 及其与 BM1、EB 的比较 |
| 离散性状如何转换，是否存在隐藏的速率差异？ | Mk 与 corHMM 风格模型 |
| 哪些演化历史与现生物种状态相容？ | SIMMAP 随机性状映射 |
| 祖先性状是什么，随时间呈现怎样的变化？ | 支持模型的节点/分支重建与滑动窗口汇总 |

## 近期进展

### 演化速率的环境依赖

`fit_t_env` 通过指数（EnvExp）或线性（EnvLin）速率函数，将连续性状的扩散速率与环境历史联系起来。与 BM1、EB 比较，可区分环境依赖的速率假设、恒定速率演化与单向时间趋势。模型沿用 RPANDA 环境依赖 BM 的统计定义，估计的是演化速率变化，而不是环境对性状均值或适应最适值的影响。

环境历史可用分段线性曲线表示，也可显式进行固定有效自由度的三次样条平滑。`fit_compare` 工作流支持比较上述四种模型，无需同时进行祖先重建。共享积分区间避免重复存储与计算；EnvLin 对这两类已知曲线采用解析积分。

在完整 InfTemp 环境历史、87-tip 树和模拟性状的本地验证中，缓存优化使四模型拟合约加速 12.5 倍，16 点 EnvExp 缓存的保留大小减少 93.7%。这些是排除编译和平滑耗时的单案例测量，不是通用性能保证；独立积分与似然对照确认数值一致性及模型排序保持稳定。

### 多变量适应响应模型

`fit_mvoubm` 将性状划分为 OU 响应变量与 BM 预测变量，联合估计预测变量对响应最适值的贡献，以及吸引和扩散参数。模型允许截距随演化状态（regime）改变，从而在映射的演化历史背景下评估适应响应关系。

该形式支持对适应响应假设进行模型化评估，但估计的性状关系本身不构成因果证据。

### 多变量 OU 的灵活参数化

对角吸引矩阵允许不同性状具有各自的吸引速率；支持的非对称结构则允许表达性状间的方向性耦合。具有正实特征值的一般参数化将非对称吸引扩展至两个以上的性状。

对角吸引矩阵并不意味着性状独立演化，因为扩散矩阵仍可包含性状间相关性。

支持的多变量 OU 拟合在结构化起点之外提供可选的随机多起点优化与可复现种子。命名候选可指定不同拟合设定，以明确比较同一模型族中的不同结构。

### 演化过程的时间异质性

`fit_shift` 与 `fit_mvshift` 新增 13 种时间转换模型，组合 BM、OU 和 EB，并包括速率转换 SR。它们提供统一结果、固定参数似然求值、可复用时间映射、可选树高归一化及显式曲率诊断。

这些模型沿用参考实现的两时期协方差构造，将各时期的随机贡献相加，其统计定义不同于一般的连续切换随机微分方程。时间边界由用户指定，拟合估计过程参数，不自动识别未知的转换时间。

### 数值稳定性与验证

近期更新修正了固定 theta 求值，以及 profile-root OU 似然中的参数相关归一化项。BMM/SR 对完整和部分缺失观测采用稳定的协方差或平方根递推，避免近奇异扩散矩阵导致的灾难性消减。

针对上述条件，项目采用专门回归测试与独立小树协方差计算评估数值一致性。这类验证不意味着拟合解已达到全局最优。

## 安装与示例

在 Julia 中从仓库安装：

```julia
using Pkg
Pkg.add(url="https://github.com/wyx619/EvoTraits.jl")
```

需要并行拟合时，以多线程启动 Julia，例如 `julia -t 8`。与 Julia 层并行配合时，限制 BLAS 线程数：

```julia
using EvoTraits
set_engine_blas_threads!(1)
```

### 拟合、比较与重建

假设 `traits.csv` 包含与树 tip 匹配的 `taxon` 和 `lnH` 列：

```julia
using EvoTraits, CSV, DataFrames

tree = load_tree("tree.tre")
traits = CSV.read("traits.csv", DataFrame)
y = align_traits_to_tree(tree, traits;
    taxon_col=:taxon, trait_cols=[:lnH])

analysis = fit_compare_estim(tree, y, [:BM1, :OU1, :EB])
analysis.aic_table
analysis.best_name
nodes = estim_node_table(tree, analysis.asr)
```

`load_tree` 直接返回可以进入计算的树结构；也保留 `read_tree` → `serialize_tree` 的显式两步路径。

### 比较环境与时间依赖的速率假设

假设 `environment.csv` 包含 `age`（距今年龄，单位与树枝长一致）和 `temperature` 列。环境历史必须覆盖现在至树根年龄，以下示例还要求超过十个不同年龄：

```julia
history = CSV.read("environment.csv", DataFrame)
environment = smooth_env(history.age, history.temperature; df=10, scale=true)

candidates = [
    ContinuousModelCandidate(name=:BM1, model=:BM1),
    ContinuousModelCandidate(name=:EB, model=:EB),
    ContinuousModelCandidate(name=:EnvExp, model=:EnvExp,
        kwargs=(environment=environment,)),
    ContinuousModelCandidate(name=:EnvLin, model=:EnvLin,
        kwargs=(environment=environment,)),
]
comparison = fit_compare(tree, y, candidates; max_iterations=2000)
comparison.aic_table
comparison.best_fit.converged
```

不需要平滑时，可用 `EnvCurve(history.age, history.temperature; scale=true)` 作分段线性插值，但须先处理重复年龄。`smooth_env` 支持重复年龄，要求显式指定 df；应在模型比较前独立确定该值。`fit_compare` 仅返回拟合结果与信息准则，不执行 ASR；环境模型目前不支持进入 `fit_compare_estim`。

### 结合 regime 拟合多变量演化

这里的 `mapped.simmap` 应描述同一棵树上的历史，`traits.csv` 还需包含 `lnVD` 列：

```julia
sim = read_simmap("mapped.simmap")
X = align_traits_to_tree(tree, traits;
    taxon_col=:taxon, trait_cols=[:lnH, :lnVD])

fit = fit_mvoumva(tree, X, sim;
    trait_names=["lnH", "lnVD"], A_decomp=:diagonal)
```

若把第一列性状作为 OU 响应，第二列作为 BM 预测变量：

```julia
adaptation = fit_mvoubm(tree, X, sim; kY=1)
```

### 检验时间转换假设

假设过程在距今 20 个时间单位处转换：

```julia
shift = fit_mvshift(tree, X; model=:OUBMi, age=20.0)
shift.loglik
shift.aic
shift.converged
```

边界必须位于树的时间范围内。单个性状使用 `fit_shift`。时间转换模型目前不接入祖先重建。

## 模型范围

| 模型族 | 单变量 | 多变量 |
|---|---|---|
| 布朗运动 | BM1、BMM | mvBM1、mvBMM |
| 早期爆发 | EB、EBM | mvEB |
| 环境依赖 BM | EnvExp、EnvLin | 暂未实现 |
| Ornstein-Uhlenbeck | OU1、OUM、OUMV、OUMA、OUMVA | mvOU1、mvOUM、mvOUMV、mvOUMA、mvOUMVA |
| 适应性 OU-BM | — | mvOUBM |
| 时间转换 | SR、OUBM/OUBMi、BMOU/BMOUi、EBOU/EBOUi、OUEB/OUEBi、EBBM/EBBMi、BMEB/BMEBi | 对应的 13 种模型 |

离散分析涵盖 Mk、corHMM 风格的观测/隐藏速率模型、模糊 tip 状态、自定义速率结构及祖先重建。SIMMAP 连接离散历史与依赖演化状态的连续模型；软件包同时提供信息准则、模拟和跨实现树节点映射。

## 适用范围与统计注意事项

- 模型支持取决于树的拓扑、观测和参数化。连续内核要求二歧树，许多路径要求超距树。当前多变量验证限制为 4,000 tips；树读取容量与模型拟合容量属于不同的约束。
- 缺失数据支持因模型而异。务必核对 taxon 对齐和 SIMMAP 分支对应关系。
- 根处理与矩阵约束属于统计假设。比较 AIC 时应保持数据和约定一致；不同观测响应之间不能直接比较 AIC。
- 有限 logLik 不代表收敛。检查 `converged` 和可用的终止诊断，必要时使用多个起点。
- 时间转换模型本轮面向超距树，不支持观测误差或 ASR；支持的 A 结构有意收敛于既定范围，而不是复制参考实现的全部参数空间。
- 环境依赖 BM 当前支持二歧超距树上的单性状分析，包括性状缺失观测，不支持观测误差或 ASR。平滑采用固定 df，尚无自动 df 选择或 GCV。已收敛的拟合仍可能趋向速率边界；对陡峭或短时集中的环境变化，应检查积分精度。

## 文档与验证

通过 Julia 帮助模式查询单个接口，例如 `?fit_t_env`、`?smooth_env`、`?fit_mvoubm` 或 `?fit_mvshift`。源码围绕离散模型、连续模型、重建和模拟组织；`test/` 保存单元和回归测试。

本地 `validation/` 与 `reference/` 是不随 Git 分发的开发资料，不是正常使用的依赖。RCall 用于本地跨实现验证，包本身不通过 R 执行计算。
