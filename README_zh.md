# EvoTraits

**基于似然的离散与连续性状演化系统发育比较分析。**

[English](README.md)

EvoTraits 是面向离散与连续性状的 Julia 系统发育比较分析软件包，以似然方法为基础，在统一计算框架中提供演化模型拟合、随机性状映射、祖先状态重建与模型比较。

项目支持谱系间演化异质性、性状协同演化、对演化预测变量的适应响应、演化过程的时间变化，以及演化速率对环境历史的依赖等研究方向，将计算效率、明确的统计假设与结构化的分析结果相结合。

## 核心特点

**离散与连续推断的整合。** 离散状态模型和 SIMMAP 历史为依赖演化状态的连续性状模型提供基础。共享的树结构与一致的结果接口支持各分析阶段之间的衔接，并保持节点及分支的对应关系。

**高效的系统发育似然计算。** 单变量和多变量模型通过树上的剪枝递推计算似然，无需构造覆盖全部观测的稠密协方差矩阵。支持多起点的拟合流程利用 Julia 线程并行优化候选解。

**灵活的演化假设表达。** 模型允许演化状态之间具有不同的最适值、吸引强度和扩散速率，支持适应最适值对演化预测变量的依赖，以及早期与晚期演化过程的差异。环境历史进一步支持检验演化速率的变化。候选模型可指定独立设定，在同一工作流中比较不同参数化方案。

**支持可重复分析的结构化结果。** 拟合对象保存参数估计、似然、信息准则和收敛状态，OU 均值诊断进一步区分参数可识别性与优化器收敛。祖先重建表保留节点身份与时间坐标，为祖先性状变化和时间趋势分析提供一致的数据基础。

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

## 建模能力

### 演化速率的环境依赖

`fit_t_env` 通过指数（EnvExp）或线性（EnvLin）速率函数，将连续性状的扩散速率与环境历史联系起来。与 BM1、EB 比较，可评估数据对环境依赖、恒定速率和指数时间趋势三类假设的支持程度。这些模型描述演化速率变化，不估计环境对性状均值或适应最适值的影响。

环境历史可用分段线性曲线表示，也可采用指定有效自由度的三次平滑样条。`fit_compare` 工作流支持比较上述四种模型，无需同时进行祖先重建。

环境模型可纳入已知的物种性状标准误，并额外估计公共末端方差。没有标准误时，可用 `error=missing` 估计该分量；已知标准误可按树 tip 顺序提供向量，或按物种名提供字典。默认 `error=nothing` 保留不含该分量的模型。

### 多变量适应响应模型

`fit_mvoubm` 将性状划分为 OU 响应变量与 BM 预测变量，联合估计预测变量对响应最适值的贡献、吸引强度与扩散参数。模型允许截距随映射的演化状态（regime）改变，支持评估性状协同适应假设。

### 多变量 OU 的灵活参数化

对角吸引矩阵允许不同性状具有各自的吸引速率；支持的非对称结构则允许表达性状间的方向性耦合。具有正实特征值的一般参数化将非对称吸引扩展至两个以上的性状。

对角吸引矩阵并不意味着性状独立演化，因为扩散矩阵仍可包含性状间相关性。

支持的多变量 OU 拟合在结构化起点之外提供可选的随机起点与可复现种子，不同参数化方案可作为命名候选模型进行比较。

### 演化过程的时间异质性

`fit_shift` 与 `fit_mvshift` 提供 13 种时间转换模型，组合 BM、OU 和 EB，并包括速率转换模型 SR。在统一拟合与比较框架下，这些模型支持检验早期与晚期演化过程的差异。

模型在用户指定的时间边界两侧组合各时期的协方差贡献。拟合估计该假设下的演化参数，不自动寻找未知的转换时间；可选的树高归一化与曲率诊断支持模型评估。

### 拟合与祖先重建的一致性

OU 模型拟合与祖先重建采用一致的根处理假设，各演化状态的最适值通过广义最小二乘估计，并提供可识别性诊断。节点和分支重建保留与原树的对应关系，时间滑动窗口汇总进一步支持研究祖先性状的历史变化。

单元测试与回归测试通过独立小树计算和参考案例，检验似然、模型关系及祖先重建的一致性，并对缺失观测和数值条件较差的协方差结构设置专门验证。

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

这里的 `mapped.simmap` 包含映射的演化历史，`traits.csv` 还需包含 `lnVD` 列。使用映射同时返回的树，可保持分支对应关系：

```julia
mapped = read_simmap("mapped.simmap")
tree, sim = mapped.tree, mapped.simmap
X = align_traits_to_tree(tree, traits;
    taxon_col=:taxon, trait_cols=[:lnH, :lnVD])

fit = fit_mvoumva(tree, X, sim;
    trait_names=["lnH", "lnVD"], A_decomp=:diagonal)
```

OU1、OUM 及其多 regime 变体（包括多变量版本）默认以根 regime 的最适值作为根均值（`root_mean_mode=:regime`），根过程方差为零（`root_cov_mode=:fixed`）。OU1／OUM 及其多变量版本另支持归一化均值设计（`:stationary`）和平稳根协方差（`:random`），可任意组合；OUMV／OUMA／OUMVA 及其多变量版本的公开接口仅接受 `:regime + :fixed`。模型不单独估计根均值；解释各 regime 最适值前，应检查 `fit.mean_diagnostics`。

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

连续模型要求二歧树，许多模型还要求超距树，缺失数据支持因模型而异。分析时应核对物种与 SIMMAP 对应关系，在相同观测和一致假设下比较 AIC，并检查收敛状态、边界估计及参数可识别性。时间转换和环境模型要求超距树，目前不支持 ASR；时间转换模型不支持观测误差，环境模型则支持物种性状标准误和公共末端方差。环境模型仅支持单性状，环境历史变化较快时须关注平滑设定与积分精度。性状间的统计关联本身不构成因果证据。

## 文档与验证

通过 Julia 帮助模式查询单个接口，例如 `?fit_t_env`、`?smooth_env`、`?fit_mvoubm` 或 `?fit_mvshift`。源码围绕离散模型、连续模型、重建和模拟组织；`test/` 保存单元和回归测试。

