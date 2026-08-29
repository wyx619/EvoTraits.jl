# EvoTraits

> 用 Julia 高效拟合并分析性状演化模型

EvoTraits 是一个 Julia 原生的系统发育比较分析框架。它的目标不是把零散方法简单堆在一起，而是把树表示、模型拟合、祖先重建、SIMMAP 感知分析和模型比较放到同一个统一引擎中。

## 为什么是 EvoTraits

传统比较分析流程往往高度碎片化：

- 离散模型一个包
- 连续模型一个包
- SIMMAP 一个包
- 祖先重建一个包
- 数据对齐、结果整理再靠脚本补

EvoTraits 想解决的正是这种割裂。它现在的基本思路是：

- 一套核心树对象：`CompactTree`
- 一套连续 / 离散分析框架
- 一套结构化结果对象
- 尽可能走树上的原生计算路径，而不是依赖外部拼装

## 当前能力范围

### 连续性状模型

#### 单变量

- 布朗运动：`fit_bm1`, `fit_bmm`
- 早爆发 / ACDC：`fit_eb`, `fit_ebm`
- OU 系列：`fit_ou1`, `fit_oum`, `fit_oumv`, `fit_ouma`, `fit_oumva`

#### 多变量

- 布朗运动：`fit_mvbm1`, `fit_mvbmm`
- 早爆发：`fit_mveb`
- OU 系列：`fit_mvou1`, `fit_mvoum`, `fit_mvoumv`, `fit_mvouma`, `fit_mvoumva`

### 离散性状模型

- Mk：`fit_mk`, `asr_mk`
- corHMM 风格隐状态速率模型：`fit_corhmm`, `asr_corhmm`

### SIMMAP 工作流

- 从离散拟合结果采样和生成 SIMMAP
- SIMMAP 读写、汇总和下游分析
- SIMMAP 感知的连续模型拟合与重建

主要入口包括：

- `simmap_corhmm`
- `simmap_sample`
- `simmap_samples`
- `read_simmap`
- `write_simmap`
- `summary_simmap`

### 祖先重建

- 节点重建：`estim_node`, `estim_node_table`
- 分支重建：`estim_branch_for_simmap`, `estim_branch_table`
- 时间分箱汇总：`summarize_node_estimates_by_time`

### 模型比较工作流

- 单变量工作流：`fit_compare_estim`
- 多变量工作流：`fit_compare_multivariate`
- 共享比较辅助：`aic`, `aicc`, `aic_table`, `best_model`

### 模拟与树参考映射

- 树模拟
- 连续性状模拟
- 跨语言树映射：`PhyloRef`

## 项目定位

EvoTraits 不是对 R 包的简单封装，而是在 Julia 中重建一个更统一的比较分析框架。它持续参考成熟生态中的统计逻辑和实践标准，例如：

- `ape`
- `phytools`
- `geiger`
- `castor`
- `OUwie`
- `mvMORPH`
- `Rphylopars`
- `corHMM`

目标不是模仿界面，而是用更清楚的内部结构承载成熟方法。

## 快速开始

### 1. 读取树并拟合单变量 OU 模型

```julia
using EvoTraits
using CSV, DataFrames

tree = serialize_tree(read_tree("tree.nwk"))
traits = CSV.read("trait.csv", DataFrame)

fit = fit_ou1(tree, traits)
asr = estim_node(tree, traits, fit)
tbl = estim_node_table(tree, asr)
```

### 2. 在 SIMMAP 上拟合多变量 OU 模型

```julia
using EvoTraits
using CSV, DataFrames

sim = read_simmap("mapped_tree.simmap")
traits = CSV.read("traits.csv", DataFrame)

fit = fit_mvoum(sim.tree, traits, sim.simmap)
asr = estim_node(sim.tree, traits, fit, sim.simmap)
tbl = estim_node_table(sim.tree, asr, fit)
```

### 3. 拟合隐状态速率离散模型并采样 SIMMAP

```julia
fit = fit_corhmm(tree, states; model = :ARD, rate_cat = 2)
asr = asr_corhmm(fit)
maps = simmap_corhmm(fit; nsim = 100)
```

### 4. 运行模型比较工作流

```julia
res = fit_compare_estim(tree, y, [:BM1, :OU1, :EB])
res.best_name
res.aic_table
```

## 数据约定

- 主树对象：`CompactTree`
- 物种顺序以树的 tips 为对齐主轴
- `DataFrame` 性状输入默认第一列为 taxon labels
- 多变量连续数据允许局部缺失
- SIMMAP 分段长度必须严格加和为枝长

## 结果风格

EvoTraits 尽量返回结构化结果对象，而不是临时元组。常见字段包括：

- `model`
- `success`
- `loglik`
- `aic`
- `nparams`
- `converged`
- `iterations`
- `f_calls`

模型特异参数会直接放在结果对象里，例如：

- `sigma2`
- `alpha`
- `beta`
- `theta`
- `A`
- `Sigma`
- `trait_names`
- `regime_names`

## 仓库结构

```text
src/
  continuous/   连续模型、工作流、祖先重建
  discrete/     Mk、corHMM、SIMMAP
  phyloref/     跨语言树映射
  simulate/     树与性状模拟
  io.jl         树结构与 Newick I/O
  criteria.jl   公共信息准则
```

```text
test/           单元测试与回归测试
validation/     真实数据与对齐验证
reference/      参考实现与研究材料
docs/           notebook 与补充文档
```

## 安装

在仓库根目录使用：

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

从其他 Julia 环境直接添加：

```julia
import Pkg
Pkg.add(url = "https://github.com/<your-org-or-user>/EvoTraits.jl")
```

## 测试

运行全部测试：

```julia
import Pkg
Pkg.test()
```

运行指定子集：

```powershell
julia --project=. test/run_subset.jl discrete/corHMM/corhmm
julia --project=. test/run_subset.jl continuous/univariate/ou_family
julia --project=. test/run_subset.jl continuous/multivariate/ou_family
julia --project=. test/run_subset.jl workflows
```

## 当前状态

EvoTraits 现在已经可以覆盖一条比较完整的比较分析链条：

- 离散性状建模
- 连续性状建模
- SIMMAP 感知分析
- 祖先重建
- 多变量 OU 系列分析
- 模型比较工作流

它仍然是持续演化中的研究型软件，但当前代码库已经不再是零散实验集合，而是一个有明确结构、有测试、有验证路径的统一框架。

## 环境要求

- Julia `1.10` 到 `1.12`
- 普通使用不依赖 R
- 部分验证流程可能依赖独立的 R 环境

## 引用

如果你在研究中使用 EvoTraits，请同时引用本仓库以及你实际使用到的系统发育比较方法文献。
