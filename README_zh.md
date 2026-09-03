# EvoTraits

> **在 Julia 中高效拟合与分析性状演化模型**

EvoTraits 是一个面向系统发育比较分析的 Julia 框架。它把树读取、离散与连续性状建模、SIMMAP 历史、祖先状态重建和模型比较组织在同一套类型化工作流中。

这个项目把系统发育树视为计算骨架，而不只是一个输入文件：数据以树上的 tip 顺序为基准进行对齐，似然通过树剪枝算法计算，重建结果保留节点、分支和时间坐标，便于继续做父子节点变化、时间分箱和事件顺序分析。

## 为什么使用 EvoTraits

R 拥有成熟而丰富的比较方法生态。EvoTraits 面向的是另一类明确需求：希望在 Julia 内部完成大规模、可组合、可追溯的分析，而不必在多个包和脚本之间反复搬运对象。

- **Julia 原生：**普通使用不依赖 R 或 R 调用桥接。
- **统一树基建：**CompactTree 以数组形式保存拓扑、枝长、遍历顺序、标签和节点距离。
- **以树剪枝为核心：**连续和离散模型直接沿树结构计算；多变量模型只在性状维度上使用小型矩阵算子，不构造随物种数增长的观测级 dense VCV 矩阵。
- **结果结构清晰：**fit 结果保存 logLik、AIC、参数数量、收敛诊断、参数估计、trait 名称、regime 名称和模型元数据。
- **分析链条连贯：**数据对齐、模型拟合、AIC 比较、历史抽样、节点或分支重建和表格导出可以留在同一引擎中完成。
- **便于交叉验证：**稳定的树节点和分支映射，以及明确的时间字段，支持与 ape、corHMM、OUwie、mvMORPH、geiger、castor 等参考实现进行验证。

EvoTraits 不是把 R 包简单包一层。它借鉴成熟方法的统计语义，同时重新组织计算路径和结果接口，让 Julia 的类型系统、原生执行和可组合工作流真正参与分析。

## 功能范围

### 连续性状模型

单变量：

- Brownian motion：fit_bm1、fit_bmm
- Early burst：fit_eb、fit_ebm
- OU 系列：fit_ou1、fit_oum、fit_oumv、fit_ouma、fit_oumva

多变量：

- Brownian motion：fit_mvbm1、fit_mvbmm
- Early burst：fit_mveb
- OU 系列：fit_mvou1、fit_mvoum、fit_mvoumv、fit_mvouma、fit_mvoumva

带 regime 的模型可以接收 SIMMAP 分支片段。不同模型的 optimum、attraction matrix 和 diffusion matrix 会保存在专门的结果类型中，而不是散落在无名数组里。

### 离散性状模型

- Mk 似然与重建：fit_mk、asr_mk
- corHMM 风格的观测状态和 hidden-rate 模型：fit_corhmm、asr_corhmm
- 速率索引与 Q 矩阵转换：rateindex、rates_to_q、qfromindex 等
- 通过 tip-prior 矩阵处理缺失和模糊状态
- 在模型支持的路径中使用 likelihood、flat、Yang、Maddison-FitzJohn、stationary 或显式概率等根先验模式

### SIMMAP

- simmap_sample、simmap_samples：通用 Mk/SIMMAP 抽样
- simmap_corhmm：从 corHMM fit 结果抽样
- read_simmap、write_simmap：SIMMAP 文件读写
- describe_simmap、summary_simmap、transition_times 和转换表：历史检查与汇总
- drop_tip_simmap、keep_tip_simmap：树和映射的筛选

每个 SimmapSample 同时保存有序的逐分支片段和 regime 持续时间。进入连续 regime 模型之前，会检查所有片段长度是否与对应枝长一致。

### 祖先状态重建

- 连续节点重建：estim_node、estim_node_table
- SIMMAP 感知的连续分支重建：estim_branch_for_simmap、estim_branch_table
- 离散 marginal 或 joint 重建：asr_mk、asr_corhmm
- 按时间窗口汇总节点估计：summarize_node_estimates_by_time

节点和分支结果同时提供 time_from_root 与 time_before_present。因此可以直接分析父子边上的变化、按时间汇总估计，或把连续重建结果与离散 regime 信息连接起来。

### 模型比较与模拟

- 信息准则：aic、aicc、aic_table、delta_aic、best_model
- 完整比较工作流：fit_compare_estim、fit_compare_multivariate
- Yule 和 birth-death 树模拟
- 单变量和多变量 BM、EB、OU 系列性状模拟
- 通过 PhyloRef 完成跨实现的节点和分支引用

## 快速开始

### 读取树并拟合 OU1

通用 Newick 解析使用 read_tree，之后显式序列化。对于大树工作流，load_tree 是直接返回 CompactTree 的高吞吐路径。

~~~julia
using EvoTraits
using CSV, DataFrames

tree = load_tree("tree.nwk")
traits = CSV.read("trait.csv", DataFrame)
lnH = align_traits_to_tree(tree, traits; taxon_col = :taxon, trait_cols = [:lnH])

fit = fit_ou1(tree, lnH; trait_name = "lnH")
asr = estim_node(tree, lnH, fit)
node_table = estim_node_table(tree, asr)
~~~

也可以显式分两步：

~~~julia
parsed = read_tree("tree.nwk")
tree = serialize_tree(parsed)
~~~

### 在 SIMMAP 上拟合多变量 OU

~~~julia
using EvoTraits
using CSV, DataFrames

tree = load_tree("tree.nwk")
sim = read_simmap("mapped_tree.simmap")
traits = CSV.read("traits.csv", DataFrame)
X = align_traits_to_tree(tree, traits; taxon_col = :taxon, trait_cols = [:lnH, :lnVD])

fit = fit_mvoum(tree, X, sim; trait_names = ["lnH", "lnVD"])
asr = estim_node(tree, X, fit, sim)
node_table = estim_node_table(tree, asr, fit; simmap = sim)
~~~

对于两个性状、需要允许非对称 attraction matrix 的模型，可以显式指定：

~~~julia
fit = fit_mvoumva(
    tree,
    X,
    sim;
    trait_names = ["lnH", "lnVD"],
    A_decomp = :schur,
)
~~~

当前 Schur 路径面向 p = 2 的非对称 A 实现；默认的 Cholesky 适用于对称正定的 attraction 参数化。模型比较时应保持分解方式一致。

### 拟合离散模型并抽样历史

~~~julia
states = ["A", "B", "?", "A"]  # 顺序与树 tip 一致
fit = fit_corhmm(tree; tip_states = states, model = :ARD)
asr = asr_corhmm(fit; mode = :marginal)
maps = simmap_corhmm(fit; nsim = 100)
~~~

需要严格复现其他实现时，应显式提供 state_order。这样可以固定 Q 矩阵行列、ASR 状态列和 SIMMAP 标签的含义。

### 比较模型

~~~julia
workflow = fit_compare_estim(
    tree,
    lnH,
    [:BM1, :OU1, :EB];
    max_iterations = 400,
)

workflow.aic_table
workflow.best_name
workflow.asr
~~~

多变量对应接口是 fit_compare_multivariate。结果中包含全部 fit、排序后的 AIC 表、最佳模型名称、最佳 fit 和对应的重建结果。

## 树与数据约定

- CompactTree 是似然、SIMMAP 和重建内核使用的内部树表示。
- load_tree(path) 将单棵 Newick 树直接解析为 CompactTree，并拒绝一个文件中的多棵树。
- read_tree(path) 返回 NewickTree.Node；serialize_tree 将其转换为 CompactTree。
- write_tree(path, tree) 写出支持的树对象；to_newick 与 from_compact_tree 提供转换辅助。
- DataFrame 对齐以树 tip 标签为基准。模型支持时，trait 列可以包含缺失值。
- 当前连续树剪枝模型要求二歧树；OU 系列在相应路径中还会检查超度量性。
- SIMMAP 分支片段必须有序，并且片段长度之和必须等于对应枝长。
- 为保证状态编号可复现，应传入 state_order，不要依赖自动排序。

## 根处理与数值选择

OU fit 结果会记录根处理元数据。根据模型，相关参数包括 root_mean_mode、root_cov_mode 和 A_decomp。

- 根均值处理区分 regime 根 optimum 与 stationary-design 处理。
- 根协方差处理区分 fixed-root covariance 与 stationary root covariance。
- 多变量 OU 默认使用 A_decomp = :cholesky；当前非对称 Schur 实现支持两个性状。

这些选项属于统计模型设定，而不只是实现细节。与 R 实现验证或比较模型时，应固定这些选项。

## 线程控制

使用所需线程数启动 Julia，例如 `julia -t 8 --project=.`。连续 OU 系列的多起点路径在有多个 Julia 线程时会使用线程并行。Mk 和 corHMM 提供 fit 层面的 `Ntrials` 与 `Nthreads` 参数控制 trial 并行。当 Julia 线程与多线程 BLAS 同时使用时，应通过 `set_engine_blas_threads!(1)` 显式限制 BLAS 线程数，避免过度订阅。

## 结果与互操作

Fit 结果通常包含：

~~~text
model, success, loglik, aic, nparams,
converged, iterations, f_calls
~~~

根据模型不同，还会包含 theta、alpha、sigma2、A、Sigma、trait_names、regime_names、root_mean_mode、root_cov_mode 和 A_decomp 等字段。

PhyloRef 与 estim_node_table 提供 EvoTraits 节点 ID 和 R/ape 风格节点 ID 之间的稳定映射，也提供 tip anchor 和 descendant-tip signature。

## 仓库结构

~~~text
src/
  io.jl          CompactTree 与 Newick I/O
  criteria.jl    公共信息准则
  phyloref/      树身份与跨实现映射
  discrete/      Mk、corHMM、SIMMAP
  continuous/   单变量、多变量、重建与工作流
  simulate/      树和性状模拟

test/            单元测试与回归测试
validation/      真实数据验证项目
reference/       本地参考实现与研究资料
docs/            notebook 与项目文档
~~~

## 安装

在本地仓库根目录执行：

~~~julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
~~~

## 测试

~~~julia
import Pkg
Pkg.test()
~~~

开发某个子系统时，可以在项目环境中单独运行对应测试文件。validation 下的验证项目与常规包测试保持分离。

## 项目状态与引用

EvoTraits 是持续开发中的研究型软件。正式分析应结合模拟数据，以及 corHMM、OUwie 或 mvMORPH 等独立实现进行数值验证。

如果在研究中使用 EvoTraits，请同时引用本仓库和实际使用的比较方法原始文献。
