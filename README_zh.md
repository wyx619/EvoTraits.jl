# EvoTraits.jl

[English](README.md)

**一个用于研究性状如何跨谱系、性状维度与时间演化的 Julia 框架。**

EvoTraits 将离散与连续性状的系统发育比较方法整合到相互衔接、基于似然的工作流中。从拟合离散性状、抽样演化历史，到检验不同演化状态下的连续性状变化与重建祖先性状，EvoTraits 帮助研究者将进化假说与可解释的模型结果联系起来。

## 核心特点

- **连接离散历史与连续演化。** 拟合 Mk 和 corHMM 模型，抽样 SIMMAP 历史，并将映射的 regime 用于连续模型，同时保留树与分支的对应关系。
- **刻画演化异质性。** 比较 BM、EB 和 OU 假说，包括单变量与多变量分析中随 regime 变化的最适值、吸引强度和扩散速率。
- **探索多性状适应。** 研究性状的联合演化，利用 OU-BM 模型评估演化中的预测变量与适应最适值之间的关系。
- **检验环境与时间假说。** 拟合环境依赖的演化速率模型，并比较指定时间边界前后的演化过程。
- **保持推断过程可追溯。** 检查似然、AIC、收敛状态和可识别性诊断，并对支持的模型重建祖先节点与分支性状。
- **发挥 Julia 的计算能力。** 基于树的似然递推与并行拟合支持可复现的分析，无须为所有观测构造稠密协方差矩阵。

## 安装

```julia
using Pkg
Pkg.add(url = "https://github.com/wyx619/EvoTraits.jl")
```

## 快速开始

```julia
using EvoTraits

tree = mktempdir() do dir
    path = joinpath(dir, "tree.nwk")
    write(path, "((A:1,B:1):1,(C:1,D:1):1);")
    load_tree(path)
end
values = Dict("A" => 1.2, "B" => 1.5, "C" => 1.1, "D" => 1.8)
y = [values[label] for label in tree.tip_labels]

fit = fit_ou1(tree, y)
fit.loglik
fit.aic

asr = estim_node(tree, y, fit)
table = estim_node_table(tree, asr)
```

带有物种名对齐的完整示例见[入门指南](doc/quickstart_zh.md)。模型接口也接受向量、矩阵和字典；`DataFrames.jl` 提供表格输入和输出，`CSV.jl` 可用于读写 CSV 文件。

## 文档

| 主题 | 指南 |
| --- | --- |
| 第一次分析 | [入门指南](doc/quickstart_zh.md) |
| 文档目录 | [中文文档目录](doc/README_zh.md) |
| 数据准备与缺失值 | [输入准备](doc/inputs_zh.md) |
| 拟合、诊断与导出 | [拟合指南](doc/fitting_zh.md) |
| 树与 tip 对齐 | [树操作](doc/trees_zh.md) |
| 离散性状 | [离散模型](doc/discrete_zh.md) |
| 随机性状映射 | [SIMMAP](doc/simmap_zh.md) |
| 单变量连续性状 | [连续模型](doc/continuous_zh.md) |
| 多变量性状与 regime | [多变量模型](doc/multivariate_zh.md) |
| 模型比较 | [工作流](doc/workflows_zh.md) |
| 祖先重建 | [重建](doc/reconstruction_zh.md) |
| 模拟树与性状 | [模拟](doc/simulation_zh.md) |
| 环境与时间转换模型 | [环境与时间转换](doc/environment_shifts_zh.md) |
| 输入、结果与约定 | [API 约定](doc/api_zh.md) |

每个中文页面都链接到对应的英文页面，英文文档目录也提供相同的阅读路径。

可独立运行的[示例脚本](doc/examples/README_zh.md)覆盖主要工作流。对齐、收敛及重建问题参见[常见问题](doc/troubleshooting_zh.md)。

## 计算说明

多起点拟合可以用多线程启动 Julia，例如 `julia -t 8`。EvoTraits 加载时默认将 BLAS 设为单线程，以避免线程过量；如需改变整个 Julia 进程的 BLAS 设置，可在加载后调用 `set_engine_blas_threads!(n)`。解释拟合结果前应检查 `success`、`converged`、边界解和可识别性诊断。模型比较必须使用相同的树、观测和缺失数据规则。

## 许可证

EvoTraits.jl 采用 GNU 通用公共许可证第 3 版（GPL-3.0），详见 [LICENSE](LICENSE)。
