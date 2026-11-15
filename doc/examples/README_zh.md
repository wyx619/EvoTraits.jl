# 可运行示例

[English](README.md) · [文档目录](../README_zh.md)

这些脚本自行构造小型合成输入。安装项目依赖后，在仓库根目录执行：

```text
julia --project -t 8 doc/examples/01_inputs.jl
julia --project -t 8 doc/examples/02_discrete_simmap.jl
julia --project -t 8 doc/examples/03_continuous.jl
julia --project -t 8 doc/examples/04_multivariate.jl
julia --project -t 8 doc/examples/05_environment_shifts.jl
```

| 脚本 | 演示内容 | 检查内容 |
| --- | --- | --- |
| [01_inputs.jl](01_inputs.jl) | 物种对齐、Newick 读写、配对裁剪 | 行顺序、保留 tip、映射有效性 |
| [02_discrete_simmap.jl](02_discrete_simmap.jl) | ARD、缺失与多态、根边际后验、SIMMAP | 有限拟合、后验和、多历史往返 |
| [03_continuous.jl](03_continuous.jl) | BM1/BMM/OUM 比较、OUM ASR、滑动窗口 | 默认不 ASR、节点编号、基准行 |
| [04_multivariate.jl](04_multivariate.jl) | 对角 A 的 mvOUM 与 ASR | 参数维度、性状命名输出 |
| [05_environment_shifts.jl](05_environment_shifts.jl) | 环境平滑、EnvExp、预计算 SR | 有限似然与有效拟合 |

[common.jl](common.jl) 提供可复现的 24-tip 树、两个连续性状及合成双 regime 历史。该历史仅用于接口演示；只有离散示例从后验抽样 SIMMAP。

结果打印到终端，树与映射往返文件写入临时目录，不污染源数据。各脚本相互独立，运行第三个不要求第二个先生成文件。

示例采用较小预算以便运行，不用于证明全局最优、实证模型支持或正式性能。即使基本检查通过，也应查看收敛与均值设计诊断。
