# 入门指南

[English](quickstart.md) · [文档目录](README_zh.md) · [首页](../README_zh.md)

示例自行创建树，并按树的 tip 顺序构造性状向量。

```julia
using EvoTraits, Random

tree = mktempdir() do dir
    path = joinpath(dir, "tree.nwk")
    write(path, "((A:1,B:1):1,(C:1,D:1):1);")
    load_tree(path)
end
values = Dict("A" => 1.2, "B" => 1.5, "C" => 1.1, "D" => 1.8)
y = [values[label] for label in tree.tip_labels]

fit = fit_ou1(tree, y; max_iterations = 1000)
fit.success, fit.converged, fit.loglik, fit.aic
nodes = estim_node_table(tree, estim_node(tree, y, fit))

comparison = fit_compare(tree, y, [:BM1, :OU1, :EB])
comparison.best_name
comparison.aic_table
```

模型比较默认不做 ASR；需要时使用 `fit_compare_estim` 或 `fit_compare(...; reconstruct=true)`，详见[工作流](workflows_zh.md)。解释结果前应检查收敛和参数诊断，参见 [API 约定](api_zh.md)。

物种名索引的 DataFrame 可以先对齐再拟合：

```julia
using DataFrames
data = DataFrame(taxon = tree.tip_labels, trait = y)
y_aligned = align_traits_to_tree(tree, data; taxon_col = :taxon, trait_cols = [:trait])
```

CSV 文件可另外通过 `CSV.read(path, DataFrame)` 读取。对齐检查见[树操作](trees_zh.md)。

## 在独立分析环境安装

建议为分析建立单独的 Julia 项目，独立记录包版本：

```julia
using Pkg
Pkg.activate("evotraits-analysis")
Pkg.add(url = "https://github.com/wyx619/EvoTraits.jl")
```

已有本地仓库时，可改用 `Pkg.develop(path="/absolute/path/to/EvoTraits.jl")`。随后通过 `julia --project=evotraits-analysis -t 8` 启动。开发环境会跟随本地源码变化，正式发表分析时应记录发行版或提交标识。

## 替换为自己的数据

前面的四 tip 示例用于演示调用顺序。实际分析建议：

1. 用 `load_tree` 读取树。
2. 读取并清洗物种及性状表。
3. 选择分析物种并按树对齐。
4. 确定响应尺度和生物学候选模型。
5. 拟合、检查诊断、比较，再进行重建。

带 regime 的模型需先拟合离散模型，再生成配对 SIMMAP。完整阅读路径为[输入准备](inputs_zh.md) → [离散模型](discrete_zh.md) → [SIMMAP](simmap_zh.md) → [连续模型](continuous_zh.md) → [祖先重建](reconstruction_zh.md)。

## 运行完整小例子

在已安装依赖的 EvoTraits 仓库根目录执行：

```text
julia --project -t 8 doc/examples/01_inputs.jl
julia --project -t 8 doc/examples/02_discrete_simmap.jl
julia --project -t 8 doc/examples/03_continuous.jl
```

每个脚本自行生成数据、打印相关结果并检查基本输出约定。文件往返测试使用临时目录，不依赖私有实证数据。多变量和环境模型示例见[示例目录](examples/README_zh.md)。

## 分析完成后保存什么

保存清洗输入、最终树、性状顺序、根假设、随机种子、优化设置、模型比较与诊断。Regime 拟合还需保存精确映射历史。大规模运行前参阅[拟合与导出](fitting_zh.md)。
