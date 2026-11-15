# 输入数据准备

[English](inputs.md) · [目录](README_zh.md) · [下一步：拟合](fitting_zh.md)

## 最小输入约定

| 输入 | 必需结构 | 顺序 |
| --- | --- | --- |
| 树 | 带枝长或定年的 Newick，读为 CompactTree | 决定节点与边身份 |
| 单性状 | 长度为 ntips 的数值向量 | tree.tip_labels |
| 多性状 | ntips 行的数值矩阵 | 行为 tip，列按声明性状顺序 |
| 离散观测 | 物种到状态映射或支持的表格 | state_order 控制状态编号 |
| Regime 历史 | 与该树配对的 SIMMAP | 保留原边编号与状态标签 |
| 环境历史 | 距今年龄与环境值 | 完整覆盖现在至根年龄 |

行顺序决定观测与 tip 的对应关系，错位会改变模型输入。

## 构造分析交集

```julia
using EvoTraits, DataFrames
tree = load_tree("tree.nwk")
traits = DataFrame(taxon = ["A", "B", "C"],
                   height = [1.2, 2.0, 0.8])
keep = intersect(tree.tip_labels, String.(traits.taxon))
length(keep) >= 3 || error("Too few shared taxa")
tree = keep_tip(tree, keep)
y = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:height])
```

物种名是精确标识符。同物异名、重复记录、空白字符应在对齐前处理。每个物种有多条记录时，先选定并记录汇总规则。

该示例假定树中存在这些物种。完全自包含的示例见 [01_inputs.jl](examples/01_inputs.jl)。

## 不同输入路径的缺失值规则

| 路径 | 规则 |
| --- | --- |
| 离散状态 | 显式缺失或完全没有该物种记录，均允许所有状态 |
| align_traits_to_tree 单列 | 必须有限，不接受 missing/NaN |
| align_traits_to_tree 多列 | missing 转 NaN，每行、每列至少一个观测 |
| 连续 DataFrame 拟合包装器 | 按第一列 taxon 对齐，支持的似然可将 missing 转为 NaN |
| 直接向量或矩阵 | 必须满足所选模型的缺失输入约定 |

单列对齐 helper 比部分拟合包装器更严格。需要保留单性状未观测 tip 时，应按模型支持情况明确构造并核验数值向量，不要用零填补缺失。连续性状全缺失行不能直接通过标准多变量对齐 helper。

只有离散观测的物种仍可用于离散拟合；进入连续分析前，将树与抽样 SIMMAP 一起裁到连续数据集合，然后按新树重新对齐。

## DataFrame 快捷接口

单变量拟合包装器以第一列为 taxon。多个数值列时应指定 `trait_col=:height`，避免歧义。多变量包装器会选择首列之后的数值列；元数据也有数字时，显式对齐更便于精确控制。

CSV 便于存储，可先读取表格，再传入对齐数组：

```julia
using CSV, DataFrames
traits = CSV.read("traits.csv", DataFrame)
y = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:height])
```

物种名应保持为字符串，导入后应检查推断类型，尤其注意文本形式的缺失编码。

## 变换与可追溯性

仅对严格正值测量取 ln。导入前明确数据是原始高度、ln(height)、标准差还是方差。枝长缩放会改变速率单位。每次分析应保存物种名单及顺序、变换、来源文件、状态顺序、根设置与随机种子。

参见[树处理](trees_zh.md)、[API 约定](api_zh.md)与[常见问题](troubleshooting_zh.md)。
