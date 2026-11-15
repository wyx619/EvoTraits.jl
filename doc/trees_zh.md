# 树与数据对齐

[English](trees.md) · [文档目录](README_zh.md) · [首页](../README_zh.md)

## 读写

`load_tree(path)` 直接将 Newick 读取为 `CompactTree`。 `read_tree(path)` 返回 NewickTree 节点，再用 `serialize_tree` 得到 CompactTree。 `write_tree(path, tree)` 写树，`to_newick(tree)` 返回 Newick 文本。

```julia
using EvoTraits
tree = load_tree("tree.nwk")
tree.ntips
tree.tip_labels
write_tree("copy.nwk", tree)
```

## 数据对齐

性状向量或矩阵必须按 `tree.tip_labels` 排列；矩阵行为 tip、列为性状。表格可按 taxon 列对齐：

```julia
using DataFrames
data = DataFrame(taxon = tree.tip_labels, height = collect(1.0:tree.ntips))
y = align_traits_to_tree(tree, data; taxon_col = :taxon, trait_cols = [:height])
```

对齐会检查重复标签和缺失 tip，忽略树外多余行。缺失值规则取决于性状维度，参见 [API 约定](api_zh.md)。

## 修剪与身份映射

`keep_tip(tree, tips)` 和 `drop_tip(tree, tips)` 返回修剪后的树。三参数形式 `keep_tip(tree, simmap, tips)` 和 `drop_tip(tree, simmap, tips)` 返回保留映射历史的 `(tree, simmap)`，见 [SIMMAP](simmap_zh.md)。节点与边编号依赖具体树；可用 `build_phyloref`、`phyloref_node_table` 和 `phyloref_edge_table` 映射到 R/ape 顺序。

## 如何选择读取函数

| 操作 | 返回值 | 适用场景 |
| --- | --- | --- |
| `load_tree(path)` | `CompactTree` | 直接进入进化模型计算 |
| `read_tree(path)` | `NewickTree.Node` | 需要逐节点对象表示 |
| `serialize_tree(node)` | `CompactTree` | 转换已读取的节点树 |
| `write_tree(path, tree)` | 写出 Newick | 导出拓扑及枝长 |

文件必须存在，Newick 必须以分号结束；支持中文等非 ASCII 路径。节点式读取在首次解析失败时会清理内部 support/bootstrap 标签后重试。这些接口每个文件读取一棵树。

枝长决定模型的时间单位。例如枝长为 Ma，则吸引速率单位为 Ma⁻¹，扩散速率单位为性状单位²/Ma。程序不会根据物种名推断年代。拟合前应核对模型条件：本指南介绍的 OU、EB 和时间转换流程要求超度量树，相关递推流程要求二叉拓扑。

## tip 位置不等于节点编号

`tree.tip_labels[i]` 对应性状向量或矩阵的第 `i` 行；该 tip 的实际节点编号是 `tree.tip_ids[i]`。内部节点编号、tip 的行号和 ape/R 编号不能混用。

| 字段 | 含义 |
| --- | --- |
| `ntips`、`nnodes`、`nedges` | tip、节点及边的数量 |
| `root` | 根节点编号 |
| `parent_of_edge[e]`、`child_of_edge[e]` | 第 `e` 条边的父、子节点 |
| `edge_length[e]` | 该边枝长 |
| `parent_of_node[v]` | 节点 `v` 的父节点 |
| `dist_from_root[v]` | 根至节点 `v` 的距离 |
| `tip_labels`、`tip_ids` | tip 标签及其节点编号 |

超度量树的高度可取 `height = maximum(tree.dist_from_root)`；节点距今年龄为 `height - tree.dist_from_root[v]`。需要跨软件对照时，先用 `build_phyloref(tree)` 建立参照，再检查 `phyloref_node_table` 和 `phyloref_edge_table`。

## 明确对齐表格

```julia
using DataFrames
# 故意采用与树不同的行顺序。
traits = DataFrame(taxon = reverse(tree.tip_labels),
                   height = collect(1.0:tree.ntips))
y = align_traits_to_tree(tree, traits;
    taxon_col = :taxon, trait_cols = [:height])
@assert length(y) == tree.ntips
```

指定 taxon 列后，重复物种名或树上 tip 无记录会报错，树外多余记录会忽略。不指定 `taxon_col` 时，函数假定表格已经按树排序，并要求行数严格等于 `tree.ntips`。包含编号、坐标或分类编码的表格应显式选择性状列。

只取交集时，先选择物种并裁树，再按**新树**的 tip 顺序对齐数据。已有 SIMMAP 时必须同时裁剪树和映射；单独裁树会使旧边编号失效。缺失值规则见[输入准备](inputs_zh.md)，配对裁剪见 [SIMMAP](simmap_zh.md)。
