# EvoTraits 文档

[English](README.md) · [中文首页](../README_zh.md)

| 分析任务 | 指南 |
| --- | --- |
| 第一次分析 | [入门指南](quickstart_zh.md) |
| 观测准备与缺失处理 | [输入准备](inputs_zh.md) |
| 优化、线程与导出 | [拟合与诊断](fitting_zh.md) |
| 树和数据对齐 | [树操作](trees_zh.md) |
| 离散性状演化 | [离散模型](discrete_zh.md) |
| 性状历史映射 | [SIMMAP](simmap_zh.md) |
| 单性状演化 | [连续模型](continuous_zh.md) |
| 多性状演化 | [多变量模型](multivariate_zh.md) |
| 候选模型比较 | [工作流](workflows_zh.md) |
| 祖先性状 | [祖先重建](reconstruction_zh.md) |
| 模拟树与性状 | [模拟](simulation_zh.md) |
| 环境与时间假设 | [环境与时间转换](environment_shifts_zh.md) |
| 输入与结果约定 | [API 约定](api_zh.md) |
| 常见问题排查 | [常见问题](troubleshooting_zh.md) |
| 运行完整示例 | [示例脚本](examples/README_zh.md) |

Regime 分析可依次阅读[离散模型](discrete_zh.md) → [SIMMAP](simmap_zh.md) → [多变量模型](multivariate_zh.md)。

## 推荐阅读路径

初次使用建议先读[入门](quickstart_zh.md)，再读[输入准备](inputs_zh.md)和[拟合与诊断](fitting_zh.md)。各模型指南分别说明生物学假设、输入、选项及结果解释。指南中的代码片段可能沿用前文的树和数据，示例脚本则可独立运行。

单连续性状分析可按[连续模型](continuous_zh.md) → [模型比较](workflows_zh.md) → [祖先重建](reconstruction_zh.md)阅读。历史环境依赖及指定时间边界的假设见[环境与时间转换](environment_shifts_zh.md)；生成数据及方法评估见[模拟](simulation_zh.md)。

所有指南均有英文对应版本。根目录 README 保留项目概览，详细使用文档统一位于 `doc/`，不需要额外的文档构建依赖。
