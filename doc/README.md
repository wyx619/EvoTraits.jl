# EvoTraits documentation

[中文目录](README_zh.md) · [Project home](../README.md)

| Task | Guide |
| --- | --- |
| First analysis | [Getting started](quickstart.md) |
| Prepare observations and missing data | [Input preparation](inputs.md) |
| Optimization, threads and exports | [Fitting and diagnostics](fitting.md) |
| Trees and alignment | [Trees](trees.md) |
| Discrete evolution | [Discrete models](discrete.md) |
| Mapped histories | [SIMMAP](simmap.md) |
| Single-trait evolution | [Continuous models](continuous.md) |
| Multiple traits | [Multivariate models](multivariate.md) |
| Candidate comparison | [Workflows](workflows.md) |
| Ancestral states | [Reconstruction](reconstruction.md) |
| Simulate trees and traits | [Simulation](simulation.md) |
| Environmental and temporal hypotheses | [Environment and shifts](environment_shifts.md) |
| Inputs and results | [API conventions](api.md) |
| Diagnose common problems | [Troubleshooting](troubleshooting.md) |
| Run complete examples | [Example scripts](examples/README.md) |

For regime-dependent analysis, follow [Discrete models](discrete.md) → [SIMMAP](simmap.md) → [Multivariate models](multivariate.md).

## Reading paths

New users should start with [Getting started](quickstart.md), then [Input preparation](inputs.md) and [Fitting](fitting.md). Each model guide explains its biological hypothesis, accepted inputs, supported options and result interpretation. Code fragments within a guide may assume the tree and data introduced earlier; the example scripts are self-contained.

For a single continuous trait, read [Continuous models](continuous.md) → [Comparison](workflows.md) → [Reconstruction](reconstruction.md). For rate dependence on a historical covariate or a specified temporal boundary, start with [Environment and shifts](environment_shifts.md). For data generation and method assessment, use [Simulation](simulation.md).

All guides have Chinese counterparts. The root README is a project overview; detailed usage is maintained here in `doc/` without a separate documentation build dependency.
