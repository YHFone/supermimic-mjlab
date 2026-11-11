# 官方 GenCoord 对齐：实施记录

> 目标：把本仓库的行为性差异改成与 `HybridRobotics/supermimic` 一致，表示层使用
> **GenCoord**（不实现 MaxCoord）。本文件记录落实的契约与证据位置；对齐前的逐条
> 差异见 `docs/compare_upstream_zh.md`。

## 1. 契约（schema v2）

| 量 | 对齐后 | 代码位置 | 测试 |
|---|---|---|---|
| full / aux / rep / pred | 71 / 7 / **68** / **35** | `algo/representation.py` | `test_representation_dims` |
| 半隐式积分系数 | $c=(d+1)/(2d)=0.625$ | `GenCoordRepresentation.substep_coeff` | `test_semi_implicit_integration` |
| WM 输入 | `rep(68)+a(29)+ξ(33)=130`，不含 goal | `algo/networks.py` | `test_world_model_shapes_and_bryson_loss` |
| 每帧 goal / 窗口 | **67** / 10 帧 × **stride 5** | `mdp/features.py` | `test_goal_dims_and_layout` |
| goal 总维 | actor **640** / critic **670** | `GoalFeatures.actor_dim/wm_dim` | 同上 |
| actor 观测 | `rep_state` 68×1（默认）/ `proprio` 93×H | `cfg.actor_obs_mode` | `test_history_beyond_one_requires_proprio` |
| critic | `rep+wm_goal+ξ=771`，**不归一化**，`clamp(max=10)` | `ActorCritic.value` | `test_critic_value_is_clamped` |
| 终止 | `σ((max(|Δz|/0.25, |Δg_z|/0.8)−1)/0.05)` | `algo/reward.py` | `test_soft_termination_thresholds` |
| WM 损失 | Bryson 分组 `(0.01,0.02,0.1,3.14)`，评 **7** 步 | `WorldModel.loss` | 同上 |
| 归一化 | `x/(std+1e-2)`，eval no-op | `algo/empirical_norm.py` | `test_empirical_norm_eval_is_noop` |
| ξ | `[friction, com(3), bias(29)]`，按 DR 范围归一 | `algo/state.py::SimContext` | `test_context_encoder_bias_...` |
| 优化器 | Muon（隐藏层 Linear）+ Adam | `trainer.build_optimizers` | 覆盖于 smoke |
| 调度 | 恒定 lr、1e6 迭代 | `config.py` / `rl_cfg.py` | `test_g1_rl_cfg_matches_official_defaults` |
| RSI | explore 0.5 / focus(25) / keep 0.1，memory 256 | `algo/sampler.py` | `test_sampler.py` |
| 窗口采样 | 严格不跨 episode | `algo/buffer.py` | `test_buffer.py` |

## 2. 与论文的三处偏离（按官方实现）

1. **actor 观测默认是 68 维 rep_state、单帧、无噪声**；论文 Table A.1 的 93 维噪声
   proprio 保留为 `actor_obs_mode="proprio"`，也是唯一可导出部署的模式。
2. **终止是解析软阈值**，没有论文 Eq.(1) 的 BCE 学习头（官方已弃用）。
3. **WM 损失用 Bryson 加权**（`1/nominal²`，按通道数摊），不是"归一化空间等权"。

## 3. 破坏性变更

- checkpoint / ONNX 带 `schema_version=2`，旧版本直接报错（`algo/schema.py`）。
- actor 输入维度：rep_state 708（68+640），proprio `93H+640`。
- 部署只支持 `actor_obs_mode="proprio"`；`rep_state` 策略 `export.infer_actor_spec`
  会明确拒绝。

## 4. 验证状态

- `pytest -q`：39 passed, 1 skipped（smoke 需要 CUDA，本机 shell 无 GPU）。
- 待办：GPU 机器上跑 `tests/test_smoke.py` 与 2 万迭代短训，检查
  `world_model/window_error` 是否落在 0.007–0.011 量级。
