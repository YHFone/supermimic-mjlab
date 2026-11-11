# 官方 SuperMimic 实现解剖与本地对照

> 对象
> - **官方**：`/home/zy0028/mjlab_project/supermimic_ws/supermimic`（`HybridRobotics/supermimic`，10,263 行）
> - **本地**：`/home/zy0028/mjlab_project/supermimic-mjlab`（本仓库，对齐后 schema v2）
>
> 方法：逐文件阅读源码 + 加载双方 checkpoint 核对形状与统计量。所有结论都标了 `文件:行`；标 **【对照】** 的是本次会话的实测数字。

---

## 0. 结论速览

| 维度 | 官方 | 本地 | 一致？ |
|---|---|---|---|
| 方法本质 | 世界模型 + 解析梯度反传（BPTT），critic 自举 | 同 | ✅ |
| 是否有 PPO | **没有**（`ppo_runner.py` 只是对照基线，入口 `train_ppo.py`） | 无 | ✅ |
| 表示 | GenCoord `68/7/35`（另有 MaxCoord `182/214/84`） | GenCoord `68/7/35` | ✅ |
| goal | 67/帧 × 10 帧，stride 5；actor 640 / critic 670 | 同 | ✅ |
| ξ | `[friction, com(3), bias(29)]`，按 DR 范围归一化 | 同 | ✅ |
| 奖励 | 8 项，FK 重构后打分，`dt` 缩放 | 同 | ✅ |
| 终止 | 解析软阈值（零参数） | 同 | ✅ |
| 默认 actor 观测 | `rep_state`（68，1 帧，无噪声） | 同（`proprio` 保留为选项，部署用） | ✅ |
| 默认 WM 损失 | Bryson `(0.01, 0.02, 0.1, 3.14)` | 同（`paper` 保留为选项） | ✅ |
| 默认 actor 优化器 | Muon + Adam | 同 | ✅ |
| 归一化器更新 | **每个收集步** | 同（`norm_interval=1`） | ✅ |
| 缺失分支 | — | velocity 槽位、MaxCoord、PPO 基线、W&B、CUDA graph | ❌ |

**最重要的一条更正**：上一轮我说"官方默认是 PPO"是**错的**。官方的 `ActorCriticPolicyLearner` 仍然是用解析梯度穿过世界模型（`policy_learner.py:95-127` 的 `_train_batch` 与 BPTT 共用），它只是比纯 BPTT 多了 critic 自举和 GAE 拟合。

---

## 1. 官方实现解剖

### 1.1 模块地图

| 模块 | 关键行 | 职责 |
|---|---|---|
| `tasks/__init__.py` | 11, 22 | 注册 `SuperMimic-GenCoord-G1`、`SuperMimic-MaxCoord-G1` |
| `tasks/env_cfgs.py` | 160-200 / 202-238 | gencoord / maxcoord 的 mjlab env 配置（观测组、DR、终止） |
| `runners/supermimic_runner.py` | 45-96 | 配置树 `BaseRunnerCfg` + `SuperMimicRunnerCfg` |
| | 122-179 | `build_world_model`：表示 + ξ + `Dynamics` |
| | 182-201 | `build_actor`：`MultiEncoderAE` |
| | 267-296 | `_init_common`：buffer、WM learner、动作管理器 |
| | 402-447 | `_step_and_store`：采集并写 buffer |
| | 607-656 | `step()`：一次仿真步 + RSI 记录 |
| | 663-762 | `run()`：主循环（`should_train`、WM/策略更新、日志、存档、ONNX） |
| `learners/wm_learner.py` | 24-43 | Bryson 权重（`1/nominal²`） |
| | 104-119 | `update()`：10 个 batch，各一步优化 |
| | 145-191 | 自回归 7 步 + 分组 MSE |
| `learners/policy_learner.py` | 19-36 | 两个 learner 的配置（γ、λ、noise、clip） |
| | 42-131 | `BpttPolicyLearner`：想象 32 步、`loss = -ret` |
| | 139-202 | `ActorCriticPolicyLearner`：加 bootstrap + GAE critic |
| `learners/learner_utils.py` | 13-36 | Muon（Sequential 中间 Linear）+ Adam(fused) |
| `envs/world_model_env.py` | 96-160 | `reset/step`：动力学 → 终止 → 奖励 → 推进参考 |
| | 162-219 | `_wm_step` 与解析软终止 |
| `envs/multi_task_world_model_env.py` | 20-66 | 多槽位 goal（motion / velocity） |
| `models.py` | 31-43 | `build_mlp`：`Linear→LayerNorm→GELU` |
| | 46-99 | `Dynamics`：proc + 归一化 + 预测 |
| | 102-229 | `MultiEncoderAE`：每任务 encoder + 共享 decoder + 槽位 + latent BC |
| `representations.py` | 96-111 | goal 常量（10 帧、stride 5） |
| | 213-234 | `assemble_anchor_window`（root 相对） |
| | 401-417 | `compute_wm_goal`：content + rel_pos + rel_ori |
| | 469-554 | `GenCoordRepresentation`（68/7/35 + 半隐式积分） |
| | 569-612 | `loss_blocks` / `world_frame_bodies` / `compute_wm_obs` |
| `sim_context.py` | 26-31, 171-192 | ξ 的 `mean/scale`（半宽 <1e-3 时 scale=1） |
| `replay_buffer.py` | 60-121 | TensorDict 环形缓冲，窗口不跨 episode |
| `utils/tracking_reward.py` | 46-89 | 8 项跟踪奖励（re-anchor 到预测 root 的 xy+yaw） |

### 1.2 一次训练迭代的数据流（默认配置）

```
[仿真侧] 4096 envs，控制 50 Hz（decimation=4，物理 200 Hz）
  obs = { full_state: 71, actor: 93, goal: 670 }
  rep  = represent(full_state)        → 68   （z, g_b, v_b, ω_b, qj, vj）
  aux  = [root_pos(3), root_quat(4)]  → 7
  ξ    = [摩擦(1), 质心(3), 偏置(29)] → 33
  a_t  = actor(history, goal_640) + N(0, 0.1²)
  env.step(a_t)
  写 buffer：state71 / action29 / prev_action29 / motion_time / terminated / truncated / ξ33
  RSI：每 20 步更新 V[b]，把新分布写回 motion command

[训练侧] should_train = 4096/10/2048/8 × 16 = 0.4，即平均每 2.5 个仿真步一次
  ① 世界模型：10 个 batch × 2048 个 8 帧窗口
       自回归 7 步（最后一个动作不计），Bryson 损失，每 batch 一步优化
  ② 策略：4096 条想象 rollout × 32 步
       a_t = actor(o_t, g_t)
       (aux', rep') = Dynamics(rep, a_t, aux, ξ)          # proc(a)=a·scale+offset
       r_t = tracking_reward(FK(reconstruct(aux',rep')), 参考体状态)
       d_t = σ((max(|Δz|/0.25, |Δg_z|/0.8) − 1)/0.05)     # 零参数
       J   = Σ_t γ^t c_t r_t + γ^H c_H V(rep_H, goal_H, ξ)  # 见 1.5
       actor 一步（Muon+Adam，clip 1.0 / 逐元素 0.1）
       critic：GAE 目标 clamp(max=10)，5 epoch × 4 minibatch
```

### 1.3 网络与超参（与 checkpoint 形状逐一核对）

| 组件 | 结构 / 取值 |
|---|---|
| `Dynamics` | 输入 `rep 68 + proc(a) 29 + ξ 33 = 130` → `Linear+LN+GELU` ×3（2048）→ 输出 **35**；rep/action 各一个 `EmpiricalNormalization`（eps=1e-2，eval 下 no-op） |
| 动作变换 | `proc(a) = a·scale + offset`，`scale=0.25·effort/stiffness`，`offset=default_joint_pos` |
| 积分 | 半隐式：`qj' = qj + (vj + c·Δvj)dt`，`c=(d+1)/(2d)=0.625`，姿态用四元数积分 |
| Actor | 每任务 encoder `640→2048→1024→512→256→32`（tanh）；共享 decoder `(32 + H·obs) → 2048→1024→512→256 → 29`（LN+GELU） |
| Critic | `771 (=68+670+33) → 256 → 128 → 1`（LN+GELU，**无输入归一化**），输出 `clamp(max=10)` |
| 超参 | `k=8`、WM batch 2048×10、`H=32`、policy batch 4096、`γ=0.99`（AC）/`1.0`（BPTT）、`λ=0.95`、lr `2e-4`、1e6 迭代、buffer 1000/env、`replay_ratio=16` |

### 1.4 七个分支

| # | 维度 | 入口 | 默认 | 可选 | 影响 |
|---|---|---|---|---|---|
1 | 任务/表示 | 位置参数 `task_id` | `SuperMimic-GenCoord-G1` | `SuperMimic-MaxCoord-G1` | `full_state` 布局与表示类 |
2 | 表示 | `--runner.representation` | `gencoord` | `maxcoord` | `68/7/35` vs `182/214/84` |
3 | actor 观测 | `--runner.actor-obs` | `rep_state` | `proprio` | 68 无噪 vs 93 带噪 |
4 | 历史帧 | `--runner.history-length` | `1` | `>1`（仅 proprio） | actor 输入宽度 |
5 | learner | `--runner.policy:{actor-critic,bptt}` | **`actor-critic`** | `bptt` | γ、是否 critic 自举 |
6 | 复位采样 | `--runner.sampling-mode` | `return` | `uniform/adaptive/start` | RSI 分布 |
7 | 任务槽 | `--runner.slot-mix` | `{motion:1.0, velocity:0.0}` | 任意和为 1 | 多 encoder + latent BC |

### 1.5 两个 learner 的精确差异

| | `BpttPolicyLearner` | `ActorCriticPolicyLearner`（默认） |
|---|---|---|
| γ | **1.0**（`policy_learner.py:29`） | **0.99**（`:34`） |
| 自举 | `_bootstrap()` 返回 `0.0`（`:133-134`） | `γ^H·V(s_H,g_H,ξ)·c_H`，V `clamp(10)`（`:169-171`） |
| critic | 不创建、不训练 | 建 771→256→128→1，GAE 拟合（`:173-194`） |
| 回报处理 | 无截断 | GAE 回报 `clamp(max=v_max)`（`:187`） |
| 优化器 | Muon+Adam（两者共用 `build_optimizers`） | 同 |
| 梯度裁剪 | `max_norm=1.0`，逐元素 0.1（`:122-124`） | 同 |

**共同点**：actor 更新都用 `loss = (-ret + aux_loss).mean()`，梯度穿过冻结的 `Dynamics`。也就是说，"actor-critic"只是"多了一个 critic 来延伸视界"，不是 PPO。

---

## 2. 本地实现解剖

### 2.1 模块映射

| 官方 | 本地 | 说明 |
|---|---|---|
`representations.py` | `algo/representation.py`（`GenCoordRepresentation`，:84-260） | 逐行移植 GenCoord |
`models.py::Dynamics` | `algo/networks.py::WorldModel`（:57-143） | 同结构，含 proc 与归一化 |
`models.py::MultiEncoderAE` | `algo/networks.py::ActorCritic`（:166-210） | 单 encoder（无 velocity 槽） |
`learners/policy_learner.py` | `algo/trainer.py::_update_policy`（:574-） | 一个类里用 `policy_mode` 切换 bptt/AC |
`learners/wm_learner.py` | `algo/trainer.py::_update_world_model`（:420-） | 含 Bryson/paper 两种权重 |
`envs/world_model_env.py` | `algo/trainer.py::_update_policy` 的 rollout 段 | 想象逻辑内联 |
`replay_buffer.py` | `algo/buffer.py` | 环形缓冲，丢弃跨 episode 窗口 |
`runners/reset_sampling.py` | `algo/sampler.py::ReturnSampler`（:41-） | explore/focus/keep 一致 |
`sim_context.py` | `algo/state.py::SimContext`（:129-） | 列序与归一化一致 |
`tasks/env_cfgs.py` | `config/g1/env_cfgs.py` | 观测组/DR/终止一致 |
`runners/exporter.py` | `algo/export.py` + `algo/deploy_features.py` | ONNX + sidecar 契约 |

### 2.2 数值契约对照

| 项 | 官方 | 本地 | 结论 |
|---|---|---|---|
| full / aux / rep / pred | 71 / 7 / 68 / 35 | 同 | ✅ |
| 半隐式系数 | `(d+1)/(2d)=0.625` | 同 | ✅ |
| goal 每帧 | `qj(29)+vj(29)+rel_pos(3)+rel_ori(6)` | 同 | ✅ |
| goal 窗口 | 10 帧 × stride 5，含当前帧 | 同 | ✅ |
| goal 总维 | actor 640 / critic 670 | 同 | ✅ |
| rel_ori | `matrix[..., :2]` 前两**行**展平 | 同（部署侧曾误用前两列，已修） | ✅ |
| ξ 列序 | friction, com, bias | 同 | ✅ |
| ξ 归一化 | `(raw−mean)/scale`，半宽<1e-3 时 scale=1 | 同 | ✅ |
| 动作变换 | `a·scale+offset` | 同（与官方 checkpoint 逐元素相同） | ✅ |
| 奖励 | 8 项 + re-anchor + dt | 同 | ✅ |
| 终止 | `σ((max(|Δz|/0.25,|Δg_z|/0.8)−1)/0.05)` | 同 | ✅ |
| WM 损失 | Bryson `1/nominal²`，评 7 步 | 同（`paper` 可选） | ✅ |
| 归一化器 | eps=1e-2 加在 std 外，eval no-op | 同 | ✅ |
| critic | 无归一化，clamp 10 | 同 | ✅ |
| 优化器 | Muon(WM+actor)+Adam | 同 | ✅ |
| 归一化更新 | 每收集步 | 同 | ✅ |
| actor 输入 | 708（rep 68 + goal 640） | 同（默认 rep_state） | ✅ |
| 多任务槽 | motion + velocity | 仅 motion | ❌ |
| MaxCoord | 有 | 无 | ❌ |

---

## 3. 差异清单

**已一致（可直接复用官方结论）**：表示与积分、goal 布局/stride、ξ 列序与归一化、动作 proc、
奖励 8 项、解析终止、critic 结构（无归一化 + clamp）、GAE 形式、Muon 分配规则、
RSI 三成分、buffer 不跨 episode、ONNX 只允许 proprio。

**已改为官方默认（2026-11-11）**：
1. `actor_obs_mode="rep_state"`、`obs_history=1`；
2. `wm_loss_mode="bryson"`；
3. `use_muon_actor=True`；
4. `norm_interval=1`（世界模型归一化器每个收集步更新）。

以上四项均保留旧行为为显式选项（`proprio` + `obs_history=10`、`wm_loss_mode="paper"`、
`use_muon_actor=False`、`norm_interval>1`），便于 A/B。

**官方有、本地无**：velocity 任务槽与 latent BC、MaxCoord 表示、PPO 基线 runner、
W&B、CUDA graph 捕获、ONNX 的 15000 帧上限。

---

## 4. 实测对照

### 4.1 训练曲线（4096 envs，LAFAN 40 条，normalized return）

| iteration | 旧代码（对齐前） | 对齐版（rep_state/bryson/Muon/AC） |
|---|---|---|
| 10 | — | 0.250 |
| 500 | 0.212 | 0.236 |
| 1000 | 0.218 | 0.250 |
| 2000 | **0.394** | 0.282 |
| 5000 | **0.550** | 0.252 |
| 6570 | **0.614** | **0.180** |
| 8000 | 0.657 | — |

对齐版在 1000 迭代后平台化并下滑；旧代码持续上升。

### 4.2 1024 envs（UTD 高 4 倍）@800 迭代

| 配置 | normalized return |
|---|---|
| A 对齐版 rep_state | 0.257 |
| B 对齐版 proprio/10 | 0.297 |
| C 旧代码 | 0.197 |

高 UTD 下对齐版反超 → 不是"代码坏了"，而是**低 UTD 下数据效率差**。

### 4.3 停掉的三组长跑（约 1250 迭代）

| 运行 | 配置 | return | episode len | WM 7 步误差 |
|---|---|---|---|---|
| `O_official` | rep_state/1 + bryson + Muon + **AC** | 0.194 | 33.9 | 0.650 |
| `B_bptt` | 同上 + **policy=bptt**（γ=1，无 critic） | **0.406** | 41.9 | 0.516 |
| `P_paper` | proprio/10 + paper + Adam + AC | 0.206 | 32.5 | **0.250** |

两个事实：
1. **同一个世界模型设置下，纯 BPTT 明显好于 AC**（0.406 vs 0.194）——critic 在早期不可靠，
   `γ^32≈0.72` 权重的自举把 actor 拉向错误方向；
2. **paper 损失把 WM 拟合好一倍**（0.250 vs 0.650），但策略回报没跟上——说明瓶颈在
   "策略如何用模型"，不在"模型精度"。

### 4.4 checkpoint 交叉评估（60 s，确定性，40 条 clip）

| 权重 | normalized return | 锚点位置误差 | 成功率 |
|---|---|---|---|
| 官方 8k 迭代 actor | **−2.76** | 1.24 m | 0/40 |
| 本地对齐版 6k 迭代 | 0.556 | 0.243 m | 0/40 |

⚠️ 该对比只作参考：官方权重是在他们自己的 motion npz 上训练的，若关节顺序/参考格式不同，
跨环境评估不代表能力。**不能据此判断官方训练好坏**。

---

## 5. 建议

1. ~~默认值回到官方口径~~ **已完成**：`rep_state/1`、`bryson`、`Muon actor`、`norm_interval=1`；
   `proprio/10`、`paper` 保留为显式选项。
2. **learner 默认改为 `bptt`**（γ=1、无 critic），至少在早期 2k–5k 迭代；AC 只在
   `H` 需要超过 32、或策略已经稳定后启用。上面 4.3 的数据支持这个顺序（先 BPTT 再开 critic，
   或者把 critic 的启动延后，类似 warmup）。
3. **WM 损失取 `paper` 或折中**：把 `joint_vel` 的名义误差从 3.14 改小（例如 0.5），
   让关节/速度通道在 BPTT 下拿到有意义的权重；`root_pos` 的 1e-2 太强。
4. 缺失分支里优先补 **velocity 槽位**（论文贡献 3/4），MaxCoord 与 PPO 基线可延后。

---

## 附录 A：官方源码索引

| 结论 | 位置 |
|---|---|
| 两个任务注册 | `tasks/__init__.py:11,22` |
| 配置默认值 | `runners/supermimic_runner.py:45-96` |
| 表示/动力学构建 | `:122-179` |
| actor 构建 | `:182-201` |
| learner 选择 | `:515-559` |
| 采集一步 | `:607-656` |
| 训练主循环 | `:663-762` |
| Bryson 权重 | `learners/wm_learner.py:64-68` |
| 7 步 rollout | `learners/wm_learner.py:145-167` |
| BPTT γ=1 / 无 bootstrap | `learners/policy_learner.py:29,133-134` |
| AC γ=0.99 / bootstrap / clamp | `learners/policy_learner.py:34,169-171,187` |
| Muon 分配 | `learners/learner_utils.py:13-36` |
| 解析软终止 | `envs/world_model_env.py:195-219` |
| 奖励 8 项 | `utils/tracking_reward.py:46-89` |
| goal 常量 | `representations.py:96-111` |
| root 相对锚点 | `representations.py:213-234` |
| GenCoord 表示 | `representations.py:469-554` |
| ξ 归一化 | `sim_context.py:26-31,171-192` |
| 窗口不跨 episode | `replay_buffer.py:88-121` |

## 附录 B：本地源码索引

| 结论 | 位置 |
|---|---|
| GenCoord 表示 | `src/supermimic/algo/representation.py:84-260` |
| 动力学 / actor / critic | `src/supermimic/algo/networks.py:57-210` |
| 训练器（采集/拟合/想象） | `src/supermimic/algo/trainer.py:335,420,574` |
| learner 分支 | `src/supermimic/algo/config.py`（`policy_mode`） |
| 解析终止 | `src/supermimic/algo/reward.py:61` |
| RSI | `src/supermimic/algo/sampler.py:41-160` |
| ξ | `src/supermimic/algo/state.py:129` |
| goal | `src/supermimic/mdp/features.py:120`、`mdp/commands.py` 的 `goal()` |
| 导出/部署契约 | `src/supermimic/algo/export.py:44,112` 与 `deploy_features.py` |
