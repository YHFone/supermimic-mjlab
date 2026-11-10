# P2 单位、尺度与归一化对账（SuperMimic-mjlab）

> 范围：`algo/{state,networks,trainer,reward}.py`，消费者追到 `mdp/commands.py`、`mdp/features.py`、
> `scripts/{play,evaluate}.py`、`algo/{deploy_features,export}.py`。
> 环境：`tfbot_rl`（RTX 4090 可用）；反例脚本在 `scratch/audit_p1/`。
> 约定：$J=29$，$H=10$，$K=10$，$dt=0.02$ s，$\gamma=0.99$，$\lambda=0.95$。

> 2026-11-10 修复后：P2-11/F4（bias 切片）与 P2-12/F1（bin 单位）已修；
> P2-2/F6 的注释已改为与实际一致；P2-8 已在 `MotionLibraryCommand` 加 `dt` 断言。
> P2-1（action_norm 带噪统计）保留（实测影响 3.7%）。
> 下表的行号/结论保留修复前状态，具体修复见 `audit_report_p1_zh.md` §10。

## 0. 结论摘要

| 编号 | 结论 | 级别 |
|---|---|---|
| P2-1 | `action_norm` 统计来自**带噪动作**，想象时却归一化**无噪动作**；实测归一化后 std 0.963（−3.7%） | 🟢 |
| P2-2 | `critic_norm` 只在真实转移上拟合，critic 却在想象状态上训练/自举；H=32 实测 std 漂移 5–7%，注释理由与代码矛盾 | 🟡 |
| P2-3 | `state_norm`/`goal_norm` 用真实/采集统计去归一化 WM 预测的状态/目标；同 P2-2 机制，漂移随 WM 质量变化 | 🟢 |
| P2-4 | 所有 RunningNorm 都是**累计平均**（count 只增不减），等效记忆=全部历史；单次更新权重 $1/(k+1)$，晚期突变被强烈阻尼（最后 3% 迭代只占 3.1%） | 🟡（设计风险） |
| P2-5 | `norm_interval=10` 期间用陈旧统计：最多 100 个 WM 梯度步 / 25 个控制步（每步 4096 env），相对“全部历史”的记忆可忽略 | 🟢 |
| P2-6 | `eps=1e-5` 在 sqrt 内；无输入 clip。常数维 var=0 时 std=0.00316，0.01 偏差→3.16σ；预测发散时无上限 | 🟢 |
| P2-7 | BPTT 穿过归一化：统计是 buffer（无梯度），输入梯度恰为 $1/\mathrm{std}$；`update()` 在 `no_grad`+`detach()` 下，无意外 detach | ✓ |
| P2-8 | `library.dt`（来自 fps）与 `env.step_dt` 未绑定/未断言；默认都是 0.02，但改 `--output-fps` 或 `decimation` 会静默错速（30 fps → 1.667×） | 🟡（潜在） |
| P2-9 | `max_return=10` 由 $0.1/(1-\gamma)$ 推出，但与 reward 权重/dt/γ 没有代码级绑定；改 reward 会静默改变 softmax 温度语义 | 🟢 |
| P2-10 | 评估/部署用**无噪** obs，却用训练时**含噪**统计归一化；joint-vel 通道实测约 0.96（−4%） | 🟢 |
| P2-11 | encoder-bias 切片 off-by-one（P1 的 F4）：把 COM x（DR ±0.025 m）当弧度加到 joint-28；这是本对账里唯一的**量纲错配** | 🟡 |
| P2-12 | `_finish_episodes` 的 `bin_frames` 单位错（P1 的 F1）：frames×dt 得 1，bin 回报变成 bin 末端 | 🔴（已在 P1） |

## 1. RunningNorm 清单（谁拟合什么分布）

| 实例 | 维度 | frozen | 更新点 | batch / 频率 | 拟合的分布 | 消费者 |
|---|---|---|---|---|---|---|
| `WorldModel.state_norm` | 71 | 3..6（quat） | `trainer.py:408-421` → `networks.py:186-191` | 65536 / 每 10 iter | 真实 buffer 状态（含边界状态） | `WorldModel.forward`、`state_loss`、`termination_logits` |
| `WorldModel.action_norm` | 29 | — | 同上 | 65536 / 每 10 iter | **带噪** buffer 动作 | `WorldModel.forward` |
| `WorldModel.context_norm` | 33 | — | 同上 | 65536 / 每 10 iter | 真实 buffer context（DR 参数） | `WorldModel.forward` |
| `WorldModel.goal_norm` | 730 | — | `trainer.py:245`（每次 collect） | 4096 / env step | 采集时的真实 goal | `termination_logits` |
| `ActorCritic.obs_norm` | 93 | — | `trainer.py:243`（每次 collect） | 4096 / env step | **带噪**观测（含 encoder bias） | `ActorCritic.act` |
| `ActorCritic.goal_norm` | 730 | — | `trainer.py:244`（每次 collect） | 4096 / env step | 采集时的真实 goal | `ActorCritic.act` |
| `ActorCritic.critic_norm` | 834 | — | `trainer.py:423-441`（每 10 iter） | 8192 / 每 10 iter | **仅真实** buffer 状态+goal+context | `ActorCritic.value` |

补充：
- `obs_norm.update` 在 `act` **之前**（`trainer.py:243-246`），所以第一个动作已经用上本批统计。
- `world_model.goal_norm` 与 `actor_critic.goal_norm` 用**同一个 goal 张量**更新（`trainer.py:244-245`），
  两者统计相同；但它们与 `state_norm` 的更新频率不同（goal 每步，state 每 10 iter）。
- `state_norm` 的 frozen 维只影响 `normalize`（原样返回）和 `state_loss` 的非四元数选择；
  `update` 仍会为这 4 维累计 mean/var（未使用）。`critic_norm` **不冻结**四元数，同一物理量在
  两个网络里用了不同归一化（设计选择，非 bug）。

## 2. 张量 × 消费者对账

| 张量（生产点） | 原始 / 无量纲 | 消费者（期望尺度） | 匹配 | 备注 |
|---|---|---|---|---|
| `state`（`state.py:156-168`） | 原始：m, 单位 quat, m/s, rad/s, rad, rad/s | buffer/reward/FK（原始）；`state_norm`（无量纲）；`critic_norm`（无量纲） | ✓ | 四元数 wxyz；根位置 env-local |
| `state_norm(state)`（`networks.py:162`） | 无量纲（quat 维原样） | WM 输入、`state_loss`、term head | ✓ | 与 `critic_norm` 对同一 state 的归一化不同，各自独立 |
| `action`（`trainer.py:246-247`） | 原始：无量纲（关节 setpoint 偏移，env 再乘 `G1_ACTION_SCALE`） | env（乘 scale）、reward action_rate（原始）、buffer | ✓ | 采集存的 `a+ε` |
| `action_norm(action)`（`networks.py:163`） | 无量纲 | WM 训练（buffer 的带噪动作） | ✓ | — |
| 同上，想象时 | 无量纲 | `_update_policy` 的**无噪**动作 | ⚠ P2-1 | 统计含 $\sigma^2=0.01$，归一化后 std 0.963 |
| `context`（`state.py:173-182`） | 原始：$\mu$ 无量纲、bias 29 rad、$\Delta c$ 3 m | WM（`context_norm`）、critic（`critic_norm`）、proprio bias（原始） | ⚠ | `context_norm`/`critic_norm` 正确；proprio 切片错（P2-11） |
| `obs` frame（`features.py:60-101`） | 原始：rad, rad/s, rad/s, 无量纲, 原始 action | `obs_norm`（无量纲）；history；buffer 不存 | ✓/⚠ | 训练加噪；eval/play 无噪（P2-10）；joint-pos 通道混入 COM x（P2-11） |
| `goal`（`features.py:139-187`） | 原始：rad, rad/s, m, 无量纲(旋转矩阵前两列), m/s, rad/s | `goal_norm`（actor/WM）、`critic_norm`（critic） | ✓ | 逐帧 73；K=10；参考系/机器人 root 约定见测试 |
| `goal_norm(goal)`（`networks.py:287`） | 无量纲 | actor encoder | ✓ | ONNX 中烘进图 |
| `goal_norm(goal)`（`networks.py:195`） | 无量纲 | WM term head | ✓ | 与 actor 的 goal_norm 同步更新 |
| `critic input`（`networks.py:293`） | 原始：state(71)+goal(730)+context(33) | `critic_norm` | ⚠ P2-2 | 统计只在真实转移上拟合 |
| `critic_norm(input)` | 无量纲 | `value()`（想象状态、策略 bootstrap） | ⚠ | 实测 H=32 std 漂移 5–7% |
| `reward` 各 term（`reward.py:146-200`） | 无量纲指数项；action_rate 无量纲²；joint_limit rad | 求和×dt；buffer reward；policy objective；critic target；logging | ✓ | 单位/σ 见 §3 |
| `G_b`（`sampler.py`） | 无量纲回报（γ 折扣） | `softmax(-25G/10)` 与 reset 概率 | ⚠ P2-12 | bin 起点 vs 末端单位错（P1 F1） |
| `episode_return` / normalized | 无量纲 | 日志 `0.1*length` 归一化 | ✓ | 与论文 normalized return 定义一致 |

## 3. 奖励 / 折扣 / 尺度常量

| 常量 | 值 | 单位 | 消费者 / 备注 |
|---|---|---|---|
| `reward.dt` | 0.02 | s | `trainer.py:93` 取 `env.step_dt`；与 50 Hz 一致 |
| `env.step_dt` | 0.02 | s | `sim.timestep=0.005` × `decimation=4` |
| `library.dt` | 0.02 | s | 文件 `fps=50`；**未与 step_dt 断言**（P2-8） |
| `root_pos_std` | 0.3 | m | exp(−m²/0.3²) |
| `root_ori_std` | 0.4 | rad | exp(−θ²/0.4²)，θ 由 $4\arccos(\lvert q_1\cdot q_2\rvert)^2$ 给出 |
| `body_pos_std` | 0.3 | m | mean over bodies |
| `body_ori_std` | 0.4 | rad | 同上 |
| `body_lin_vel_std` | 1.0 | m/s | 世界系速度差 |
| `body_ang_vel_std` | 3.14 | rad/s | ≈π |
| `action_rate_weight` | −0.1 | 无量纲 | $\sum_j(a_t-a_{t-1})_j^2$，再 ×dt |
| `joint_limit_weight` | −2 | rad | 正违反和，再 ×dt |
| `γ` | 0.99 | — | 每控制步；50 Hz 下 2 s 视界 |
| `λ` | 0.95 | — | 仅 critic target |
| `max_return` | 10 | 无量纲 | $0.1/(1-0.99)=10$；与 reward 配置无代码绑定（P2-9） |
| `temperature` | 25 | — | 与论文 D.1 的 $-25G/G_{\max}$ 一致 |

`reward` 的结构检查：每个指数项的分母是“容差²”，分子是“误差²”，指数无量纲；
`action_rate` 用原始动作（env 的 `G1_ACTION_SCALE` 未乘），`joint_limit` 用绝对关节角对
soft limits；所有项最后乘 `dt`，与 mjlab `RewardManager(scale_by_dt=True)` 和论文 Table A.2 一致。

## 4. 数值证据

### P2-1 `action_norm`：带噪统计 vs 无噪消费

`WorldModel.update_normalizers` 用 buffer 的 `a+ε` 更新 `action_norm`；`_update_policy` 传入
的是 actor 的**无噪**输出。实测（34k checkpoint，120 步 rollout，8 env）：

```
clean per-dim action std      = 0.3579
noisy std (σ=0.1)             = 0.3716
normalized clean std ratio    = 0.9631
```

即 WM 在想象里看到的动作被系统性压缩 3.7%。量级小，但属于“用 A 的统计量归一化 B 的数据”。
修法：用无噪动作拟合 `action_norm`（buffer 另存 clean action），或想象时也加同分布噪声。

### P2-2 `critic_norm`：真实统计 vs 想象输入

`_update_critic_normalizer` 的注释说“统计跟踪 replay buffer，也是 bootstrap 被评估的地方”；
但 `_update_critic` 的 `states` 来自 `chunk_states`（想象），critic 只在这些状态上训练/自举。
实测（34k checkpoint，B=256，H=32，`critic_norm` 只用 512 个 live 真实状态拟合）：

| block | real mean/std/max\|z\|/frac>4 | imagined (t=31) |
|---|---|---|
| state | 0.065 / 0.960 / 4.881 / 0.0153 | 0.052 / 0.897 / 4.885 / 0.0143 |
| goal | 0.036 / 0.870 / 4.045 / 0.0011 | 0.031 / 0.854 / 4.040 / 0.0001 |
| ctx | −0.095 / 0.889 / 1.697 / 0.0 | 同 real（context 是逐 episode 常数） |

漂移 5–7%（std），max|z| 与 frac>4 基本不变。结论：**注释与实现矛盾是确定的，数值影响在当前
checkpoint 上不大**；若 WM 更差/训练早期，漂移会放大。修法：用想象 batch 增量更新 `critic_norm`，
或改注释并明确“按真实分布归一化”的选择。

### P2-3 `state_norm`/`goal_norm` 的同类问题

`state_norm` 只在真实状态上拟合，却要在想象时归一化预测状态；`goal_norm` 在采集 goal 上拟合，
想象 goal 用预测的 root quat。机制与 P2-2 相同，无单独数值；用 P2-2 的 state 漂移做上界。

### P2-4 累计平均：不是硬冻结，而是“全部历史”记忆

`RunningNorm.update` 的合并权重是 `batch_count / total`；count 只增不减。默认 4096 env：

| iter | obs/goal 单次权重 | state/action/context 单次权重 | critic 单次权重 |
|---|---|---|---|
| 1,000 | 4.00e−4 | 1.00e−2 | 1.00e−2 |
| 10,000 | 4.00e−5 | 1.00e−3 | 1.00e−3 |
| 320,000 | 1.25e−6 | 3.13e−5 | 3.13e−5 |

对**持续**分布漂移，累计平均最终会跟上（100k 处阶跃到 320k 已吸收 68.8%）；但对**近期突变**阻尼很强
（310k 处阶跃到 320k 只吸收 3.1%）。没有 EMA/窗口/重置。这是设计风险而非数学错误；论文未提归一化。
建议：用 EMA（固定 horizon）或在阶段切换时重置 count，并在文档中写清。

### P2-5 `norm_interval` 期间的陈旧统计

`state/action/context` 与 `critic_norm` 每 10 iter 更新一次；期间最多 100 个 WM 梯度步 / 25 个控制步（每步 4096 env）。
相对 P2-4 的“全部历史”记忆，这点漂移可忽略；`goal_norm` 每步更新、`state_norm` 每 10 iter 更新，
两者频率不同但统计窗口都覆盖全部历史，影响小。

### P2-6 eps / clip

`std = (var + 1e-5).sqrt()`：var=0 时 std=0.00316，0.01 的偏差→z=3.16；`normalize` 不 clip。
实测真实/想象输入的 max|z|≈4.9，未出现极端值；若 WM 预测发散，归一化输入没有上限，只能靠
`torch.isfinite(loss)` 跳过更新。建议在 WM 输入/损失处加 clip 或 NaN 计数（与 P1 F9 一致）。

### P2-7 BPTT 穿过归一化（✓）

`RunningNorm.mean/var/count` 都是 buffer（`requires_grad=False`）；`update()` 在 `@torch.no_grad()`
里且 `x.detach()`；`normalize()` 对输入不做 detach。实测（`probe_norm_grad.py`）：

```
grad per dim       : [0.5014, 1.0, 1.98269]
expected (frozen=1): [0.5014, 1.0, 1.98269]
stats require_grad : [False, False, False]
```

即：统计量被当作常数（正确），输入梯度为 $1/\mathrm{std}$（未被切断）；frozen 维梯度为 1。
`WorldModel.forward`、`state_loss`、`termination_logits`、`ActorCritic.act/value` 都复用这个性质。

### P2-8 `library.dt` 与 `env.step_dt` 未绑定（潜在）

默认：`library.dt=0.02`（fps=50）、`step_dt=0.02`、`reward.dt=0.02`、`sampler.bin_size=50`，一致。
但代码没有断言，`_update_command` 固定“每控制步前进 1 帧”，`_finish_episodes` 用 `start_time+t`
（t=控制步）当帧偏移。若用 `--output-fps 30` 预处理或改 `decimation`：

```
reference speed factor = (1/30)/(1/50) = 1.667
goal window 10 frames = 0.333 s clip vs 0.200 s control
```

参考会以 1.667× 速度播放，reward 的“下一状态 vs ref_t”错位、bin 的 1 s 语义、`max_return` 的
折扣视界都会静默改变。修法（2026-11-10 已落地）：在 `MotionLibraryCommand.__init__` 断言
`abs(library.dt - env.step_dt) < 1e-6`。

### P2-9 `max_return` 与 reward 配置解耦

`max_return=10` 恰好等于 $0.1/(1-0.99)$（每步最大 reward 0.1 = dt×5 个权重之和）。若改 reward 权重、
dt 或 γ，代码不会重算，`softmax(-25G/10)` 的温度语义随之漂移。建议由 `TrackingRewardCfg`/γ 推导。

### P2-10 评估/部署：无噪 obs 用含噪统计

`obs_norm` 用训练时的带噪 obs 拟合；`play.py`/`evaluate.py` 调 `add_noise=False`，ONNX 部署时
也是干净观测。均匀噪声 ±a 的 std 为 $a/\sqrt3$，归一化后干净通道的压缩比：

```
joint_pos    noise_std=0.0058  clean_std=0.30 -> 0.9998
joint_vel    noise_std=0.2887  clean_std=1.00 -> 0.9608
base_ang_vel noise_std=0.1155  clean_std=1.00 -> 0.9934
gravity      noise_std=0.0289  clean_std=0.50 -> 0.9983
```

只有 joint-vel 通道有约 4% 的尺度差；属于“训练加噪、评估去噪”的正常代价，非 bug，但应在部署文档写明。

### P2-11 bias 切片（量纲错配）

`context` 布局是 `[μ(1), b(29), Δc(3)]`，所有消费者用 `context[:, 2:2+J]`：
joint 0 的 bias 丢失，末维把 COM x（DR ±0.025 m）当弧度加进 joint-28。这是本对账里唯一的“m→rad”
量纲错配，详见 P1 F4；修法一行 `[1:1+J]`，并加单测对比 mjlab 的 `joint_pos_rel(biased=True)`。

### P2-12 bin 单位（P1 F1）

`bin_frames = round(sampler.bin_size * library.dt)` = `round(50*0.02)` = 1；每帧写一次 bin，
`index_copy_` 只留最后一次，于是 $G_b$ 实际是 bin 末端回报。已由 `probe_finish_episodes_real.py`
与真实 `_collect`（visits=[83,73]）确认。

## 5. 已核对一致

- `env.step_dt = library.dt = reward.dt = 0.02`（默认配置）；`bin_size=50` 帧 = 1 s。
- reward 的六个 std、两个权重、$dt$ 缩放与论文 Table A.2 / mjlab tracking 逐项一致。
- $\gamma=0.99$、$\lambda=0.95$；`max_return=10` 由 $0.1/(1-\gamma)$ 推出。
- `state_norm` 冻结四元数维（WM/term head 原样使用，`state_loss` 单独处理）；
  `critic_norm` 不冻结，同一物理量在两个网络中归一化方式不同（设计选择）。
- 没有“归一化后的量再喂回 `update()`”的情况：所有 `update` 都接原始张量。
- ONNX 中烘入的 `obs_norm`/`goal_norm` 与 PyTorch 的 `RunningNorm` 逐位一致（`tests/test_export.py`，误差 <1e-4）。
- 观测/目标逐切片单位与布局：obs 93 = 29 rad + 29 rad/s + 3 rad/s + 3 无量纲 + 29 原始动作；
  goal 73 = 29 rad + 29 rad/s + 3 m + 6 无量纲 + 3 m/s + 3 rad/s。

## 6. 复现命令

```bash
cd /home/zy0028/mjlab_project/supermimic-mjlab
export MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src
PY=/home/zy0028/miniconda3/envs/tfbot_rl/bin/python

$PY scratch/audit_p1/probe_norm_budget.py        # P2-1/4/6/8/9/10 的数字
$PY scratch/audit_p1/probe_norm_grad.py          # P2-7
$PY scratch/audit_p1/probe_action_scale.py       # P2-1（需 GPU + checkpoint）
$PY scratch/audit_p1/probe_critic_norm_drift.py  # P2-2（需 GPU + checkpoint）
$PY scratch/audit_p1/probe_slice_scales.py       # 各切片量级
```
