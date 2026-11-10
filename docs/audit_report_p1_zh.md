# SuperMimic-mjlab 训练代码正确性审计（P1）

> 范围：`src/supermimic/algo` + `src/supermimic/mdp`（`scripts/` 仅在跨文件消费者需要时引用）。
> 环境：`tfbot_rl` conda（torch 2.14）。审计期间 GPU 变为可见（RTX 4090），
> 因此除静态/CPU 复现外，还跑了 `pytest` 全量与一次真实 `_collect` 探针（不训练）。
> 所有反例脚本在 `scratch/audit_p1/`，可直接重跑。
> 基线：`pytest tests -q -p no:warnings` → 30 passed（含 GPU smoke）。

## 0. 结论摘要

| 级别 | 编号 | 问题 | 位置 |
|---|---|---|---|
| 🔴 | F1 | bin 长度单位错用（frames × dt 当 frames），return 写进 bin 的是 **bin 末端**回报 | `trainer.py:319` |
| 🟡 | F2 | 同一 bin 的批量 EMA 用 `index_copy_`，重复索引只保留一个样本 | `sampler.py:106` |
| 🟡 | F3 | 想象 rollout 第一步把起点观测重复追加，历史窗口错位 | `trainer.py:529` |
| 🟡 | F4 | encoder-bias 切片 off-by-one：丢掉 joint 0，末维灌入 COM x | `trainer.py:203` 等 10 处 |
| 🟡 | F5 | 无干净窗口时 fallback 返回跨 episode 的窗口，静默污染 WM | `buffer.py:105` |
| 🟡 | F6 | critic 归一化器只在真实数据上拟合，critic 却在想象状态上训练/自举 | `trainer.py:423` |
| 🟢 | F7 | 真实/想象两侧 action_rate 的噪声方差不同（2σ² vs σ²） | `trainer.py:247` |
| 🟢 | F8 | `max_bins` 硬编码 1 s，`clip_and_bin` 无人调用（bin_size_s≠1 时失效） | `motion_library.py:132` |
| 🟢 | F9 | `nan_to_num` 把 NaN/Inf 静默变 0，掩盖 FK 发散 | `reward.py:201` |
| 🟢 | F10 | 多处 `torch.rand` 未传 generator，种子不可复现 | `buffer.py:97` |
| 🟢 | F11 | `MotionLibrary.fps` 被每个文件覆盖，混 fps 目录静默用最后一个的 dt | `motion_library.py:53` |
| 🟢 | F12 | checkpoint 不存 optimizer/buffer；`load` 对 WM 不检查缺失键 | `trainer.py:905` |
| 🟢 | F13 | env 自带 observation_manager 每步照算，但 trainer 另建一套特征 | `env_cfgs.py:110` |

论文原文加入后的补充发现见 §9（F14–F21）。

同时确认了以下"应该对"的地方确实对（第 5 节给证据）：FK vs MuJoCo、goal 与部署契约、
reward 与 mjlab tracking 的逐项对齐、reward 时序与上游一致、buffer 正常路径的窗口边界、
RunningNorm 合并公式、四元数误差公式、TD(λ) 目标递推、reset 处历史重建。

---

## 1. 数据流与单位（每步 $dt=0.02$ s，50 Hz）

| 阶段 | 张量 | 形状 | 单位 / 参考系 |
|---|---|---|---|
| sim | `root_link_pos_w` | $(N,3)$ | m，世界系 |
| WM state $s_t$ | `[p, q, v, ω, q_j, \dot q_j]` | $(N,71)$，$J=29$ | $p$ 为 env-local（减 `env_origins`）；$q$ 单位四元数 wxyz；$v,\omega$ 世界系 |
| context $\xi$ | `[μ, b, Δc]` | $(N,33)$ | $\mu$ 无量纲；$b\in\mathbb{R}^{29}$ rad；$\Delta c\in\mathbb{R}^3$ m |
| proprio $o_t$ | `[q_j-q_0+b, \dot q_j, R(q)^Tω, R(q)^Tg, a_{t-1}]` | $(N,93)$ | rad / rad·s⁻¹；$a$ 为**原始策略输出**（env 再乘 `G1_ACTION_SCALE`） |
| history | $o_{t-9..t}$，oldest→newest | $(N,10,93)$ | 同上 |
| goal $g_t$ | 10 帧 | $(N,730)$，$73=2J+15$ | 见第 2 节不变量 G；位置在 ref 帧 $t$ 的局部系，朝向相对机器人 root |
| buffer | `states/actions/contexts/rewards/dones/terminations/clip_ids/time_steps` | 每 env 1000 步 | reward 已乘 $dt$；`terminations` 只含硬终止 |
| WM 窗口 | $s_{t-8..t}$，$a_{t-8..t-1}$，$\xi$ | $(2048,9,71)$ 等 | 8 步自回归 |
| 想象 | $\hat s_t, a_t, g_t, r_t, d_t$ | $(4096,\cdot)$，$H=32$ | WM 参数冻结、可微 |
| critic | $[s,g,\xi]$ → $V$ | $(B,834)$ → $(B,)$ | 在**想象状态**上回归 TD(λ) |
| 导出 | $[o\vert g]$ → $a$ | $(930+730)$ → 29 | obs/goal 归一化与 tanh 烘进 ONNX |

关键常量：$\gamma=0.99$，$\lambda=0.95$，reward 以 $dt$ 缩放（与 mjlab `RewardManager(scale_by_dt=True)` 一致），
motion 库 $dt=1/50$，`env.step_dt=0.02`（`sim.timestep=0.005`，`decimation=4`）。

## 2. 不变量清单（组件 × 数学上必须成立的条件）

**S. 状态/上下文**
$$
s_t=[p_t,q_t,v_t,\omega_t,q^j_t,\dot q^j_t],\quad \|q_t\|=1,\quad \dim s=13+2J
$$
$$
\xi=[\mu,b,\Delta c],\quad b\in\mathbb{R}^{J},\quad \text{bias 的消费者必须取 } \xi[1:1+J]
$$

**O. 观测**（$b$ 加在关节位置通道，$a_{t-1}$ 是原始动作）
$$
o_t=\big[q^j_t-q_0+b,\ \dot q^j_t,\ R(q_t)^T\omega_t,\ R(q_t)^T g_w,\ a_{t-1}\big],\quad \dim o=3J+6
$$
历史必须是同一 episode 内连续的 10 帧；reset 后整窗填第一个观测。

**G. 目标**（$k=1..K$，$K=10$）
$$
g^{(k)}_t=\Big[q^{ref}_{t+k}-q_0,\ \dot q^{ref}_{t+k},\ R(q^{ref}_t)^T(p^{ref}_{t+k}-p^{ref}_t),\ \big[R(q_t)^TR(q^{ref}_{t+k})\big]_{:,1:2},\ R(q^{ref}_{t+k})^Tv^{ref}_{t+k},\ R(q^{ref}_{t+k})^T\omega^{ref}_{t+k}\Big]
$$

**R. 奖励**（$e$ 为平方误差，$\sigma$ 为容差；body 项按 yaw 对齐且 z 用参考 anchor 高度）
$$
r_t=\Delta t\Big[\sum_i w_i\exp(-e_i/\sigma_i^2)+w_{ar}\sum_j(a_t-a_{t-1})_j^2+w_{jl}\sum_j \mathrm{viol}_j\Big],\quad e_{ori}=4\arccos(\lvert q_1\cdot q_2\rvert)^2
$$

**B. 回放窗口**：长度 $k+1$ 的窗口不得跨 episode：
$$
\forall i\in[0,k-1]:\ \mathrm{done}(p-k+i)=0
$$

**W. 世界模型**：$\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi)$，四元数用增量旋转
$$
\hat q_{t+1}=\mathrm{normalize}\big(\exp(r_\omega)\otimes \hat q_t\big),\quad
\mathcal L=\mathrm{MSE}_{nonquat}(\hat s,s)+\ 4\,\mathrm{MSE}_{quat}(\hat q,q)+\mathrm{BCE}(d_\omega, d^{sim})
$$

**P. $\rho_0$（bins）**：每个 bin 维护**bin 起点**的折扣回报运行估计
$$
G_b=\sum_{t\ge bB}\gamma^{t-bB}\,r_t,\qquad B=50\ \text{frames}
P(b)=0.5\,U+0.4\,\mathrm{softmax}(-25G_b/G_{max})+0.1\,P_{past}
$$
（实现中 $t_b=$ bin 起点，$G_{max}=10$，EMA=0.1，每 rollout 每 bin 更新一次。）

**I. 想象与目标**（冻结 $\omega,\psi$；$S_t=\prod_{k<t}(1-d_k)$）
$$
\hat s_{t+1}=f_\omega(\hat s_t,\pi_\theta(o_{t-9..t},g_t),\xi),\quad
\hat J=\sum_{t=0}^{H-1}\gamma^tS_tr_t+\gamma^HS_HV_\psi(\hat s_H,g_H,\xi)
$$

**C. Critic**（`no_grad` 目标）
$$
G_t=r_t+\gamma(1-d_t)\big[(1-\lambda)V_{t+1}+\lambda G_{t+1}\big],\quad G_H=V_H
$$

**N. 归一化**：$z=(x-\mu)/\sqrt{\sigma^2+\varepsilon}$，running 统计按样本数合并；
$q$ 维在 `state_norm` 中冻结（不归一化）。

**E. 导出**：ONNX 图与 `ActorCritic.act` 逐位一致，obs/goal 统计量烘进图，encoder 的 tanh 保留，误差 $<10^{-4}$。

## 3. 核对结果表

| # | 不变量 | 违反点 | 数值反例 | 影响面 | 最小修法 | 回归断言 |
|---|---|---|---|---|---|---|
| F1 🔴 | P：bin 回报取 bin 起点 | `trainer.py:319` | 真实调用 `_finish_episodes` 后 `bin_returns=[4.02,1.50,0.079]`，正确应为 `[5.59,4.03,1.45]`；恰好等于 bin 末端 `[4.08,1.52,0.08]` | reset 分布整体后移约 1 s；末端 bin 权重约 ×31；`visits` 虚增 50× | `bin_frames = sampler.bin_size`（50） | 单 episode 后 `bin_returns[b] == G(t_b)` 且 `visits==1` |
| F2 🟡 | P：每个 rollout 每 bin 贡献一次 EMA | `sampler.py:106` | 同 bin 三个回报 `[1,2,3]` → 结果 0.30，期望 EMA(mean=2)=0.20 | 多 env 同 clip/bin 时只保留一个样本；CUDA 上顺序未定义 | 先按 bin 聚合 mean，再 EMA | 两 env 同 bin：`G==0.9·old+0.1·mean` |
| F3 🟡 | O/I：想象起点历史应等于采集时的历史 | `trainer.py:529` vs `457-481` | buffer 历史末端已等于 `obs(s_t)`，循环再 append 一次 → `[o_{t-8..t},o_t]`；训练后 actor 动作差 mean=0.0200（RMS 1.133 的 1.8%） | 每个 rollout 前 ~9/32 步的 actor 输入与真实分布不一致 | t=0 直接用现成 history；或让 `_history_from_buffer` 只到 $s_{t-1}$ | 第一步后 `history[-1] != history[-2]`（无重复） |
| F4 🟡 | S：bias 取 $\xi[1:1+J]$ | `trainer.py:203,239,296,462,527`；`play.py:117,209,243`；`evaluate.py:83,127` | `context[:,2:31]` = `bias[1:]` + `com_x`；joint 0 的 bias 永不使用，末维=0.012 m 当弧度用；actor 动作差 mean=0.0217（3.3% RMS） | 观测语义错位；COM x 通过 joint-28 通道泄漏；部署端输入语义不同 | 全部改为 `[1:1+J]`，抽 `context_bias()` 单一入口 | `obs == mjlab joint_pos_rel(biased=True)`；末维不含 com |
| F5 🟡 | B：窗口不得跨 episode | `buffer.py:105-109` | done 每 2 步时 16/16 个返回窗口都含 done 转移，末端固定 11 | 频繁摔倒的 env 的 WM 训练数据在 reset 处拼接；无告警 | 无效则重采样/丢弃该行，或断言 | 对抗 done 模式下返回窗口 `dones[:,:k].any()==False` |
| F6 🟡 | N/C：critic 输入统计应覆盖 critic 实际输入分布 | `trainer.py:423-441` vs `646-650` | 注释称"bootstrap 在 replay buffer 上评估"，代码在 `chunk_states`（想象状态）上取 V | horizon 越长漂移越大；value 外推；文档与实现矛盾 | 每次更新用想象 batch 增量更新 `critic_norm`，或改注释并说明取舍 | 统计更新来源断言 + 文档一致 |
| F7 🟢 | R：两侧 action_rate 同分布 | `trainer.py:247-255` vs `541-550` | 真实：E‖Δa‖²+2σ²；想象：+σ²；σ=0.1, w=-0.1, dt=0.02 → 2e-5/步 | 可忽略（reward≈0.05/步） | 想象侧对 a 加同分布噪声，或 buffer 存无噪动作 | 两侧 action_rate 期望差 `<1e-3` |
| F8 🟢 | P/库：bin 尺寸唯一来源 | `motion_library.py:132-140` | `max_bins` 硬编码 1.0 s，`clip_and_bin` 无调用者 | `bin_size_s≠1` 时 `max_bins` 与 sampler 不一致 | 删除或参数化 | 构造 `bin_size_s=0.5` 的库断言 |
| F9 🟢 | R：NaN 是错误信号 | `reward.py:201` | FK 出现 NaN 时 reward 变 0，训练继续 | 静默掩盖数值发散 | 计数/log NaN，或断言 finite | 注入 NaN 状态，断言抛错/计数>0 |
| F10 🟢 | 可复现性 | `buffer.py:97,118,141`；`sampler.py:87`；`trainer.py:247` | 固定 seed 后两次采样序列不同 | 复现/回归测试不稳定 | 统一传 `generator=self._rng` | 同 seed 两次采样张量相等 |
| F11 🟢 | E/库：dt 唯一 | `motion_library.py:53,71` | 混 fps 目录时 `dt` 取最后一个文件 | 动作速度与 $dt$ 错配 | 断言所有文件 fps 相同 | 混 fps 输入报错 |
| F12 🟢 | 训练连续性 | `trainer.py:905-931` | `save` 无 optimizer/buffer；`load` 对 WM `strict=False` 不报告 | resume 后动量丢失、buffer 空（已由 warmup 兜底） | 存 optimizer state；WM 也检查 missing | 断点续训前后 lr/梯度一致 |
| F13 🟢 | 单一特征定义 | `env_cfgs.py:110-140` vs `trainer.py:95-104` | 两套 obs/goal 定义并存，F4 已导致漂移 | 计算浪费 + 维护风险 | 删掉 env observation 或让 trainer 复用 | 两套特征逐位相等断言 |

## 4. 发现详情

### F1 🔴 `_finish_episodes` 的 bin 长度单位错用

`sampler.bin_size` 已经是**帧数**（`round(1.0/0.02)=50`），代码却乘了 `library.dt`：

```python
bin_frames = max(1, round(self.command.sampler.bin_size * self.library.dt))  # 50*0.02 -> 1
for t in range(0, length, bin_frames):                                     # 每帧都写
    time_list.append(start_time + t)
```

后果有两层：
1. 每个 bin 被写 50 次；`update_returns` 用 `index_copy_`，重复索引**只保留最后一次**，
   于是 `G_b` 实际是 bin **末端**帧的回报，而不是 bin 起点的回报（整体后移 1 s）。
2. 每次写都做一次 EMA，但只有最后一次生效；若未来把 `index_copy_` 改成 `index_add_`，
   会变成"50 次 EMA 的几何平均"，同样不是论文定义的 bin 起点回报。

数值（`scratch/audit_p1/probe_finish_episodes_real.py`，直接调用真实方法，$r=0.08$/步，$L=120$，收敛 40 个 episode）：

| bin | 正确 $G$（起点） | 实测 | bin 末端 $G$ |
|---|---|---|---|
| 0 | 5.605 | 4.021 | 4.081 |
| 1 | 4.041 | 1.500 | 1.522 |
| 2 | 1.457 | 0.079 | 0.080 |

`visits` 变成 `[2000,2000,800]`（应为 `[40,40,40]`）。在 `softmax(-25G/10)` 下，
末端 bin 的权重相对正确值约 ×31，采样会偏向更难/更靠后的起点。

真实管线复核（`probe_runtime_bins.py`：4 env、`episode_length_s=0.4`、44 步 `_collect`，
共 10 个 episode、总长 156 帧）：`visits=[83,73]`，正确应为 `[≤10,≤10]`；
即每帧都往 bin 里写一次，与 stub 结论一致。

**最小修法**：`bin_frames = int(self.command.sampler.bin_size)`（并让 `ReturnBasedSampler`
暴露一个 `bin_frames` 属性，避免两处各算一遍）。

**回归断言**：构造单 episode，断言每个 bin 更新后 `bin_returns[b] ≈ G(t_b)`、`visits[b]==1`；
变异验证：把 `bin_size` 改回 `bin_size*dt` 后测试必须失败。

### F2 🟡 同一 bin 的批量 EMA 不是均值

```python
self.bin_returns.index_copy_(
    0, bin_ids, self.bin_returns[bin_ids] * (1 - ema) + values * ema)
```

`bin_ids` 有重复时（多个 env 在同一 collect 调用里结束、落在同一 clip/bin），
`index_copy_` 的语义是"后写覆盖先写"，只有最后一个样本生效；CUDA 上顺序未定义。
`scratch/audit_p1/probe_index_copy.py`：`[1,2,3] → 0.30`，正确应为 `0.20`。
与 F1 叠加时，单 episode 内同一 bin 就有 50 个重复索引。

**修法**：`scatter_reduce_` 求每个 bin 的 batch mean 与 count，再
`G = (1-ema)·G + ema·mean`（或对每个样本顺序 EMA，但不能用 `index_copy_`）。

**断言**：两个 env 同 bin、回报 1 和 3 → `G == 0.9·old + 0.1·2`。

### F3 🟡 想象 rollout 第一步重复起点观测

`_history_from_buffer` 返回的是 $o_{t-9..t}$（末端就是 `obs(s_t)`），
但 `_update_policy` 在每一步（包括 $t=0$）都：

```python
obs = self.proprio(state, prev_action, ...)
history = torch.cat([history[:, 1:], obs[:, None]], dim=1)
action = self.act(history.reshape(...), goal)
```

于是第一个动作看到的是 $[o_{t-8},\dots,o_t,o_t]$：最近一帧重复、最早一帧丢失；
重复会随窗口左移保留约 9 步（$H=10$），即前 9/32 步的 actor 输入与采集分布不一致。

`scratch/audit_p1/probe_history_dup.py` 复现了 `history[-1] == obs(s_t)` 与
append 后 `[5,6,7,8,8]`。用 34k checkpoint 量化（`probe_dup_impact.py`）：
动作平均差 `0.0200`，action RMS `1.133`（约 1.8%）。

**修法**：$t=0$ 直接 `act(history, goal)`，从 $t=1$ 起再 append；或让
`_history_from_buffer` 只取到 $s_{t-1}$。

**断言**：想象第一步的 `history` 与采集端在 $s_t$ 的历史逐位相等（无重复帧）。

### F4 🟡 encoder-bias 切片 off-by-one（10 处：trainer 5、play 3、evaluate 2）

`StateExtractor.context` 的布局是 `[μ (1), b (J), Δc (3)]`（`state.py:182`），
但所有消费者用 `context[:, 2:2+J]`：

```python
bias = context[:, 2 : 2 + self.spec.num_joints]
```

`scratch/audit_p1/probe_context.py`（$J=29$）：取到 `bias[1:]` + `com_x`，
joint 0 的 bias 永远丢失，末维变成 $0.012$ m 的 COM 偏移当弧度。
`probe_bias_impact.py`：训练后 actor 的动作差 mean `0.0217`（RMS 0.657 的 3.3%）。

影响不止"少一个 bias"：COM x 是每个 episode 固定的域随机化参数，经此通道泄漏进观测，
策略可能把它当"特权特征"；部署端该通道是真实关节编码器偏置，语义不一致。

**修法**：统一 `context[:, 1:1+J]`；更稳的是在 `StateExtractor` 上加
`encoder_bias_from_context(ctx)`，禁止在 6 个地方手写切片。

**断言**：`ProprioFeatures(state, a, bias_from_context)` 与 mjlab
`joint_pos_rel(biased=True)` 逐位相等；末维不得含 COM。

### F5 🟡 `sample_windows` 的 fallback 破坏 episode 边界

正常路径（32 次拒绝采样）是对的（`tests/test_buffer.py` + 200 轮随机 done 模式验证 0 越界）。
但当某 env 的 buffer 里**所有**长度 $k+1$ 窗口都含 done 时，fallback 直接返回 `ptr-1`：

```python
if not bool(valid.all()):
    positions[missing] = (self.ptr[e] - 1) % self.capacity
```

`scratch/audit_p1/probe_fallback.py`（done 每 2 步）：16/16 个窗口跨 reset，
WM 会把不同 episode 的 $s_{t+1}$ 当成 $f(s_t,a_t)$ 的目标。真实训练里只在
"摔倒极频繁/早期"时触发，但触发时静默。

**修法**：无效则重采样 env 行；若仍无效则丢该行并在日志里计数，绝不返回坏窗口。

**断言**：对抗 done 模式下，返回的窗口要么为空，要么 `dones[:, :k].any()==False`。

### F6 🟡 critic 归一化器与 critic 实际输入分布不一致

`_update_critic_normalizer` 只在真实 buffer 转移上拟合（`trainer.py:431-441`），
注释解释为"bootstrap 在这些状态上评估"；但 `_update_critic` 的 `states` 全部来自
`chunk_states`（想象轨迹），critic 也在这些想象状态上回归。结果是
critic 用真实状态的均值/方差去归一化随 horizon 漂移的想象状态。

**修法**：在 `_update_critic` 里用同一批想象状态增量更新 `critic_norm`
（或明确选择"只按真实分布归一化"并改注释、加 horizon 监控）。

**断言**：记录 `critic_norm` 的更新来源 batch id；跑 100 iter 后比较真实/想象状态的
归一化后均值与方差，偏差超过阈值即失败。

### F7–F13（🟢，简述）

- **F7**：采集端 reward 的 action_rate 用 `(a+ε_t)-(a_{t-1}+ε_{t-1})`，期望多出 $2\sigma^2$；
  想象端只有 $a_t-(a_{t-1}+\epsilon)$，多出 $\sigma^2$。量级 $2\sigma^2\vert w\vert dt\approx2\times10^{-5}$/步，
  相对每步 reward ~0.05 可忽略；若要严格一致，想象侧也加同分布噪声。
- **F8**：`max_bins = ceil(max_len*dt/1.0)` 与 `clip_size_s` 无关，且 `clip_and_bin` 无调用者；
  `bin_size_s≠1` 时会与 sampler 不一致。
- **F9**：`nan_to_num(...)=0` 把 NaN/Inf 变成合法 0 奖励，训练继续但不收敛；建议计数并告警。
- **F10**：`torch.rand` 未传 generator（buffer/sampler/action noise），固定 seed 不能复现采样流。
- **F11**：`MotionLibrary.fps` 在循环里被覆盖；混 fps 目录时 `dt` 取最后一个文件。
- **F12**：`save` 不含 optimizer state 与 buffer；`load` 对 world_model 用 `strict=False` 且不检查缺失键。
- **F13**：`env_cfgs` 的 observation_manager 每步照算 proprio/goal（含 10 帧历史），
  但 trainer 用自己的 `ProprioFeatures/GoalFeatures`；两套定义已经因 F4 漂移。

## 5. 已核对为正确的项（附证据）

1. **可微 FK**：`kinematics.py:219-324` 与 MuJoCo 的 `xpos/xquat/cvel` 在 G1 上
   位置/朝向/线速度/角速度误差 $<10^{-5}$（`tests/test_kinematics.py::test_fk_matches_mujoco`，
   `pytest` 通过）。hinge 的 `anchor - R_new·jnt_pos`、slide 的 `axis·q`、速度递推均与 `mj_kinematics` 一致。
2. **Goal 与部署契约**：`features.GoalFeatures` 的 10×73 布局、帧偏移 $t+1+k$、
   `R(q^{ref}_t)^T(p_k-p_t)`、前两列朝向、参考系速度，与 `deploy_features.build_goal`
   逐位一致（`tests/test_deploy_features.py` 全过；含 clip 末端 clamp 与 yaw offset 用例）。
3. **Reward 对齐 mjlab tracking**：`reward.py` 的 6 个指数项、$\sigma$、权重、$dt$ 缩放、
   `action_rate` 的 raw-action 定义、`joint_limit` 的正违反和，与
   `mjlab/tasks/tracking/mdp/rewards.py` + `tracking_env_cfg.py` 逐项相同；两处差异
   （去自碰撞、joint_limit $-10\to-2$）与论文声明一致（`tests/test_paper_params.py` 也钉住了数值）。
4. **Reward 时序与上游一致**：mjlab `env.step` 中 `reward_manager.compute`（`manager_based_rl_env.py:456`）
   在 `command_manager.compute`（`:486/488`）之前，所以上游也是 $r(s_{t+1}, ref_t)$；
   trainer 在 step 前 clone `clip_ids/time_steps` 再算 reward（`trainer.py:248-254`）与之一致，
   想象侧同样用 `next_state` + `ref_t`（`trainer.py:532-550`），两侧没有互相错位。
5. **reset 处历史重建**：`_history_from_buffer` 在窗口内含 reset 时给出
   `[o_{reset}×3, o_{t-1}, o_t]`，与采集端"reset 时整窗填同一个 obs"的语义一致
   （`probe_history3.py` 的期望/实测完全相同）。
6. **窗口采样正常路径**：`sample_windows` 对 done 的检查覆盖 $p..p+k-1$，
   环形回绕与 `count<capacity` 两种情形都正确（`tests/test_buffer.py` + 200 轮随机 done 模式 0 越界；
   仅 F5 的 fallback 例外）。
7. **RunningNorm**：合并公式与 `torch.cat` 全量统计一致（mean 误差 $<5\times10^{-7}$，
   var 误差 $<2\times10^{-6}$，归一化后 mean $\approx0$、std $\approx1$）；
   `frozen` 维度（四元数）不归一化。
8. **四元数误差**：`4·acos(|dot|)^2 = θ^2`，clamp 到 $1-10^{-6}$ 后梯度有界（$|\partial e/\partial d|\le 8$），
   不会在完美跟踪处发散。
9. **`_rotvec_to_quat`**：小角分支用 $0.5-\theta^2/48$，输出重新归一化，原点处光滑且单位模。
10. **TD(λ) 目标**：`trainer.py:646-657` 的递推
    $G_t=r_t+\gamma(1-d_t)[(1-\lambda)V_{t+1}+\lambda G_{t+1}]$、$G_H=V_H$ 与定义一致；
    `values/targets` 都在 `no_grad` 下算。
11. **想象目标**：$\sum\gamma^tS_tr_t+\gamma^HS_HV(\hat s_H,g_H,\xi)$，$S_t$ 在 $r_t$ 之后才乘
    $(1-d_t)$，与"终止发生在该步之后"一致；WM/critic 参数被 `requires_grad_(False)`，
    actor 梯度只沿 action→FK→reward/termination→回报 传播。
12. **chunk 等价性**：每个 chunk 的 `loss` 乘 `weight=(chunk/total)`，
    梯度求和等于整批均值（结构上成立；`bad_grads` 会阻止半步更新）。
13. **episode 记账**：`episode_start_clip/time` 与 `_finish_episodes` 的 `start_time+t`
    对齐；`episode_len` 在 reset 前读取；`terms` 的 `done` 与 `terminations` 分离，
    终止头只学硬终止（time-out 不算）；但其输入/标签与论文 Eq. 1–3 的错位见 F14。
14. **PNG/ONNX 导出**：`export.py` 从 state dict 推断形状并在加载时校验；
    `tests/test_export.py` 的 onnxruntime 对比通过（阈值 $10^{-4}$）。

## 6. 原“未定项”的结论（2026-11-10 论文原文补充）

> `docs/SuperMimic.pdf` 已加入仓库；三项旧问题的核对见 §9.1。原来的三个问题保留如下供追溯，
> 结论一律以 §9 为准。

| 原问题 | 论文原文结论 | 代码判定 |
|---|---|---|
| U1 想象起点 $s_0\sim\rho_0$ 是否 return-based | Algorithm A.1 第 5 行只写 "sample ... a start state from B"；D.1 的 return-based 采样只描述 **simulator episodes reset** | 代码均匀抽 buffer 状态**不违背论文**；`docs/paper_reading_zh.md:815` 的"ρ0 = return-based 混合"是误读 |
| U2 BONES-SEED 关节顺序 | 数据层 | CSV 表头 29 列与 entity `joint_names` **逐项一致**（`scratch/audit_p1/probe_joint_order.py`），`preprocess.py:148` 正确 |
| U3 "UTD 0.4" 语义 | Table A.5：一次更新 = 10 batches × 2048 windows；1 policy update per WM update；UTD 0.4 | 代码 2.5 env steps/iteration + 10 WM minibatch + 1 次 policy update，在"update = 一次 WM 拟合"的解读下**一致** |

## 7. 复现命令

```bash
cd /home/zy0028/mjlab_project/supermimic-mjlab
export MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src
PY=/home/zy0028/miniconda3/envs/tfbot_rl/bin/python

$PY -m pytest tests -q

$PY scratch/audit_p1/probe_finish_episodes_real.py   # F1（stub）
$PY scratch/audit_p1/probe_runtime_bins.py           # F1（真实 env + _collect, 需 GPU）
$PY scratch/audit_p1/probe_index_copy.py             # F2
$PY scratch/audit_p1/probe_history_dup.py            # F3 机制
$PY scratch/audit_p1/probe_dup_impact.py             # F3 量化（需 logs/.../checkpoint_0034000.pt）
$PY scratch/audit_p1/probe_context.py                # F4 机制
$PY scratch/audit_p1/probe_bias_impact.py            # F4 量化
$PY scratch/audit_p1/probe_fallback.py               # F5
$PY scratch/audit_p1/probe_joint_order.py            # U2
$PY scratch/audit_p1/probe_eval_friction.py          # F20（需 GPU）
$PY scratch/audit_p1/probe_norm_buffer.py            # 归一化 + 正常窗口边界
$PY scratch/audit_p1/probe_history3.py               # reset 历史重建（应一致）
```

## 8. 建议修复顺序

1. F1（🔴，一行 + 单测）；2. F4（10 处切片 + 统一入口）；3. F2（EMA 聚合）；
4. F3（第一步不 append）；5. F5（掉坏窗口）；6. F6（critic 归一化）；
7. F7–F13 按需。

修完后重跑第 7 节脚本与 `pytest`，并补 4 个回归断言（F1/F2/F3/F4）。

---

## 9. 论文原文补充核对（`docs/SuperMimic.pdf`，2026-11-10）

### 9.1 三项旧“未定”的结论

| 旧编号 | 论文原文 | 代码判定 |
|---|---|---|
| U1 想象起点 $s_0\sim\rho_0$ | Algorithm A.1 第 5 行：“Sample an interface $i\sim p(I)$ and a **start state from B**”；Appendix D.1 的 return-based 采样只描述 **simulator episodes reset** | 代码 `trainer.py:444` 均匀抽 buffer 状态**不违背论文**；`docs/paper_reading_zh.md:815` 的“ρ0 = return-based 混合”是无依据引申 |
| U2 BONES-SEED 关节顺序 | 数据集层，论文不涉及 | CSV 表头 29 列与 `Entity(get_g1_robot_cfg()).joint_names` **逐项完全一致**（`scratch/audit_p1/probe_joint_order.py`），`preprocess.py:148` 不重排正确 |
| U3 “UTD 0.4” | Table A.5：“10 batches of 2048 windows per update”“Policy updates per world-model update 1”“Update-to-data ratio 0.4” | 代码 2.5 env steps/iteration、10 个 WM minibatch、1 次 policy update，在“一次 update = 一次 WM 拟合”的解读下**一致** |

### 9.2 新增发现

| 级别 | 编号 | 问题 | 位置 |
|---|---|---|---|
| 🟡 | F14 | 终止头在 $s_t$ 上评估；论文 Eq. 2/3 用 $s_{t+1}$ 的 $\hat d_{t+1}$。标签又是“下一状态终止”，导致动作边缘化，终端态没有训练样本 | `trainer.py:265,374,525` |
| 🟡 | F15 | Eq. 1 是 $\sum_{j=1}^k$，代码用 $\frac{1}{k}\sum$，BCE 相对权重 $\times k$；state 项改为 running-normalized MSE + $4\times$ quat | `trainer.py:358-362,377`；`networks.py:212-224` |
| 🟡 | F16 | 训练回合固定 10 s 截断；论文未规定回合长度，clip 可达 30 s | `env_cfgs.py:69`、`train.py:28` |
| 🟡 | F20 | 评估脚底摩擦 0.6（G1 XML 默认），论文 F.1 要求 1.0 | `evaluate.py:48` |
| 🟡 | F21 | 评估按 start 模式随机**有放回**抽 clip；论文 F.1 要求每条 motion 跑一次 | `evaluate.py:52-55` |
| 🟢 | F17 | goal 每帧 73 维（含 6 维参考速度）；论文只写 K 帧参考序列，未给逐维；代码自述比 67 维多 6 维 | `features.py:116,130-137` |
| 🟢 | F18 | reset 的 pose/velocity/joint 扰动沿用 mjlab tracking 默认，论文 Appendix A / Table A.3 未列 | `env_cfgs.py:88-97` |
| 🟢 | F19 | obs/goal/state/action/critic 的 running normalization、term-head bias 5% 初始化，论文未提 | `networks.py:37,155` |

### 9.3 F14 细节：终止头与 Eq. 1–3 的状态错位

论文（§3.1 + Eq. 3）：
$$
d_t=1 \iff s_t\text{ terminal},\qquad \hat d_t=d_\omega(\hat s_t,g_t),\qquad c_{t+1}=c_t(1-d_{t+1})
$$
$$
\hat R_t=\hat r_{t+1}+\gamma(1-\hat d_{t+1})\big[(1-\lambda)V_{t+1}+\lambda \hat R_{t+1}\big]
$$

代码：
- 采集时 `terminated` 来自 `env.step`，mjlab 在 step 之后计算，因此它是 **$s_{t+1}$ 的终止**；该标签存在 buffer 位置 $t$（`trainer.py:265`）。
- 终止头训练：`termination_loss(term_states=s_t, label=terminations[t])`（`trainer.py:374-376`），
  即 $\hat d(s_t,g_t)\approx P(s_{t+1}\text{ terminal}\mid s_t)$。
- 想象时：`term_prob = termination_prob(state, goal)` 在 $s_t$、**动作之前**计算（`trainer.py:525`），
  再作为 $1-\hat d$ 进入 Eq. 2/3。

所以论文的 $\hat d_{t+1}$ 在 $s_{t+1}$（依赖刚执行的动作）上评估；代码是在 $s_t$ 上评估的
**动作边缘概率**。两个后果：

1. Eq. 2/3 的生存权重缺少 $\partial \hat d/\partial a_t$ 的直接路径（只剩经 $s_t$ 的间接路径）；
2. buffer 只存 pre-action 状态，终端状态 $s_T$ 从未作为输入出现，$\hat d(s_T)$ 没有训练信号；
   Eq. 1 的正样本也只是“下一状态终止”标签，而不是“当前状态终止”。

**修法**：WM step 之后在 $(s_{t+1},g_{t+1})$ 上算 $\hat d$（critic target 同样），并让终止头使用
状态终止标签（需要把 terminal state 存入 buffer 或单独缓存）。若保留 transition-label 约定，
至少要在文档/对照表里声明它与论文 Eq. 1 的定义不同。

置信度：代码语义高；影响量级中等（生存权重被平滑，跌倒的即时梯度变弱）。

### 9.4 F15 细节：Eq. 1 的 mean vs sum 与度量替换

论文 Eq. 1：
$$
L_{WM}(\omega)=\mathbb{E}_{\mathcal B}\Big[\sum_{j=1}^{k}\big(\|\hat s_j-s_j\|^2+(\hat r_j-r_j)^2+\|\hat{\bar o}_j-\bar o_j\|^2+\mathrm{BCE}(\hat d_j,d_j)\big)\Big]
$$

代码（`trainer.py:358-362,377`）：
```
loss = 0
for j in range(k):
    loss = loss + state_loss(pred, states[:, j + 1])
loss = loss / k
...
loss = loss + 1.0 * term_loss
```

- 论文是 $\sum_{j=1}^k$，代码是 $\frac{1}{k}\sum$：state 项总权重变成 1，BCE 相对权重被放大 $k=8$ 倍。
  若严格按 Eq. 1，应 `loss = Σ state_loss + BCE`（或把 `wm_term_loss_weight` 设为 $1/k$）。
- `state_loss` 不是原始平方和，而是 running-normalized MSE（四元数用 $4\times$ sign-invariant 项，
  `networks.py:212-224`）；这会改变位置/关节/速度各维的相对权重。作为工程度量可以接受，但论文没有这一层，
  需要在“与 Eq. 1 对齐”的说明里写明。
- BCE 用的还不是窗口内的 $j$ 标签，而是另一批 `sample_random` 状态（见 F14），所以“BCE 权重”比较
  只在每样本层面成立。

### 9.5 F16–F21 简述

- **F16（🟡）**：`episode_length_s=10.0`（`env_cfgs.py:69`，`train.py:28` 也可覆盖）。论文没有固定回合长度，
  只说 reset 到 motion 库的点；clip 2–30 s。>10 s 的 clip 在训练里被 time-out 截断，`motion_end` 永不触发，
  尾段只能靠 reset 采样覆盖。评估用 60 s，不受影响。
- **F20（🟡）**：`evaluate.py:48` 设 `env_cfg.events={}`，实测脚底摩擦为 **0.6**（G1 XML 默认，
  `probe_eval_friction.py`）；论文 F.1 的 benchmark 明确写 foot friction **1.0**。这会改变失败数/回报，
  不能直接对比 Table A.8。
- **F21（🟡）**：`evaluate.py:52-55` 用 `sampling_mode="start"`，而 `_resample_command`（`commands.py:285`）
  对 clip 做 `torch.randint` **有放回**采样；默认 64 env × 4 episodes = 256 次，远小于 129,514 条 benchmark，
  对 40 条 LAFAN1 也会重复/漏掉。README 的“one rollout per motion”不成立。
- **F17（🟢）**：goal 帧 $2J+15=73$；论文只写 “K=10 future reference frames”，没有逐维布局。
  代码 docstring 自称比 67 维多 6 维参考速度；若是原实现的 67，则这是**文档化扩展**，不是 bug，
  但 encoder 输入从 670 变 730，部署契约随之改变。
- **F18（🟢）**：reset 的 `pose_range`/`velocity_range`/`joint_position_range`（`env_cfgs.py:88-97`）来自
  mjlab tracking 默认，论文 Appendix A/Table A.3 未列；它不是回合内 push，但“resets to points of the
  motion library”并不等于精确复位。若严格复现论文，应确认这些是否属于论文实现。
- **F19（🟢）**：running normalization、frozen quat dims、term-head bias 5% 初始化，论文未提；
  F6 的 critic-norm 真实/想象不一致仍然独立存在。

### 9.6 论文已确认一致的部分

- **Algorithm A.1 顺序**：collect → fit WM → improve policy（含 critic）✓ `trainer.py:713-744`。
- **Table A.5 全部关键值**：$k=8$、10×2048、$\alpha=2\times10^{-4}$、$H=32$、batch 4096、
  grad clip 1 + elementwise 0.1、$\gamma=0.99$、$\lambda=0.95$、critic 5×4、4096 envs、
  $\epsilon\sim\mathcal N(0,0.1^2)$、buffer 1000、UTD 0.4、1 policy/WM update、
  cosine 250k→0.03、320k iterations ✓（`config/g1/rl_cfg.py`、`tests/test_paper_params.py`）。
- **Eq. 2/3 结构**：survival $c_t$、bootstrap $\gamma^H c_H V$、TD(λ) 目标只算一次、
  无 EMA target、critic 只在想象转移上拟合 ✓（`trainer.py:560-566,646-657`）。
- **Eq. 1 的 head 政策**：论文允许 analytic head 无 loss；代码 reward/obs analytic、state+BCE learned ✓。
- **Table A.1**：obs 93 维、均匀噪声 ±0.01/±0.5/±0.2/±0.05 ✓（`features.py:37-43`）。
- **Table A.2**：6 个指数项、$\sigma$、权重、$dt$ 缩放、整体 mean-over-bodies、去自碰撞、
  joint-limit −2 ✓（`reward.py`，逐项对照 mjlab tracking）。
- **Appendix A 终止**：$\lvert e_{p,z}\rvert>0.25$、$e_{\text{tilt}}>0.8$、无 EE 终止 ✓（`env_cfgs.py`）。
- **Table A.3**：$\mu\sim U[0.3,1.2]$、$\Delta q^0\sim U[-0.01,0.01]$、
  $\Delta c=(0.025,0.05,0.05)$ ✓；无 push 事件 ✓。
- **Action** $q^{sp}=\alpha a+q^0$ ✓；**goal** $K=10$ ✓；**observation history** 10 ✓。
- **Evaluation** 无 DR/无观测噪声/确定性/60 s ✓（除 F20/F21）。
- **BONES-SEED 关节顺序** ✓（CSV 表头比对）。

### 9.7 修订后的修复优先级

F1（🔴）→ F14（🟡，目标函数语义）→ F15（🟡，loss 平衡/度量）→ F4 → F2 → F3 → F5 → F6
→ F20/F21（评估可比性）→ F16 → 其余。

---

## 10. 修复记录（2026-11-10）

> 本节对应上面的 P1 报告（§0–§9）。修复前的位置引用保留在正文里；修复后行号可能移动。
> 验证：`pytest tests -q` → **37 passed**（含 GPU smoke）；`evaluate` 在 2 条 clip 上跑通；
> 真实 `_collect`/强制摔倒训练/采样探针均重跑通过。

### 10.1 已修复

| 编号 | 修改 | 文件 |
|---|---|---|
| F1 🔴 | `_finish_episodes` 改用 `sampler.bin_frames`（帧数），不再乘 `library.dt`；新增 `bin_frames` 属性 | `algo/sampler.py`、`algo/trainer.py` |
| F2 🟡 | `update_returns` 先按 bin 聚合 `mean` 再一次 EMA，杜绝 `index_copy_` 重复索引覆盖 | `algo/sampler.py` |
| F3 🟡 | 想象循环 $t=0$ 不再 append 起点观测（buffer 历史已含 $o_t$），$t>0$ 才 append | `algo/trainer.py` |
| F4 🟡 | 新增 `context_encoder_bias()`（取 `context[..., 1:1+J]`），替换 trainer 5 处、play 3 处、evaluate 2 处 | `algo/state.py`、`trainer.py`、`play.py`、`evaluate.py` |
| F5 🟡 | `sample_windows` 不再 fallback 到跨 reset 的窗口，改为丢弃无效行；WM 训练/误差统计跳过空 batch | `algo/buffer.py`、`algo/trainer.py` |
| F6 🟡 | 修正 `_update_critic_normalizer` 注释：明确统计只拟合真实转移，而 critic 在想象状态上训练/自举，并给出实测漂移 | `algo/trainer.py` |
| F8 🟢 | `MotionLibrary.max_bins` 改为 `max_bins(bin_size_s=1.0)`，不再硬编码 1 s | `mdp/motion_library.py` |
| F10 🟢 | `sample_windows`/`sample_starts`/`sample_random`/`sample_terminal` 接受并传递 `generator`；action noise 用 trainer RNG | `algo/buffer.py`、`algo/trainer.py` |
| F11 🟢 | 混 fps 的 motion 目录直接 `ValueError`，不再静默用最后一个文件的 dt | `mdp/motion_library.py` |
| F12 🟢 | checkpoint 保存三个 optimizer 的 state；load 时恢复，并报告 world_model 的 missing/unexpected keys | `algo/trainer.py` |
| F14 🟡 | 新增 hard-terminal state 环缓（`terminal_states/clip_ids/time_steps`）；终止头改为“输入状态是否 terminal”的 BCE（正样本来自该缓冲，负样本为普通 buffer 状态）；想象/critic target 改为在 $s_{t+1}$ 上算 $\hat d$ | `algo/buffer.py`、`algo/trainer.py` |
| F15 🟡 | WM loss 的终止项改为 `term_loss / k`，与论文 Eq. 1 的 $1/k$ 相对权重一致（state 项仍是 mean） | `algo/trainer.py` |
| F16 🟡 | 训练初始化时若最长 clip > `episode_length_s`，打印明确警告（不改变默认 10 s 行为） | `algo/trainer.py` |
| F20 🟡 | evaluate 增加 `eval_friction` reset 事件，把脚底摩擦固定为 1.0（与论文 F.1 一致） | `scripts/evaluate.py` |
| F21 🟡 | 新增 `sampling_mode="cycle"` 和 `seen_clips` 去重：评估按顺序扫库，每条 motion 只记录第一次完成的 rollout | `mdp/commands.py`、`scripts/evaluate.py` |

回归断言新增：
- `tests/test_sampler.py`：F1（bin 帧数 + 单 episode 一 bin 一次）、F2（重复 bin 聚合）。
- `tests/test_state.py`：F4（bias 切片 + proprio 加 bias）。
- `tests/test_buffer.py`：F14（terminal state 存储/采样）、F5（坏窗口丢弃）。
- `tests/test_state.py`：F3（buffer 历史已含 $o_t$，循环首步不得重复 append）。

### 10.2 保留 / 仅文档说明

| 编号 | 状态 | 原因 |
|---|---|---|
| F7 🟢 | 未改 | 真实/想象 action_rate 的噪声方差差 $2\sigma^2$ vs $\sigma^2$，量级 $2\times10^{-5}$/步，可忽略 |
| F9 🟢 | 未改 | `nan_to_num` 仍会把 NaN 变 0；建议后续加计数（当前只靠 `isfinite(loss)` 跳过更新） |
| F13 🟢 | 未改 | env 自带 observation_manager 与 trainer 特征两套定义仍在；删除会影响 mjlab 观测接口，留作后续清理 |
| F17/F18/F19 🟢 | 保留 | goal 多 6 维参考速度、reset pose/velocity/joint 扰动、running normalization/term-bias 初始化，都是论文未给细节的工程扩展；已在报告 §9 说明 |
| P2-1 🟢 | 未改 | `action_norm` 仍用带噪动作统计；实测归一化后 clean 动作 std 0.963，影响小 |
| P2-8 🟡 | 已加断言 | `MotionLibraryCommand.__init__` 现在要求 `abs(library.dt - env.step_dt) < 1e-6`，混帧率会直接报错；默认 50 Hz 不受影响 |

### 10.3 修复后的验证

```bash
cd /home/zy0028/mjlab_project/supermimic-mjlab
export MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src
PY=/home/zy0028/miniconda3/envs/tfbot_rl/bin/python

$PY -m pytest tests -q                 # 37 passed（含 GPU smoke）
$PY scratch/audit_p1/probe_runtime_bins.py    # F1: visits 按 episode 计，不再按帧
$PY scratch/audit_p1/probe_terminal_training.py  # F14: forced falls, terminal_count>0
$PY -m supermimic.scripts.evaluate --checkpoint logs/2026-10-29_20-19-56/checkpoint_0034000.pt \
    --motion-dir /tmp/eval2 --num-envs 2 --device cuda:0   # F20/F21: 2 clips -> 2 episodes
```
