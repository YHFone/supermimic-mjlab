# SuperMimic-mjlab 远程代码审计（P1 + P3 + P4 + P5）

> 范围：`/data02/zy0028/supermimic-mjlab`，代码版本 `933bd72`（审计修复后的版本）。
> 远程工作区只有 `README.md` 未提交；源码与本地 `933bd72` 一致。
> 本地 GPU 用于数值复现（RTX 4090），远程 H20 上的训练未受干扰。
> 结论：**没有新的 🔴**；上一版的 F1（bin 单位）已修复。本轮发现 5 条 🟡（终止率标定、
> warmup 守卫、空窗口静默、评估 max_steps、10 s 回合截断）和若干 🟢。

## 1. 数据流（形状与单位）

| 阶段 | 张量 | 形状 | 单位 / 参考系 |
|---|---|---|---|
| env | action | $(N,29)$ | 原始策略输出，无量纲；env 乘 `G1_ACTION_SCALE` 加默认位姿 |
| step | `step_dt` | — | 0.02 s（`sim.timestep=0.005` × `decimation=4`） |
| state $s_t$ | `[p,q,v,ω,q_j,\dot q_j]` | $(N,71)$ | m、wxyz 单位四元数、m/s、rad/s、rad、rad/s；$p$ 为 env-local |
| context $\xi$ | `[μ,b,Δc]` | $(N,33)$ | 无量纲、29 个 rad、3 个 m |
| obs $o_t$ | `[q_j-q_0+b,\dot q_j,R(q)^Tω,R(q)^Tg,a_{t-1}]` | $(N,93)$ | rad、rad/s、rad/s、无量纲、原始 action |
| history | $o_{t-9..t}$ | $(N,10,93)$，展平 930 | 同一 episode；reset 时整窗填第一个 obs |
| goal $g_t$ | 10 帧，每帧 73 | $(N,730)$ | rad、rad/s、m、旋转矩阵前两列、m/s、rad/s |
| buffer | `states/actions/contexts` | $(N,1000,71/29/33)$ | 原始量；reward 已乘 dt |
| terminal ring | `terminal_states` | $(65536,71)$ | 硬终止状态 $s_{t+1}$；用于终止头正样本 |
| WM 窗口 | $s_{t-8..t}$，$a_{t-8..t-1}$ | $(2048,9,71)$、$(2048,8,29)$ | 8 步自回归；loss 除以 $k$ |
| imagination | $\hat s_t,a_t,g_t,r_t,d_t$ | $B=4096$，$H=32$ | 同一 dt/γ/λ/尺度；chunk 默认 4096 |
| critic | $[s,g,\xi]$ → $V$ | $(B,834)$ → $(B,)$ | 归一化后输入；目标 no_grad |
| eval | 40 clips，episode 60 s | 40 环境 | 无 DR、摩擦 1.0、每条 clip 一次 |
| ONNX | $[o\vert g]$ → $a$ | $(1660)$ → $(29)$ | obs 930 + goal 730；归一化与 tanh 烘入图 |

## 2. 不变量

**S 状态/观测**
$$
\|q_t\|=1,\quad \dim s=13+2J,\quad \dim o=3J+6
$$
$$
o_t=[q^j_t-q_0+b,\ \dot q^j_t,\ R(q_t)^T\omega_t,\ R(q_t)^Tg_w,\ a_{t-1}]
$$

**G 目标**（$k=1..K$，$K=10$）
$$
g^{(k)}_t=[q^{ref}_{t+k}-q_0,\ \dot q^{ref}_{t+k},\ R(q^{ref}_t)^T(p^{ref}_{t+k}-p^{ref}_t),\ [R(q_t)^TR(q^{ref}_{t+k})]_{:,1:2},\ R(q^{ref}_{t+k})^Tv^{ref}_{t+k},\ R(q^{ref}_{t+k})^T\omega^{ref}_{t+k}]
$$

**B 窗口/分箱**
$$
\forall i\in[0,k-1]:\ \mathrm{done}(p-k+i)=0,\qquad
G_b=\sum_{t\ge bB}\gamma^{t-bB}r_t,\ B=50
$$
$$
P(b)=0.5U+0.4\,\mathrm{softmax}(-25G_b/10)+0.1P_{past}
$$

**W 世界模型**
$$
\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),\quad \hat q_{t+1}=\mathrm{normalize}(\exp(r_\omega)\otimes\hat q_t)
$$
$$
\mathcal L=\frac1k\sum_{j=1}^k\mathrm{MSE}_{norm}(\hat s_j,s_j)+\frac1k\mathrm{BCE}(\hat d_j,d_j)
$$

**I 想象/评论家**（$c_t=\prod_{k<t}(1-d_k)$，$d$ 在 $s_{t+1}$ 上评估）
$$
\hat J=\sum_{t=0}^{H-1}\gamma^t c_t r_t+\gamma^H c_H V_\psi(\hat s_H,g_H,\xi)
$$
$$
\hat R_t=r_t+\gamma(1-d_t)[(1-\lambda)V_{t+1}+\lambda\hat R_{t+1}],\quad \hat R_H=V_H
$$

**N 归一化**：$z=(x-\mu)/\sqrt{\sigma^2+\varepsilon}$；统计量是 buffer（无梯度）；`update` 只接原始张量。
**E 导出**：$a=\mathrm{dec}(\tanh(\mathrm{enc}((g-\mu_g)/\sigma_g)),(o-\mu_o)/\sigma_o)$，误差目标 $<10^{-4}$。

## 3. 核对表

| 级别 | 不变量 | 违反点 | 数值反例/证据 | 影响面 | 最小修法 | 回归断言 |
|---|---|---|---|---|---|---|
| 🟡 | 终止头正样本应反映当前终止率 | `trainer.py:389-400`；`buffer.py:18,42` | terminal ring 上限 65536；4096×1000 buffer 下 `rate` 上限 1.6%。若早期摔倒率高，ring 被旧正样本占满，当前低摔倒率时 rate 仍为 1.6%，正样本被高估约 5× | 终止头标定 → 生存权重 → 有效 horizon | 用 `terminated.float().mean()` 的 EMA 作为正样本比例，或扩大/重采样 ring | 构造“旧正样本多、当前终止率低”的 buffer，断言 n_pos 跟随当前 EMA |
| 🟡 | warmup 前不得采样 | `trainer.py:785`；`buffer.py:103,148` | `warmup_steps=0`：`sample_windows` 返回 0 行；`sample_starts` 返回 `positions=[15,15]`、state 全 0，`_update_policy` 在零状态上更新 | 冷启动静默训练垃圾 | `assert warmup_steps >= wm_window + 1` 或自动抬到该值 | 空 buffer 调 `sample_windows` 必须抛错/返回明确标志 |
| 🟡 | 无干净窗口时不能静默停训 | `trainer.py:373,438`；`buffer.py:103-146` | 每 2 步 done 的对抗输入：`sample_windows` 返回 0/0，WM loss 分支直接 `continue`，无日志 | 高摔倒率阶段 WM 静默停止学习 | 计数 `dropped_windows` 并定期打印/断言 | 对抗 done 模式下断言 dropped>0 且日志可见 |
| 🟡 | 评估能覆盖全部 motion | `evaluate.py:39,105,165` | `max_steps=5000` 是外层步数上限；64 env、每条 ~3000 步时只够 ~1.6 episode/env≈100 条 rollout，129,514 条 benchmark 需要 ~6M 步 | 无法复现论文 Table A.8 全量评估 | `max_steps = ceil(num_clips/num_envs)*ceil(episode_length_s/step_dt)+margin`，或循环到 `seen_clips` 满 | 40/129k 两种规模断言 `len(seen_clips)==num_clips` |
| 🟡 | 训练回合应覆盖 clip | `env_cfgs.py:69`；`train.py:28` | 日志：`episode_length_s=10.00 < longest clip 273.10s`；>10 s 的尾段只能靠 reset 覆盖，`motion_end` 不触发 | 长 clip 尾段训练不足 | 提高训练回合上限或按 clip 长度配置；至少保留警告 | 断言 `episode_length_s >= min(max_clip_s, 上限)` |
| 🟢 | ONNX 与 PyTorch 逐位一致 | `export.py:262` | 36k checkpoint：max abs diff **1.18e-5**；同一输入 max\|a\|=15.87 时相对 4.8e-7；测试阈值 1e-4 | 部署动作有小数值误差，无语义错位 | 保持 float32；若要求 1e-6，需 float64/固定 matmul 顺序 | 记录 worst error；goal/obs 契约单独断言 |
| 🟢 | 策略 bootstrap 的输入应 detach？ | `trainer.py:625`、`706` | H=1 手算：$r=0.0781128,d=0.0499873,V=-0.0531811$，$\hat J=r+\gamma(1-d)V=0.02808443$，与 `policy/objective` **完全相等** | 若按字面 detach，会切断 actor 经 $s_H$ 的梯度；当前实现符合 Eq. 2 | 修正不变量表述：critic target 的 values 才 detach；policy bootstrap 输入保留图 | 断言 bootstrap 对 actor 输入有梯度、critic target 无梯度 |
| 🟢 | 终止缓冲应随 checkpoint 恢复 | `buffer.py:39-53`；`trainer.py:956-999` | resume 后 `terminal_count=0`，终止头要等新摔倒才恢复正样本 | 续训前若干步终止头偏置；不影响从头训练 | 把 terminal ring 一并保存/恢复，或 resume 时重采样 | 续训前后 `terminal_count` 一致 |
| 🟢 | $G_b$ 应从 bin 起点算 | `trainer.py:331-359`；`sampler.py:131` | episode 从 frame 123 开始时，记录到 bin 2 的值是 123 的回报，不是 100 的回报；首/末 bin 均为实际 reset 点 | 分箱回报是“从 bin 内某个实际起点”的估计，不是严格 bin 起点 | 文档说明；若要严格，按 bin 边界重采样起点 | 构造 start=123 的 episode，断言记录值来自 123 |
| 🟢 | critic 归一化统计应覆盖输入 | `trainer.py:475-497`、`706-712` | 真实 vs 想象 H=32：state std 0.960→0.897（−6.6%），goal 0.870→0.854（−1.8%），max\|z\|≈4.9 两边相同 | 影响小；注释已改为“稳定参考” | 需要时用想象 batch 增量更新 | 统计来源单测 + 漂移监控 |
| 🟢 | `action_norm` 应用干净动作 | `networks.py:163`；`trainer.py:460-473` | 实测 clean action std 0.3579；带 σ=0.1 统计的 noisy std 0.3716；归一化 clean std 0.963（−3.7%） | WM 想象输入被压缩 3.7% | 用 clean action 拟合，或想象加同噪声 | 断言 clean/noisy 归一化 std 一致 |
| 🟡 | RunningNorm 不应无限记忆 | `networks.py:37-88`；`trainer.py:460-497` | 单次权重：obs 1/(2.5i)，state/critic 10/i；320k iter 时 obs 1.25e−6、state 3.13e−5；最后 3% 迭代只占 3.1% | 晚期分布漂移吸收慢 | EMA/固定窗口或 cap count | 阶跃分布测试：新均值在 1 个窗口内跟上 |
| 🟢 | 采集/想象 action_rate 同分布 | `trainer.py:247-255` vs `541-616` | 采集 $E[\Delta^2]+2\sigma^2$，想象 $+\sigma^2$；$2\sigma^2\vert w\vert dt\approx2\times10^{-5}$/步 | 可忽略 | 想象侧加同分布噪声 | 两侧 action_rate 期望差 <1e-3 |
| 🟢 | eval CLI 语义 | `evaluate.py:33` | `episodes_per_env` 参数已不参与 `seen_clips` 逻辑 | 用户以为 4 episodes/env，实际每条 clip 一次 | 删除参数或改 help | CLI help 断言 |
| 🟢 | 状态/目标归一化覆盖想象分布 | `networks.py:157-224` | 真实→想象 state std 漂移 ~6.6%；`goal_norm` 同理 | 与 critic 同类，量级小 | 监控/EMA | 同 P2 |

## 4. P3 想象循环与 BPTT

### 4.1 梯度路径

```
buffer: s_t, a_{t-1}, ξ, o_{t-9..t}        (detached)
  │
  ├─ actor π_θ(o_{t-9..t}, g_t) ──► a_t     (θ 可导)
  │
  ├─ f_ω(s_t, a_t, ξ) ──► s_{t+1}           (ω frozen, 输入可导)
  │     ├─ FK ──► body poses/vel           (无参数)
  │     ├─ reward ──► r_t                  (g_t=ref_t)
  │     └─ d_ω(s_{t+1}, g_{t+1}) ──► d_t   (ω frozen, 输入可导)
  │
  ├─ c_t = Π_{k<t}(1-d_k);  J = Σ γ^t c_t r_t + γ^H c_H V(s_H)
  │                                      └─ critic frozen, 输入可导
  └─ ∇_θ J
critic target: V(s_t) under no_grad; R_t = r_t + γ(1-d_t)[(1-λ)V_{t+1}+λR_{t+1}]
```

### 4.2 数值

- `probe_p3_chunk.py`（2 env，H=4，batch=4）：
  - chunk=2：objective `0.2750189`，actor grad norm `0.006108`，WM grad tensors `0`，critic grad tensors `4`（来自 critic 拟合）；
  - chunk=4：objective `0.27501854`，grad norm `0.006111`；
  - 差值：objective `3.58e-7`，grad norm `2.29e-6`（相对 0.04%）。
- `probe_p3_h1.py`（H=1，单样本手算）：`0.02808443084359169` vs objective 相同，diff `0.0`。
- WM 参数在 policy backward 时 `requires_grad=False`，`.grad` 张量数 0；critic 只在 `_update_critic` 阶段被写梯度；`all_states/goals/rewards/terms/context` 都 `.detach()` 后进 critic；`values/targets` 在 `torch.no_grad()` 下算。
- **detach 结论**：没有多余 detach；policy bootstrap 的输入**故意不 detach**（否则 actor 失去经 $s_H$ 的梯度）。不变量应写成“critic target 的 values 无梯度；policy bootstrap 的 critic 参数冻结但输入保留图”。

## 5. P4 Buffer / 采样 / episode 边界

三 episode，长度 `[2,5,3]`：

| 窗口 | episodes | 跨 reset? |
|---|---|---|
| t0=0,t1=2 | [0,0,1] | 是 |
| t0=1,t1=3 | [0,1,1] | 是 |
| t0=2,t1=4 | [1,1,1] | 否 |
| t0=3,t1=5 | [1,1,1] | 否 |
| t0=4,t1=6 | [1,1,1] | 否 |
| t0=5,t1=7 | [1,1,2] | 是 |
| t0=6,t1=8 | [1,2,2] | 是 |
| t0=7,t1=9 | [2,2,2] | 否 |

`sample_windows(200,length=3)`：返回 200 行，跨 reset **0**；有效 end 位置 `4,5,6,9`（ep1 起点 2,3,4；ep2 起点 7）。ring capacity=8 回绕后仍 **0** 跨 reset。
`sample_windows` 在“每 2 步 done”的对抗输入下返回 **0 行**（不是坏窗口）；`_update_world_model` 会 `continue`，见上表 A3。
`_finish_episodes`：`bin_frames = sampler.bin_frames = 50`；120 帧 episode → visits `[1,1,1]`，bin_returns = `0.1*G(0/50/100)`；真实 `_collect` 10 episodes → visits `[3,8]`（sum 11），不再按帧写。
`sample_windows` 的 `length=k+1` 返回 $s_{t-k..t}$，训练只用 $a_{t-k..t-1}$ 的 k 个动作；索引正确。
`warmup=0`：`sample_windows` 0 行、`sample_starts` 返回零状态（positions=[15,15]），见 A2。

## 6. P5 参考系 / 四元数 / 部署契约

- 同一 clip，frame 100，训练 `GoalFeatures` vs `deploy_features.build_goal`：
  - 总 max abs diff **2.38e-7**；joint_pos/joint_vel 0；anchor_pos 2.98e-8；anchor_ori 3.21e-8；lin_vel 2.38e-7；ang_vel 5.96e-8。
  - 观察 `ProprioFeatures` vs `build_observation`：**0.0**。
- 约定：四元数 wxyz；`quat_mul` Hamilton、左乘 world rotation；`quat_inv(root)*ref`；`matrix_from_quat(...)[..., :2]` 与 `quat_to_matrix(rel)[:, :, :2]` 一致；goal 帧 $t+1+k$；位置 $R(q^{ref}_t)^T(p_k-p_t)$；速度在 ref 帧；BONES-SEED CSV 列序与 entity 关节序逐项一致（已验证）。
- 动作：`q^{sp}=q_0+\alpha a$；`export_onnx` 用 `0.25*effort_limit/stiffness` 生成 `action_scale`，与 `G1_ACTION_SCALE` 同式；env 的 `JointPositionAction` 再减 encoder bias，部署 C++ 按 sidecar 的 scale/default 组装，契约一致。
- ONNX：`load_actor` + `export_actor_onnx` + `verify_onnx`：输入 `[batch,1660]`，输出 `[batch,29]`；max abs diff **1.18e-5**；同一输入 max\|a\|=15.87 时相对 **4.8e-7**；来源是 float32 矩阵乘/ORT 累加顺序，不是布局或归一化错误（goal/obs 契约已到 1e-7）。测试阈值 1e-4。

## 7. 已验证为正确（本轮）

- P3：frozen 参数、actor 单一路径、critic no_grad、TD(λ) 递推、survival 累计、H=1 手算、chunk 不变性。
- P4：3 episode 窗口、ring wrap、`sample_windows` 拒绝跨 reset、`_finish_episodes` bin 帧数、`update_returns` 重复 bin 聚合、done/reset/写入顺序、最后 reward 计入。
- P5：goal/obs 契约、四元数顺序与乘法、anchor-relative 公式、帧索引、速度参考系、关节序、action scale、ONNX 归一化/tanh。
- 上一版 F1（bin 单位）、F2（重复 EMA）、F3（历史重复）、F4（bias 切片）、F14（终止头 $s_t$）、F15（BCE 权重）已修复；本轮未复现。

## 8. 复现命令

```bash
cd /home/zy0028/mjlab_project/supermimic-mjlab
export MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src
PY=/home/zy0028/miniconda3/envs/tfbot_rl/bin/python

$PY scratch/audit_p1/probe_p3_chunk.py       # P3 chunk/frozen
$PY scratch/audit_p1/probe_p3_h1.py          # P3 H=1 手算
$PY scratch/audit_p1/probe_episode_windows.py # P4 三 episode
$PY scratch/audit_p1/probe_p5_contract.py    # P5 goal/obs
$PY scratch/audit_p1/probe_p5_onnx.py        # P5 ONNX
$PY -m pytest tests -q                       # 37 passed
```
