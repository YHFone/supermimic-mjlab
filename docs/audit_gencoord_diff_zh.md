# 审计：本地实现 vs 官方 `SuperMimic-GenCoord-G1`

> **实施状态（2026-11-11，本次提交）**
>
> | 项 | 状态 |
> |---|---|
> | D1 DR 采样模式 `reset→startup` | ✅ 已改 |
> | D2 TF32 | ❌ **误报，已澄清**：官方 `setup_device()` 内部调用 `configure_torch_backends()`（默认 `allow_tf32=True`），双方都开 TF32 |
> | D3 RSI 失败/超时优先级 | ✅ 已改（`terminated & ~truncated`、`truncated & ~clip_end`） |
> | D4 UTD 相位 | ✅ 已改为官方循环（每迭代 1 步 + `Ratio(0.4)` 触发） |
> | D5 RSI 分箱 | ✅ `bin_count = T//50 + 1`，连续映射 |
> | D7 种子 | ✅ 配置默认 `None`；**本轮补**：`TrainArgs.seed` 也从 0 改为 None，否则入口仍会传 0 |
> | D8 critic perm | ✅ 每次更新只采样一次排列 |
> | D10 episode length 口径 | ✅ 主指标改为官方口径（失败/500 步 flush，deque-100），另加 `Mean true episode length` |
> | D6 auto_reset / 奖励来源 | ✅ 已改：`auto_reset=True`，buffer/RSI 用 env 侧奖励，解析奖励只用于想象；尾部奖励日志改读 `extras["Episode_Reward/*"]` |
> | D9 compile / CUDA graph | ✅ 已改：想象步 `torch.compile`；策略更新用 `CapturedUpdate`（`algo/capture.py`，warmup 3 次后捕获、之后 replay），在本机 4090 上验证 `captured=True`，eager/捕获两条路径指标一致；优化器改为 capturable Adam + device lr |
> | D11 观测组 / state 列序 | ⏸ 保留（内部布局，rep 接口相同） |
> | D12 checkpoint 格式 | ⏸ 保留（三文件 vs 单文件，纯格式） |
> | **A1** `warmup_steps=64` | ✅ 已改为 0（官方首个训练轮在 buffer 满 8 帧时） |
> | **A2** `TrainArgs.seed=0` | ✅ 已改为 None |
> | **A4** Adam fused | ✅ 参数在 CUDA 上时启用 `fused=True`（官方 fused+capturable） |
> | **A5** 复位时关节限位裁剪 | ✅ 已删除（mjlab 不裁剪） |

> 范围：官方 `supermimic_ws/supermimic` + `supermimic_ws/mjlab_research`（fork）的 GenCoord 训练线。
> 排除：velocity 任务槽、MaxCoord 表示、PPO 基线（按要求不在范围内）。
> 日期：2026-11-11。本地对应 commit `343062a` 之后的工作树。

---

## 0. 结论

**方法、网络、损失、超参、数据：一致。**
**仍有 2 处会改变训练语义（🔴）、5 处会带来数值差异（🟡）、4 处属工程/日志（🟢）。**

其中 **🔴-1（域随机化的采样模式）** 是这份审计里唯一可能解释"对齐后训练平台化"的实质差异，之前几轮都没有发现。

---

## 1. 已逐项核实一致（不作为差异）

| 组件 | 核对内容 | 证据 |
|---|---|---|
| 仿真参数 | `nconmax=35, njmax=250, timestep=0.005, iterations=10, ls_iterations=20, decimation=4` | `mjlab_research/.../tracking_env_cfg.py:308-317` vs `config/g1/env_cfgs.py:49-58` |
| episode 长度 | 训练 `10.0`；play `1e9` | `tracking_env_cfg.py:316`；`.../g1/env_cfgs.py:84-89` |
| 奖励项 | 8 项同名同权重：`0.5/0.5/1/1/1/1/-0.1/-2` | `tracking_env_cfg.py:209-246` + supermimic 覆盖（pop self_collisions、joint_limit→-2） |
| 奖励 σ | `0.3/0.4/0.3/0.4/1.0/3.14` | 同上 `params={"std": ...}` |
| 奖励公式 | re-anchor 到"预测 root xy + 参考 z / 侧倾俯仰" | `tracking/mdp/rewards.py:44-60` 与 `MotionCommand.update_relative_body_poses` |
| 终止 | `time_out / motion_end(time_out) / anchor_pos 0.25 / anchor_ori 0.8`，`ee_body_pos` 被移除 | `tracking_env_cfg.py:257-281` |
| DR 范围 | 摩擦 `(0.3,1.2)`、CoM `(±0.025, ±0.05, ±0.05)`、编码器偏置 `(±0.01)` | `tracking_env_cfg.py:165-200` |
| 复位扰动 | `pose_range` 6 项、`velocity_range=VELOCITY_RANGE`、`joint_position_range=(-0.1,0.1)` | `tracking_env_cfg.py:138-160` |
| 参考数据 | 同 40 条 LAFAN clip，`joint_pos` 逐值差 mean 3.8e-6 | 实测 |
| 表示/积分 | 68/7/35、`c=0.625`、半隐式 | `representations.py:505-554` vs `algo/representation.py` |
| goal | 67/帧 × 10、stride 5、root 相对、actor 640 / critic 670 | `representations.py:96-116,213-234,401-417` |
| ξ | 列序 `[friction, com, bias]`、`(raw-mean)/scale`、退化半宽→1 | `sim_context.py:26-31,171-192` vs `algo/state.py:129-` |
| 动作变换 | `a·scale+offset`（与官方 checkpoint 逐元素相同） | `models.py:82-96` |
| WM 损失 | Bryson `1/nominal²`，评 7 步 | `learners/wm_learner.py:64-68,145-167` |
| 归一化器 | `x/(std+1e-2)`、eval no-op、每收集步更新 | `models.py:15-28,73-74`；`wm_learner.py:78-101` |
| 优化器 | Muon（Sequential 中间 Linear）+ Adam(fused)，WM/actor/critic 均如此 | `learners/learner_utils.py:13-36` |
| learner | actor-critic：γ=0.99、`γ^H·V·c_H`、GAE clamp、5×4 | `learners/policy_learner.py:139-202` |
| RSI 参数 | explore .5 / steepness 25 / keep .1 / memory 256 / horizon 20 / v_max 10 | `runners/reset_sampling.py:18-40` |
| buffer | 1000/env，窗口不跨 episode | `replay_buffer.py:88-121` |

---

## 2. 仍存在的差异

### 🔴 会改变训练语义

#### D1. 域随机化的采样模式：`startup` vs `reset`

| | 官方 | 本地 |
|---|---|---|
| base_com / encoder_bias / foot_friction | `mode="startup"`：**每个环境在仿真启动时采样一次，整个训练不变** | `mode="reset"`：**每个回合重置时都重新采样** |
| 证据 | `mjlab_research/.../tracking_env_cfg.py:164,172,181`（`mode="startup"`）；supermimic 只做名字过滤，不改 mode（`supermimic/tasks/env_cfgs.py:187-190`） | `src/supermimic/config/g1/env_cfgs.py:139,152,160`（`mode="reset"`） |

**影响**（这是本次审计最重要的一条）：
- 官方：每个 env 的动力学在整个 run 中固定，`ξ` 是"这个 env 的编号"，世界模型只需学 4096 种条件；同一个 env 的历史数据永远来自同一动力学。
- 本地：每个回合换一次动力学，缓冲区内同一个 env 的 1000 步会混入约 2 个不同的 ξ；世界模型要拟合的"状态→下一状态"映射在时间上非平稳，BPTT 的梯度目标随之漂移。
- 这正好对得上之前的现象：**高 UTD（1024 envs）时新代码更好，低 UTD（4096 envs）时平台化** —— UTD 低时模型更新次数少，更受这种非平稳性拖累。

**修法**：三处 `mode="reset"` → `mode="startup"`。一行改动，但会显著改变训练分布，建议单独一次对照实验验证。

#### D2. TF32

| | 官方 | 本地 |
|---|---|---|
| matmul 精度 | 未设置 → PyTorch 默认 `allow_tf32=False`（fp32） | `allow_tf32=True` + `set_float32_matmul_precision("high")` |
| 证据 | `grep tf32 supermimic/` 无命中 | `algo/trainer.py:70-73` |

**影响**：所有 matmul（动力学、actor、critic）本地走 TF32，官方走 fp32。数值差 ~1e-3 量级，梯度也略有不同；对比曲线时可能贡献小偏差。**修法**：默认 `use_tf32=False`，或在文档里标明这是本地加速选项。

### 🟡 数值/调度差异

#### D3. RSI 的"失败/超时"优先级
官方：`sampler.record(..., terminated & ~truncated, clip_end, truncated & ~clip_end)`（`supermimic_runner.py:645-651`）。
本地：`sampler.record(pre_frames, reward, terminated, self.is_motion_end(), timed_out)`（`mdp/commands.py::record_step`）。
同一步既失败又超时时，官方归为超时，本地归为失败。**修法**：传 `terminated & ~truncated`。

#### D4. UTD 相位
官方：每仿真步采 1 步，按 `Ratio(0.4)` 以 0.4 概率触发一次训练（`supermimic_runner.py:672-692`）。
本地：每轮采 2.5 步（2/3 交替）后固定触发一次（`algo/trainer.py` 的 `_step_accumulator`）。
平均比值相同；差异在"更新时刻的数据新鲜度"。**修法**（可选）：改成逐触发、按概率。

#### D5. RSI 分箱数差 1，bin→frame 映射略不同
官方：`bin_count = int(T // (1/step_dt)) + 1`（`mjlab_research/.../tracking/mdp/commands.py:166`），采样时 `((bin + U(0,1)) / B * (T-1)).long()`（`:231-240`）。
本地：`bin_count = round(T * dt / 1.0)`（`algo/sampler.py:59-62`），整数 bin 边界 `[b*T//B, (b+1)*T//B)`。
T=129k 时 2581 vs 2580，差 1 个 bin；映射方式一个是"连续比例"、一个是"整数区间"。**影响很小**。

#### D6. `auto_reset` 的归属
官方：mjlab 默认 `auto_reset=True`（`manager_based_rl_env.py:144`），env 内部重置。
本地：`env_cfgs.py` 显式 `auto_reset=False`，trainer 手动 `env.reset(env_ids)`。
缓冲数据等价（存的是 pre-step state），差异在观测返回时机与 RSI 记录的时序。**影响小**。

#### D7. 随机种子
官方：`scripts/train.py` 不设种子（无 `torch.manual_seed`/`env.seed`）。
本地：`train.py` 设 `env_cfg.seed`、`torch.manual_seed`、独立 `Generator`。
本地可复现，官方不可复现；对比时无法"同种子"。

### 🟢 工程/日志

| # | 项 | 官方 | 本地 |
|---|---|---|---|
| D8 | critic minibatch 排列 | 每轮固定一个 `perm`，5 个 epoch 复用（`policy_learner.py:190-193`） | 每 epoch 重新 `randperm` |
| D9 | 编译/图 | `torch.compile(wm_env.step/reset)` + CUDA graph 捕获更新（`learner_utils.py:87-136`） | 只 `torch.compile` FK |
| D10 | 日志 | W&B + 控制台；`mean_episode_length` 只在硬失败或 500 步 flush，deque(100) 平滑 | TensorBoard；`mean_episode_length` 按真实回合（含截断），按 log 窗口平均 |
| D11 | env 观测组 | `actor(93) / full_state(71) / goal(670)`，`full_state` 列序 `[p,q,qj,v,w,vj]` | 只定义 proprio(93) 组（trainer 不用）；内部 state 列序 `[p,q,v,w,qj,vj]` |
| D12 | checkpoint | 三份 `.pt`（wm bundle / actor / critic） | 单文件 + schema + cfg |

---

## 3. 建议的处理顺序

1. **D1（DR 模式）**：`mode="reset"` → `"startup"`，跑 4096-env × 2k 迭代对照；这是唯一可能解释平台化的差异。
2. **D3（RSI 优先级）**：一行，顺手改。
3. **D2（TF32）**：默认关掉，或保留但记录在案（它让本地比官方"快但略不准"）。
4. D4/D5/D6/D7：影响小，若要严格对齐再改；D7 建议保留本地种子（可复现是优点）。
5. D8–D12：不影响算法，按需。

---

## 4. 回归断言建议

| 差异 | 断言 |
|---|---|
| D1 | `env.cfg.events["foot_friction"].mode == "startup"`（三处） |
| D3 | 构造 `terminated=truncated=True` 的一步，断言 sampler 记为 truncated |
| D5 | `bin_count == int(T // 50) + 1` |
| D2 | `cfg.use_tf32 is False`（若采纳） |
