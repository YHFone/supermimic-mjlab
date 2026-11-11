# 官方仓库 vs 本仓库对比（SuperMimic）

> **另见 `docs/official_vs_local_zh.md`**：按模块/分支重新整理的官方实现解剖与逐项对照，
> 含"两种 learner（bptt vs actor-critic）"的精确差异和数据对照。本文保留为最早一轮的
> 静态对比记录。

> **状态（2026-11-11）：已按官方 GenCoord 实现对齐（schema v2）。**
> 下表记录的是对齐**之前**的差异，保留作为证据链。对齐后的契约：
> rep/aux/pred = 68/7/35；goal = 67×10（stride 5），actor 640 / critic 670；
> 终止为解析软阈值；WM 损失为 Bryson 分组加权、评 7 步；`EmpiricalNorm`
> （`x/(std+1e-2)`、eval no-op）；Muon + Adam；恒定 lr、1e6 迭代；
> `actor_obs` 默认 `rep_state`（H=1），`proprio`（93×H）为可部署模式。
> 详见 `docs/align_gencoord_plan_zh.md` 与 `tests/test_official_contract.py`。

> 对象：`/home/zy0028/mjlab_project/supermimic_ws/supermimic`（官方）与 `/home/zy0028/mjlab_project/supermimic-mjlab`（本仓库）。
> 方法：只读对比。git 历史 / 逐文件阅读 / 去重代码行重合率。未执行任何写操作，未改动源码。
> 约定：$J=29$（G1 关节）、$B=14$（跟踪体）、$dt=0.02$ s。标注【推断】的结论仅来自静态阅读，未经运行时验证。

---

## 0. 结论

**两个仓库不是同一份代码的两个版本，而是同一篇论文的两个独立实现。**

| 判据 | 官方 | 本仓库 |
|---|---|---|
| remote | `HybridRobotics/supermimic` | `YHFone/supermimic-mjlab` |
| commits | 285（2026-05 起） | 9（2026-10-27 起） |
| 作者 | qiayuanl、Kevin Zakka、Yiyang Shao、Junfeng Long（与论文作者表一致） | yinhoufang / YHFone |
| 去重代码行重合 | 5661 行 | 125 行与官方重合（2.2%，均为 import/样板） |

因此"对齐官方"不存在逐行 diff 的路径，只能按语义逐项对账。以下所有差异都带 `文件:行` 证据。

---

## 1. 工程身份

| 维度 | 官方 | 本仓库 |
|---|---|---|
| Python / 构建 | ≥3.13，`uv_build` | ≥3.10，`hatchling` |
| mjlab | **路径依赖** `../mjlab_research`（`pyproject.toml` `[tool.uv.sources]`） | PyPI `mjlab>=1.6.0` |
| 其它依赖 | `tensordict`、`rsl_rl`、`viser`、`wandb` | `torch`、`mujoco`、`tyro`、`tqdm`、`pillow` |
| 任务 | `SuperMimic-GenCoord-G1`、`SuperMimic-MaxCoord-G1` | G1 单任务 + 多 clip 运动库 |
| 入口 | tyro + mjlab task registry | 7 个 `supermimic-*` console script |
| 代码量 | 10,263 行 | 6,908 行 |
| 测试 | 20 个测试文件 | 13 个测试文件 |

本机不存在 `../mjlab_research` 目录，官方仓库无法开箱运行。本机的 `/home/zy0028/mjlab_project/mjlab`（含 `src/mjlab/tasks/tracking`、`velocity`，依赖 `rsl_rl`）是可能替代品，但 `MotionCommand` 的 `external` 采样模式、`mjlab.rl.exporter_utils`、`ForwardKinematics` 的 API 是否与作者 fork 一致需先验证【推断】。

---

## 2. 结构映射

| 官方 | 本仓库 |
|---|---|
| `representations.py` (800) | `algo/state.py` + `algo/reward.py` + `algo/kinematics.py` |
| `models.py` (248) | `algo/networks.py` (294) |
| `learners/{wm,policy}_learner.py` + `learner_utils.py` | `algo/trainer.py` (1051) |
| `runners/{supermimic,ppp,exporter,run_config,reset_sampling}.py` | `algo/trainer.py` 的循环部分 + `algo/sampler.py` + `algo/export.py` |
| `replay_buffer.py` (121) | `algo/buffer.py` (216) |
| `sim_context.py` (204) | `algo/state.py` 的 context 部分 |
| `envs/world_model_env.py` (290) | `algo/trainer.py` 的想象 rollout |
| `tasks/{env_cfgs,task_spec}.py` | `config/g1/env_cfgs.py` + `mdp/*` |

---

## 3. 数值契约一致的部分

| 量 | 官方 | 本仓库 |
|---|---|---|
| 动作噪声 | 0.1 | 0.1 |
| proprio 噪声半宽 | 0.05 / 0.2 / 0.01 / 0.5 | 同 |
| WM 窗长 / batch / 每轮步数 / lr | 8 / 2048 / 10 / 2e-4 | 同 |
| WM 隐层 | (2048,2048,2048) | 同 |
| BPTT H / batch / lr | 32 / 4096 / 2e-4 | 同 |
| γ / λ / 逐元素裁剪 | 0.99 / 0.95 / 0.1 | 同（纯 BPTT 分支官方为 γ=1.0） |
| actor / critic 隐层、latent | (2048,1024,512,256) / (256,128) / 32 | 同 |
| critic 拟合 | 5 epoch × 4 minibatch | 同 |
| 奖励 8 项系数与 std | 0.5/0.3、0.5/0.4、1/0.3、1/0.4、1/1.0、1/3.14、−0.1、−2.0 | 完全相同 |
| 单步奖励上界 | 0.02×5 = 0.1 | 同 |
| RSI | explore 0.5 / keep 0.1 / steepness 25 / $v_\max=10$ | uniform 0.5 / softmax 0.4 / past 0.1 / T=25 / max_return=10 |
| UTD | 4096/10/2048/8×16 ≈ 0.4 | `num_steps_per_env=2.5` ⇒ 0.4 |
| 环境数 / buffer | 4096 / 1000 | 同 |
| 动作维度 / 缩放 | 29，`G1_ACTION_SCALE`（$0.25\cdot$effort/stiffness） | 同 |

---

## 4. 环境侧差异

### 4.1 状态表示

| | 官方 GenCoord | 本仓库 |
|---|---|---|
| full state | 71 = `[p3,q4,qj29,v3,w3,vj29]` | 71 = 同布局（`state.py:156-168`） |
| 网络状态 | **68** = `[z,g_b3,v_b3,w_b3,qj29,vj29]`，**不含四元数、不含 $x,y$** | **71**，含四元数与世界系速度 |
| 预测目标 | **35** = $[\Delta v_b,\Delta\omega_b,\Delta\dot q_j]$（加速度级） | **70** = 除四元数外逐分量残差 + rotvec 3 |
| 积分 | 半隐式，$c=\frac{d+1}{2d}$，$q_{j}'=q_j+(v_j+c\Delta v_j)dt$ | 无 dt：$s'=s+\delta$，$q'=\mathrm{norm}(\exp(\delta_{rot})\otimes q)$ |
| 水平位置 | 不预测（仅高度 $z$），平移不变 | 预测 $x,y$（env-local） |
| MaxCoord 表示 | 有：full 182 / rep 214 / pred 84（`representations.py:612`、`maxcoord_helpers.py:43-48`） | 无 |

### 4.2 仿真上下文 ξ（33 维）

| | 官方 | 本仓库 |
|---|---|---|
| 列序 | `[friction 1, com 3, encoder_bias 29]`（`sim_context.py:171-183`） | `[friction 1, encoder_bias 29, com 3]`（`state.py:173-182`） |
| 归一化 | 按 DR 范围解析归一，半宽 <1e-3 视为常数（scale=1）（`sim_context.py:26-31,189-191`） | **裸值**，仅在 WM 内过 `RunningNorm` |
| friction | 读首个匹配 geom | 左右足 geom 均值 |
| com | `body_ipos` | `body_ipos − default` |

第 2–30 列两侧语义互换，直接搬运 ξ 会错位。

### 4.3 观测与目标

| | 官方 | 本仓库 |
|---|---|---|
| proprio 5 项 | `[gravity3, ang_vel3, qj_rel29, vj29, action29]` = 93 | `[qj_rel29, vj29, ang_vel3, gravity3, action29]` = 93（**顺序不同**） |
| 历史 | 默认 1；`>1` 仅 proprio 模式允许 | 10（frame-major） |
| 每帧 goal | 58 + rel_pos 3 + rel_ori 6 = **67** | 58 + 3 + 6 + lin_vel 3 + ang_vel 3 = **73** |
| goal 总维 | actor 640 / wm 670 | 730 |
| 窗口 | 10 帧 × stride **5**（含当前帧 $t$，≈0.9 s 前瞻） | 10 帧 × stride **1**（$t{+}1..t{+}10$，≈0.2 s） |
| 关节量 | $q_j$ 绝对值 | $q_j-q_0$ |
| 平移项 | 相对机器人 root | 相对参考帧 $t$ 的 anchor |
| actor 输入 | 默认 68+640 = 708 | 930+730 = 1660 |
| critic 输入 | 68+670+33 = 771，**不归一化** | 71+730+33 = 834，整体 `RunningNorm` |

### 4.4 终止

| | 官方 | 本仓库 |
|---|---|---|
| 仿真侧 | $|\Delta z|>0.25$、$|\Delta g_z|>0.8$、超时 10 s；`ee_body_pos` 被移除 | 同 0.25/0.8 + 10 s，另加 `motion_end` 终止 |
| 想象侧 | 解析软阈值 $\sigma\!\big((\max(|\Delta z|/0.25,\ |\Delta g_z|/0.8)-1)/0.05\big)$，**零参数** | **BCE 学习头** $d_\omega=\sigma(\mathrm{MLP}([\hat s,\hat g]))$，bias 先验 5% |
| 损失项 | 无 | $+\lambda\,\mathrm{BCE}/k$ |

### 4.5 动作

官方 $\mathrm{proc}(a)=a\cdot s+b$（$b$=`default_joint_pos`）在 `Dynamics` **内部**执行，WM 看到的是物理关节命令；本仓库 WM 直接归一化 raw action，**没有 proc 变换**（env 侧才缩放）。这会让两边的动作归一化统计落在不同空间。

---

## 5. 模型侧差异

### 5.1 世界模型

| | 官方 | 本仓库 |
|---|---|---|
| MLP | `Linear→LayerNorm→GELU`，末层普通 init | `Linear→SiLU`，末层 orthogonal gain=0.01 |
| 归一化器 | `EmpiricalNormalization`：eps=**1e-2 加在 std 外**，无冻结维，eval 下 no-op | `RunningNorm`：eps=**1e-5 在根号内**，冻结四元数 4 维，无 train/eval 门禁 |
| 终止头 | 无 | `(256,128)`，bias 初值 $\log(0.05/0.95)$ |

### 5.2 WM 损失

官方在物理量空间按 Bryson 加权并摊到通道：

$$
\mathcal L_\text{O}=\frac{\sum_g w_g n_g\,\overline{\Delta_g^2}}{\sum_g n_g},\quad
w=(10^4,2500,100,0.1014),\ n=(3,9,29,29)
$$

每通道有效权重 $428.6,\ 321.4,\ 41.4,\ 0.042$ —— 根位置是关节速度的 $9.9\times10^4$ 倍。

本仓库在归一化空间等权 + 四元数 4 倍：

$$
\mathcal L_\text{L}=\frac1k\sum_{t=1}^{k}\Big[\overline{\Delta_{\ne q}^2}+4\,\overline{(\mathrm{sgn}\,q-\hat q)^2}\Big]+\frac{\lambda}{k}\mathrm{BCE}
$$

同一 `lr=2e-4`，**有效步长相差 2–3 个数量级**。另外官方只评 7 步（`range(T-1)`，最后一动作不计），本仓库评 8 步。

### 5.3 Actor / Critic / 优化器

| | 官方 | 本仓库 |
|---|---|---|
| actor | `MultiEncoderAE`：每任务一个 goal 编码器 + `slot_mix`；`latent_bc_weight=0.1`（默认 `slot_mix={motion:1.0, velocity:0.0}` ⇒ **aux 恒为 0**） | 单编码器；无 slot、无 BC |
| 输出 | decoder 线性输出，无 tanh/clamp（docstring 称 "post-clamp" 与代码不符） | 同；`min_std` 是死配置 |
| critic 值 | `clamp(max=v_max)`，$v_\max=0.1/(1-\gamma)=10$ | **无截断** |
| 优化器 | Adam(fused,capturable) + **Muon**（Sequential 中间 Linear，`match_rms_adamw`） | 纯 Adam（actor β=0.9/0.95） |
| lr 调度 | 恒定 | 余弦衰减至 0.03× @250k |
| PPO | 独立基线 runner（hidden 512/256/128、lr 1e-3、clip 0.2、KL 0.01、ent 0.005） | 无 |

---

## 6. 工程与流程差异

| 项 | 官方 | 本仓库 |
|---|---|---|
| 窗口采样 | `sample_causal` 用 `cummin(reachable)` 保证不跨 episode，398 行测试覆盖 | ring + terminal ring；跨 episode fallback（见审计 F5） |
| 训练循环 | CUDA graph 捕获 update，`torch.compile` 覆盖 rollout/loss/represent | 无 graph，compile 只用于 FK |
| 迭代预算 | 1e6，save 1000，log 10 | 320k，save 500，log 1 |
| 日志/权重 | W&B + model artifact + registry motion | 本地 logs，无 W&B |
| 部署 | ONNX（仅在 proprio 且 ≤15k 帧时导出），`term_major_to_temporal` 在导出边界转换 | ONNX + JSON sidecar + `deploy_features.py` 契约 + `evaluate` |
| 数据 | mjlab `MotionLoader`（npz）+ W&B registry artifact | 自建 preprocess（BONES-SEED / LAFAN1）→ 运动库 npz |

---

## 7. 各自独有

**官方独有**：MaxCoord 表示 + 解析 IK、velocity 任务与多任务 slot 混合、解析软终止、Muon、CUDA graph、W&B artifact、`viewers.py`(492)、`benchmark.py`、PPO 基线、sim context 存档进 `world_model.pt`。

**本仓库独有**：BONES-SEED/LAFAN1 预处理链、部署数值契约与 sidecar、确定性 evaluate、中文文档与视觉素材、三份审计报告。

---

## 8. 对本仓库审计结论的影响

1. **P1/P2 审计结论不可迁移到官方**：F1–F13、P2-* 全部指向本仓库独有的路径（学习终止头、`RunningNorm`、terminal ring、bin 单位、`index_copy_` 重复索引……），官方没有对应实现。
2. **也不能假设"官方即论文"**：官方已把式 (1) 的学习终止项换成解析阈值、奖励改为 FK 参考、加入 Muon/CUDA graph，比论文更工程化。
3. **"复现 Table A.5" 需限定**：goal 730（67→73 维、stride 5→1）、默认 proprio×10（官方默认 rep_state×1）三处均为自选细节。
4. **布局风险**：本仓库 `config/g1/env_cfgs.py` 的 proprio 组声明 `history_length=10, flatten_history_dim=True`（mjlab 展平为 term-major），而 `trainer.py` 自建特征为 frame-major —— 与审计 F13 同源，官方在 `representations.py:123-138` 专门处理了该转换。

---

## 9. 对齐官方的最小改动清单

1. goal：730 → 670/640（去每帧速度 6 维、stride 5、含当前帧、$q_j$ 绝对值、平移改 root 相对）。
2. 状态：71 → 68 gencoord（去 $x,y$、去 quat），输出 35，改半隐式解析积分（含 `substep_coeff`）。
3. 终止：删学习头与 BCE，改解析软阈值。
4. WM 输入：动作加 `proc = a·scale + offset`。
5. 损失：改用 Bryson 加权（或至少显式记录两种权重不可比）。
6. 归一化：eps 语义对齐（1e-2、加在 std 外、eval 不更新）。
7. 优化器：隐藏层 Linear 走 Muon，其余 fused capturable Adam。
8. 不可迁移：checkpoint、ONNX、部署契约、运动数据格式均需重新生成。

---

## 10. 机器上的第三份副本

`/home/zy0028/mjlab_project/mjlab/src/mjlab/tasks/supermimic/` 另有一份 vendored 拷贝，与本仓库有约 1716 行差异，缺 `deploy_features.py`/`export.py`，属抽取前旧版。若训练或审计时实际 import 的是这一份，本仓库的修复不会生效。

---

## 附录：证据索引

| 结论 | 证据 |
|---|---|
| goal 10×stride5 / 670 / 640 | `supermimic/representations.py:99-111,168` |
| GenCoord 68/35/半隐式积分 | `representations.py:469-490,505-554` |
| MaxCoord 182/214/84 | `representations.py:612`、`maxcoord_helpers.py:43-48` |
| ξ 列序与归一化 | `sim_context.py:26-31,171-192` |
| 软终止 σ 公式 | `envs/world_model_env.py:200-219` |
| 奖励 8 项 | `utils/tracking_reward.py:69-89` |
| WM 权重 1/nominal² | `learners/wm_learner.py:32-37,190` |
| 只评 7 步 | `learners/wm_learner.py:150-167` |
| Muon + Adam | `learners/learner_utils.py:13-36` |
| critic v_max | `learners/policy_learner.py:163,171,187` |
| RSI 参数 | `runners/reset_sampling.py:18-40` |
| 窗口 episode-safe | `replay_buffer.py:70-121` |
| 本仓库 71/70/73/730 | `algo/state.py:57-58`、`algo/networks.py:137`、`mdp/features.py:130-137` |
| 本仓库 ξ 裸值 | `algo/state.py:173-182` |
| 本仓库学习终止头 | `algo/networks.py:145-155,203-210` |
| 本仓库 WM 损失 | `algo/networks.py:212-224`、`algo/trainer.py:369-431` |
| 本仓库 critic λ-return | `algo/trainer.py:708-756` |
| 本仓库 RSI | `algo/sampler.py:9-27` |
