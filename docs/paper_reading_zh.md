# SuperMimic 论文精读：中文全译 + 公式与符号逐条详解

> **论文**：SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model
> **作者**：Qiayuan Liao, Yiyang Shao$^{*}$, Junfeng Long$^{*}$, Kevin Zakka, Roberto Horowitz, Pieter Abbeel$^{\dagger}$, Koushil Sreenath$^{\dagger}$（UC Berkeley）
> **出处**：CoRL 2026（Oral）｜arXiv:2610.28480v1（2026-10-20 提交）
> **项目页**：https://supermimic.github.io
> **本笔记范围**：正文 §1–§5 + 附录 A–G 的全部内容、全部公式（式 (1)–(3)、式 (A.1)、奖励/终止/域随机化的成组公式）以及逐符号字典。
> **本地文件**：PDF 在 `SuperMimic_arXiv_2610.28480.pdf`；LaTeX 源码在 `.tmp_supermimic/sections/` 下，可对照检查公式排版。
>
> **记号约定**：全文数学使用标准 LaTeX（行内 `$...$`、行间 `$$...$$`），公式编号与论文一致；$\hat{s}_t$ 表示世界模型的预测量，$x^{\text{des}}$ 表示参考动作给出的期望值。**带 💡 的段落是我的讲解/推断，不是论文原文**；论文原意用「」标注。

---

## 一、论文速览

### 1.1 四句话版本

| 问题 | 人形机器人「全身动作跟踪」要覆盖大规模 mocap 动作库，但现有 pipeline 靠 model-free RL（PPO），算力随动作数暴涨（SONIC：$128$ 张 L40 × 7 天 $\approx 21{,}000$ GPU·h）。 |
|---|---|
| 观察 | 动作各不相同，但**被跟踪的动力学是同一具身体**。model-free RL 不显式利用这个共享结构，每个动作都从零重新估计「动作 $\to$ 后果」。 |
| 方法 | 学**一个**可微世界模型 $f_\omega$（从标准刚体仿真器的 transition 日志里拟合），然后在它上面做**短视界 BPTT** 来更新策略 —— 一个模型把所有动作的学习「摊销」掉。 |
| 结果 | 单卡 H100 **3 小时**超过 SONIC 的归一化回报；GPU·h 降 **$>1000\times$**；$129{,}514$ 条动作里失败 25 条（SONIC 179 条），双卡 9.8 h 后**零失败**；真机 Unitree G1 完成跳跃、侧滚、$90^\circ$/$360^\circ$ 空翻。 |

### 1.2 贡献列表（原文 Contributions 逐条翻译）

1. **面向可部署人形控制的 MBRL**：证明「在学到的可微世界模型上做 BPTT」可以在长期被 model-free PPO 统治的设定里训练全身跟踪策略，且控制器能迁移到真实 Unitree G1。
2. **算力不再是 scaling 的瓶颈**：单卡超过 128 卡 PPO pipeline 的回报，GPU·h 降 $1000\times$ 以上；动作库扩大时训练成本增长很慢，使「不需要大集群也能研究 scaling law」成为现实。
3. **拆解收益来源**：三个世界模型性质 —— 预测误差在数据量 $264\times$ 增长下保持稳定、rollout 比仿真器快 $10\times$ 以上、解析梯度让策略用约 $64\times$ 更少的 imagined steps 达到 PPO-on-WM 的最好回报。
4. **超越蒸馏的下游控制**：同一个世界模型可以用下游接口**自己的任务奖励**来训练 goal encoder（速度指令任务上优于蒸馏 baseline）。

### 1.3 章节地图

| 位置 | 内容 | 本笔记对应 |
|---|---|---|
| §1 | Introduction + 贡献列表 | §一 / §二 |
| §2 | Related Work + 表 1（方法对比） | §2.5 |
| §3.1 | 训练循环、POMDP 记号、soft survival return | §3.1 |
| §3.2 | 世界模型：四个头 + $k$ 步自回归损失 | §3.2 / §6.2 |
| §3.3 | 策略目标 + critic/GAE + 与 SHAC/PPO 的关系 | §3.3 / §6.3–6.4 |
| §4.1–4.5 | 主结果、世界模型性质、critic 消融、真机、下游 | §五 |
| §5 | 结论 + 局限 | §5.6 |
| 附录 A | 观测/目标/动作/奖励/终止/域随机化的具体实例化 | §四 / §6.6 |
| 附录 B | 多 goal 接口联合训练（式 (A.1)） | §3.4 / §6.5 |
| 附录 C | 网络结构表 | §4.7 |
| 附录 D | 算法伪代码 + 超参 + return-based reset 采样 | §3.5 / §4.8 / §4.9 |
| 附录 E | 动作数据过滤（14.2 万 $\to$ 12.95 万条） | §4.10 |
| 附录 F | 评测协议与失败动作统计 | §5.1 |
| 附录 G | 速度指令任务与 step test | §5.5 |

---

## 二、为什么要做：算力瓶颈的来龙去脉

### 2.1 两条主流路线，都很贵

物理仿真里的动作模仿（physics-based motion imitation，DeepMimic 那条线）现在有两条规模化路线：

1. **多专家 + 蒸馏**：给每个（或每簇）动作训一个专家策略，再蒸馏进一个通用控制器。BeyondMimic 走这条，下游用 diffusion/classifier guidance 生成目标。
2. **直接训练一个通才策略**：在一个巨大的动作库上硬训一个 PPO 策略。SONIC 走这条，用 128 张 L40 训 7 天。

两条路线的共同点：**动作库越多样 $\to$ 控制器越强 $\to$ 算力越贵**。因为 model-free RL 把动力学当黑盒，它对「同一具身体的同一套动力学」的估计在每个动作、每个 batch 上都要重做一遍。

### 2.2 论文的关键观察（一句话的「摊销」论证）

> 「Although reference motions differ, they share the same humanoid hybrid rigid-body dynamics. Model-free RL fails to explicitly model this shared structure, instead repeatedly estimating action consequences from noisy simulator rollouts.」

翻译：参考动作虽然不同，但它们共享同一套人形「混合刚体动力学」（浮动基座 + 闭链/接触）。model-free RL 没有显式建模这个共享结构，而是反复从带噪的仿真 rollout 里估计动作后果。于是「扩展动作数量」就退化成「扩展算力」。

**💡 讲解**：这是全文的立论支点。它把问题从「怎么训得更快」重构成「同一份动力学被重复学习了 $N$ 次」。只要能把这份动力学**学一次、存下来、反复求导**，边际成本就从「每条动作一次完整 RL」降到「在共享模型上多跑几条 rollout」。这就是 amortize（摊销）这个词的来源。

### 2.3 为什么不用「可微仿真器」直接 BPTT

这是最自然的疑问：既然要梯度，为什么不直接对 MuJoCo 求导？

论文的回答（§3.2 开头）：

> 「The simulator's articulated multi-rigid-body dynamics are known, but the simulator need not provide gradients, so SuperMimic learns a differentiable surrogate... Because these gradients come from the learned model rather than from differentiating the simulator, SuperMimic remains compatible with rigid-body simulators, including those with non-smooth contacts and non-differentiable implementations.」

**💡 讲解**：接触（contact）是分段光滑甚至不连续的 —— 脚落地那一瞬间，梯度要么不存在，要么是冲激。可微仿真的 BPTT 在接触丰富的动作（侧滚、空翻）上极易炸。学一个**光滑的代理模型**（MLP），梯度永远良定义，代价是引入偏差（见 §9.2 的批判）。这是典型的 bias–variance / bias–tractability 交换。

### 2.4 算力预算对照（§4.1 Compute accounting）

| 项目 | SONIC（baseline） | SuperMimic |
|---|---|---|
| 硬件 | $128 \times$ NVIDIA L40 | $1 \times$ NVIDIA H100 (80 GB SXM5) |
| 墙钟时间 | 7 天 $=168$ h | 3 h |
| GPU·小时 | $\approx 21{,}000$ | 3 |
| 训练数据量 | 611 h 动作 | 264.56 h（BONES-SEED 公开子集） |
| 129,514 条基准上的失败数 | 179 | 25（3 h / 1 卡）$\to$ **0**（9.8 h / 2 卡） |
| 归一化回报 | 0.90 | 0.936（3.16 GPU·h 时） |

论文给出的比值与自我审计：

- 墙钟：$168/3 \approx \mathbf{56\times}$（文中保守写 ">50×"）
- GPU·h：$21{,}000/3 \approx \mathbf{7{,}000\times}$（文中保守写 ">1000×"，精确值约 $7{,}200\times$）
- 不公平因素自查：数据少 $2.3\times$；H100 峰值 fp32 吞吐是 L40 的 $0.74\times$，tensor core 是 $5.5\times$。$\Rightarrow$「$2.3\times$ 的数据差和最多 $5.5\times$ 的峰值算力差，解释不了 $1000\times$ 的差距。」

**💡 讲解**：这段审计写得相当坦诚，是这篇论文可信度的重要来源。但仍要注意：SONIC 是 128 卡规模下的工程实现，超参不是为单卡调的；「3 小时」是「回报超过 SONIC checkpoint」的时刻，不是收敛时刻；且两者评测的是**训练集动作**（论文自己承认，见 §9.2）。

### 2.5 相关工作：一张表看懂自己在哪个格子（表 1 中译）

论文把每个方法的 pipeline 拆成三个阶段来对比：

- **Mimic**：动作跟踪/模仿这一阶段用什么算法；
- **Latent**：潜在动作表示怎么训；
- **Downstream**：下游 goal 接口怎么训；
- **Task signal**：下游接口训什么信号 —— **RL**（接口自己的任务奖励）还是 **Distill**（参考跟踪信号）；
- **Real robot**：是否上过真机。

合并单元格 = 这三阶段**联合训练**；相邻未合并 = **串行**训练（前一个阶段冻结后再训下一个）。

| 方法 | Mimic | Latent | Downstream | Task signal | 真机 |
|---|---|---|---|---|---|
| NPMP | SVG(0) | BC | SVG(0) | RL | ✗ |
| ControlVAE | MBRL | MBRL（联合） | MBRL（联合） | RL | ✗ |
| PhysicsVAE | PPO | BC$^{1}$ | PPO | RL | ✗ |
| PULSE | PPO | DAgger | PPO | RL | ✗ |
| MaskedMimic | A2C | DAgger（联合） | DAgger（联合） | Distill | ✗ |
| BeyondMimic | PPO | DAgger | CGD | Distill | ✓ |
| SONIC | PPO | PPO（联合） | PPO（联合） | Distill$^{2}$ | ✓ |
| LATENT | PPO | DAgger | PPO | RL | ✓ |
| **SuperMimic** | **MBRL** | **MBRL（联合）** | **MBRL（联合）** | **Distill / RL** | ✓ |

「联合」= 该阶段与前一个阶段在同一个循环里训练（论文原表用合并单元格表示）；没有「联合」标记的相邻格 = 串行训练（前一阶段冻结后再训下一个）。

脚注原文：$^{1}$ 行为克隆 + 通过冻结的已学动力学模型的一步下一状态一致性项；$^{2}$ goal encoder 与跟踪器在**动作跟踪奖励 + latent 对齐损失**上联合训练，没有任务专属奖励。缩写：SVG(0) = stochastic value gradients；BC = 在固定专家 rollout 上的离线行为克隆（区别于 DAgger 的学生 rollout）；CGD = classifier guidance diffusion；A2C = advantage actor–critic；DAgger = dataset aggregation。

**💡 讲解**：这张表的信息量在于「**SuperMimic 是唯一一个三阶段用 MBRL 打通、且下游同时支持蒸馏和 RL、又上了真机**」的方法。SONIC 是「全 PPO + 蒸馏」，BeyondMimic 是「PPO + DAgger + diffusion 蒸馏」。表里「联合」和「串行」的区别，本质就是「共享动力学是否被复用」。

正文用四段文字补充定位（要点）：

- **大规模物理动作跟踪**：DeepMimic $\to$ PHC（大动作集 + 鲁棒恢复）$\to$ SONIC（数据/容量/算力一起 scale）。共同瓶颈是仿真 rollout 量。
- **学到的可微动力学用于动作控制**：SuperTrack、ControlVAE、MoConVQ、SuperDyno。SuperMimic 的增量是「单卡训练大规模人形跟踪器 + 真机部署」，且世界模型以**仿真上下文 $\xi$ 为条件**，部署时 actor 只看带噪本体观测 + goal。
- **基于模型的策略优化与部署**：Dreamer/DreamerV3（latent 想象 + actor-critic）、TD-MPC（短视界规划 + 终端价值）、DayDreamer（真机上直接学世界模型）、SHAC（可微仿真上的短视界 actor-critic，本文 critic 自举目标的来源）。RWM / RWM-U 也做「神经仿真器 + 真机迁移」，但用 **PPO** 而非 BPTT，且**预训练**世界模型；SuperMimic 是在训练过程中用自己的策略数据拟合。
- **可复用的动作表示与下游控制**：NPMP、PhysicsVAE、PULSE、LATENT、MaskedMimic（动作修补支持部分身体/多模态目标）、BeyondMimic（diffusion guidance）、SONIC（联合训练 encoder + latent 对齐）、ControlVAE（冻结技能控制器、用学到的模型优化下游策略）。SuperMimic 把「多 encoder + 共享 decoder + 共享动力学」放在一起联合训。

---

## 三、方法（§3）逐节精译与讲解

### 3.1 训练循环（§3.1，图 2，算法 A.1）

#### 原文精译

> SuperMimic 在一个多技能动作库上训练**一个** goal-conditioned 全身控制器，并支持多种 goal 接口。它把共享的人形动力学装进一个可微世界模型，并通过短的「想象 rollout」改进策略。训练在**同一个循环**里交替三个阶段（图 2）：
>
> 1. 当前策略在仿真里采集 transition 存入 replay buffer；
> 2. 在短的 replay 窗口上拟合可微世界模型；
> 3. 把「想象出来的回报」穿过**冻结的**世界模型反传，以 Dyna 风格从 buffer 里的状态出发做 rollout，从而改进策略。
>
> buffer、模型、策略在训练全程都在更新，所以动力学学习和控制是**一起**变好的，而不是分成两个阶段。**来自早期策略的 transition 依然可以用于拟合世界模型，因为动力学不依赖采集它的策略。**

**💡 讲解（第三阶段为什么能 work）**：这是 off-policy 动力学学习。策略更新（on-policy 的部分）只发生在**想象**里；真实数据只是用来拟合 $f_\omega$，而 $f_\omega$ 学的是「这具身体怎么动」，与「谁在动它」无关。所以数据可以一直复用，replay buffer 不必清空 —— 这是它比 PPO 省样本的结构性原因。

三段循环的示意（全部用公式写出）：

$$
\underbrace{\pi_\theta \xrightarrow{\;a_t+\epsilon_t\;} \text{Sim}}_{\text{(1) collect}}
\;\longrightarrow\;
\underbrace{(s_t,\;a_t+\epsilon_t,\;\xi)\;\longrightarrow\;\mathcal{B}}_{\text{to replay buffer}}
\;\xrightarrow[\;k\text{ steps}\;]{\text{(2) fit}}\;
\nabla_\omega \mathcal{L}_{\text{WM}}
$$

$$
\underbrace{\hat{s}_0=s_0\sim\mathcal{B},\quad
\hat{s}_{t+1}=f_\omega\!\left(\hat{s}_t,\;\pi_\theta(\hat{\bar{o}}_t+\eta_t,\,g_t),\;\xi\right)}_{\text{(3) improve policy: }\omega\text{ frozen, }H\text{ steps}}
\quad\Longrightarrow\quad
\theta \leftarrow \theta+\alpha_\theta\,\nabla_\theta \hat{J}(\theta)
$$

（critic $V_\psi$ 用同一条 rollout 的 GAE 目标拟合。）

#### POMDP 记号（逐句精译）

> 我们用 $\mathcal{M}$ 表示 goal-conditioned POMDP，采用紧凑记号：状态 $s_t$、目标 $g_t$、动作 $a_t$。
> 状态 $s_t$ 给出本体感受观测 $o_t$ 和终止标志 $d_t$（$s_t$ 为终止状态时 $d_t=1$）。
> 在状态 $s_t$ 执行动作 $a_t$ 得到奖励 $r_{t+1}$ 和下一状态 $s_{t+1}$。
> 策略是**确定性**的：$a_t=\pi_\theta(o_t,g_t)$，$a_t$ 参数化一个**归一化的关节设定点**；做动作跟踪时 $g_t$ 从参考动作里读取。
> 数据采集时，仿真器执行**带噪动作** $a_t+\epsilon_t$，$\epsilon_t\sim\rho_\epsilon$，replay buffer 存的是这个实际执行过的带噪动作。「噪声扩大了世界模型拟合所覆盖的动作范围；它只服务于模型拟合，是可选的。」
> 回报使用**软存活权重**：
>
> $$
> c_0=1,\qquad c_{t+1}=c_t\,(1-d_{t+1}),
> \qquad J(\theta)=\mathbb{E}\!\left[\sum_{t\ge 0}\gamma^{t}\,c_t\,r_{t+1}\right],\qquad \gamma\in[0,1)
> $$
>
> 仿真器记录**硬终止**，而学到的模型预测**终止概率** $\hat{d}_t\in[0,1]$，这让存活权重在想象 rollout 里保持**可微**。
> 全文用帽子标记仿真量的「世界模型对应物」。观测还被分解为 $o_t=\bar{o}_t+\eta_t$，其中 $\bar{o}_t$ 是无噪观测，$\eta_t\sim\rho_\eta$ 是已知的观测噪声，**在世界模型之外**采样。

**💡 三个要点**：

1. **动作噪声 $\epsilon_t$ 只是数据增强**。它让 buffer 里出现「策略本来不会选的动作」，从而让 $f_\omega$ 在动作空间里插值得更好，尤其是策略更新后会走到的邻域。BPTT 时用的是策略的**干净**输出 $a_t$（不加 $\epsilon$）。
2. **观测噪声 $\eta_t$ 放在模型外面重加**。模型内部是确定性的、无噪的，$\eta$ 在每次想象的每一步重新采样 —— 这样模型不用去学噪声，梯度路径也更干净。这相当于把「随机性」全部外置成显式的、已知的注入点。
3. **软终止 $\hat{d}_t\in[0,1]$** 是相对 SHAC 的关键改动。硬终止会让 $c_t$ 变成阶跃函数，梯度在某一步直接断掉；而可微的存活权重让策略能「预期到」自己将要失败，并把这种预期传回前面的动作（展开见下一节，以及 §6.3）。

#### 存活权重 $c_t$ 到底是什么

$c_t$ 是全文最容易「看一眼就跳过、但跳过就看不懂」的符号。它只干一件事：**在回合终止之后，把后续奖励从回报里乘掉**。它出现两次——真实目标 $J(\theta)$ 里（硬终止），想象目标 $\hat{J}(\theta)$ 里（软终止）。

**定义与等价形式**

$$
c_0=1,\qquad c_{t+1}=c_t\,(1-d_{t+1})
\quad\Longrightarrow\quad
c_t=\prod_{i=1}^{t}\big(1-d_i\big)
$$

因为 $d_i\in\{0,1\}$，它就是一个示性函数：

$$
c_t=\mathbf{1}\{\text{no termination in steps }1,\dots,t\}
=\mathbf{1}\{\text{episode alive at step }t\}
$$

**它解决的问题：无限视界求和的截断**

$$
J(\theta)=\mathbb{E}\Big[\sum_{t\ge 0}\gamma^{t}c_t r_{t+1}\Big]
=\mathbb{E}\Big[\sum_{t=0}^{T_{\text{end}}-1}\gamma^{t}r_{t+1}\Big]
$$

右边需要事先知道回合长度 $T_{\text{end}}$；左边不需要——它逐帧递推，终止之后自动乘 0。这一点在**想象 rollout** 里是刚需：终止时刻由模型在线给出，写 rollout 的时候你还不知道它会在哪一步结束。

**下标约定：为什么是 $c_t\,r_{t+1}$，不是 $c_{t+1}\,r_{t+1}$**

$r_{t+1}$ 是「$s_t\to s_{t+1}$」这一步的奖励，**进入终止状态那一步的奖励应当保留**。若写成 $c_{t+1}r_{t+1}$，终止步的奖励会被一起抹掉，策略会倾向于「不敢结束」。举一个 $d_3=1$（第 3 个状态是终止态）的例子：

| 量 | $t=0$ | $t=1$ | $t=2$ | $t=3$ | $t=4$ |
|---|---|---|---|---|---|
| $d_{t+1}$ | 0 | 0 | 1 | — | — |
| $c_t$ | 1 | 1 | 1 | 0 | 0 |
| 计入回报的奖励 | $r_1$ | $r_2$ | $r_3$ | 无 | 无 |

即：$r_3$ 由 $c_2=1$ 保留（走到终止态的那一步照拿），而 $r_4$ 由 $c_3=0$ 被剔掉。

**硬 vs 软：全文最关键的一次替换**

| | 数据采集（仿真） | 策略更新（想象） |
|---|---|---|
| 终止量 | $d_t\in\{0,1\}$，硬判决 | $\hat{d}_t\in[0,1]$，终止概率 |
| 存活权重 | $c_t=\prod_{i\le t}(1-d_i)$ | $\hat{c}_t=\prod_{i\le t}(1-\hat{d}_i)$ |
| 形状 | 阶跃：某步直接归 0 | 平滑衰减：$0.9,\ 0.81,\ 0.73,\dots$ |
| 对 $\theta$ 的梯度 | 终止处断掉 | 可微，能传回「导致失败」的动作 |

数值感受：若每步终止概率 $\hat{d}=0.05$、$H=32$，则

$$
\hat{c}_{32}=0.95^{32}\approx 0.19
$$

即「活着走完 32 步」的概率只有 19%，末端奖励被自动降权——越危险的状态越不被当作可靠的自举起点，这就是「软视界」的自适应效果。

**可微性带来什么：梯度**

$\hat{c}_t$ 是 $\theta$ 的光滑函数（通过 $\hat{d}$），所以

$$
\frac{\partial \hat{c}_t}{\partial\theta}
=-\sum_{i=1}^{t}\Big(\prod_{j<i}(1-\hat{d}_j)\Big)\frac{\partial \hat{d}_i}{\partial\theta}\prod_{i<k\le t}(1-\hat{d}_k)
$$

每一项的读法：**「第 $i$ 步终止概率上升」×「之前一路活着」×「之后一直活到第 $t$ 步」**。所以策略不是在摔倒那一刻才收到惩罚，而是在「摔倒概率上升」的那一刻就开始被扣分——这正是硬终止做不到的信用分配。

**与式 (3) 的关系：同一个东西的两处用法**

critic 的 GAE 递推里

$$
\hat{R}_t=\hat{r}_{t+1}+\gamma\,\big(1-\hat{d}_{t+1}\big)\Big[(1-\lambda)V_{t+1}+\lambda\hat{R}_{t+1}\Big]
$$

那个 $(1-\hat{d}_{t+1})$ 就是**逐步版**的 $c$：终止概率越高，自举项越弱，退化成 $\hat{R}_t\approx\hat{r}_{t+1}$。所以 $c_t$ 负责「回报求和」，$(1-\hat{d})$ 负责「价值自举」，两者是同一思想的两种写法。

**实现（顺序很重要）**

```python
c = 1.0
for t in range(H):
    d = d_head(s, g_t)            # d̂_t ∈ [0,1]
    c = c * (1.0 - d)             # ĉ_t   ← 先更新 c
    a = policy(o_t, g_t)
    r = r_head(s, a, g_t)         # r̂_{t+1}
    J += (gamma ** t) * c * r     # 再乘 r̂_{t+1}
    s = dyn(s, a, xi)
J += (gamma ** H) * c * V(s, g_H, xi)   # bootstrap 也乘 ĉ_H
```

两个细节：

1. **顺序**：先算 $\hat{d}_t$ 更新 $c$，再乘 $\hat{r}_{t+1}$（对应 $c_t\hat{r}_{t+1}$）。写反会差一步，末端奖励会错位。
2. **数值**：$c$ 是连乘，长视界下会 underflow，工程上在 log 空间累加更稳：$\log\hat{c}_t=\sum_{i\le t}\log(1-\hat{d}_i)$。

**为什么是连乘，不是取 min 或求和**

$c_t$ 语义上等于「前 $t$ 步都没终止」的**联合概率**；在各步独立的前提下，联合概率就是逐项相乘。取 min 会丢掉「部分步骤风险叠加」的信息，求和则没有概率意义。

**三个容易混的东西**

| 符号 | 是什么 | 不是什么 |
|---|---|---|
| $c_t$ | 存活权重（$1\to 0$ 的乘子） | 不是 cost，也不是折扣因子 |
| $\hat{c}_t$ | 想象中的**软**存活权重（可微） | 只有 $\hat{d}\in\{0,1\}$ 时才等于 $c_t$ |
| $V_\psi$ | critic，估计未来回报 | 不参与「是否终止」的判定 |

**💡 一句话**：$c_t$ 就是标准 RL 代码里的 `alive mask`（`mask = 1 - done` 的累乘）。SuperMimic 的新意不在这一项本身，而在于把它从**二值掩码**升级成**可微的生存概率**，从而让 BPTT 能把梯度传回「导致失败的动作」。这也是论文说「用存活权重取代 SHAC 的硬回合终止」的确切含义。

#### 为什么世界模型是确定性的

> 「Since the simulator's randomness has known sources, the world model stays deterministic: it conditions on the per-episode sim context $\xi$ (friction, calibration offsets, and torso center-of-mass) and predicts the noise-free observation $\bar{o}_t$.」

**💡 讲解**：这是「把随机性拆成"已知条件 + 显式噪声"」的设计。域随机化（DR）采样的物理参数 $\xi$ 是**每回合固定**的，把它作为条件输入，模型就学的是 $p(s_{t+1}\mid s_t,a_t,\xi)$ 的**确定性映射**；而动作噪声 $\epsilon$、观测噪声 $\eta$ 是逐步独立的，放在模型外。好处：BPTT 路径确定、可复现、无随机节点；模型容量不用浪费在建模噪声上。

### 3.2 学习可微世界模型（§3.2）

#### 四个头（正文中未编号的成组公式）

$$
\begin{aligned}
\hat{s}_{t+1} &= f_\omega(\hat{s}_t,\,a_t,\,\xi) &&\text{dynamics (learned)}\\
\hat{r}_{t+1} &= r_\omega(\hat{s}_t,\,a_t,\,g_t) &&\text{reward (learned / analytic)}\\
\hat{\bar{o}}_t &= g_\omega(\hat{s}_t,\,\xi) &&\text{observation (learned / analytic)}\\
\hat{d}_t &= d_\omega(\hat{s}_t,\,g_t) &&\text{termination (learned / analytic)}
\end{aligned}
$$

> 「世界模型包含一个学到的动力学模型，以及奖励、观测、终止三个头；每个头既可以被学习，也可以被解析计算。」

**💡 注意符号复用**：观测头叫 $g_\omega$，目标却叫 $g_t$，两者毫无关系，读的时候别混。另外附录 C 的网络表只列了动力学 $f_\omega$ 一个网络，说明**奖励与观测在实践中是解析计算的**（BeyondMimic 的奖励是状态与目标的闭式函数；观测是状态的子集），需要学的只有动力学 $f_\omega$ 和终止概率 $\hat{d}$（后者需要一个小的输出头）。论文没明说，但从「解析头没有损失项」+ 网络表可以推出来。

#### 世界模型损失（式 (1)）

$$
\mathcal{L}_{\text{WM}}(\omega)
=\mathbb{E}_{\mathcal{B}}\sum_{j=1}^{k}\Big[
\lVert \hat{s}_j-s_j\rVert^2
+(\hat{r}_j-r_j)^2
+\lVert \hat{\bar{o}}_j-\bar{o}_j\rVert^2
+\operatorname{BCE}(\hat{d}_j,\,d_j)
\Big]
\qquad (1)
$$

> 「我们在 **$k$ 步自回归 replay 窗口**（$k=8$）上拟合网络权重 $\omega$：以日志状态为种子 $\hat{s}_0=s_0$，沿**日志动作**向前滚动，最小化式 (1)。解析头没有对应的损失项。$r_j$ 是仿真器奖励，$d_j\in\{0,1\}$ 是仿真器终止标签，BCE 是二元交叉熵。」

**💡 三个关键细节**：

1. **状态自回归、动作取自日志**。滚动时 $\hat{s}_j$ 是模型自己的预测（不是真值 $s_j$）作为下一步输入 —— 也就是 **multi-step / 自回归**训练，模型必须学会纠正自己的漂移。这正是它后来能在 $H=32$ 甚至更长 rollout 上稳定的原因（见 §5.2）。而动作用的是日志里的真实动作，避免了「用自己策略的动作去预测别人的轨迹」的错配。
2. **四项损失同权重相加**（论文没给权重系数，字面就是 $1:1:1:1$），且都是**平方误差**，只有终止用 BCE。注意终止项的量纲与其他项不同，这是一个隐含的调参点。
3. **$k=8$ 步 $\approx 0.16$ s**（控制周期 $0.02$ s）。用 8 步窗口训练、却做 32 步想象，靠的是自回归拟合带来的外推稳定性 —— 论文把这归为经验观察（"In practice, the learned model remains stable over indefinitely long rollouts"），没有理论保证。

### 3.3 通过世界模型学习策略（§3.3）

#### 原文精译

> 每次策略更新时，我们**固定**世界模型权重 $\omega$，在 $H=32$ 步的想象 rollout 上优化 $\pi_\theta$ 和 $V_\psi$。critic $V_\psi$ 估计第 $H$ 步之后的回报，所以模型只需要在这 $H$ 步内准确。这个想象视界**超过了**拟合模型用的 $k$ 步窗口。
>
> 每条想象 rollout 从 replay buffer 里的状态 $s_0\sim\rho_0$ 及其记录的 $\xi$ 出发。在第 $t$ 步，模型给出 $\hat{\bar{o}}_t$，我们重新加入传感器噪声，策略行动：
>
> $$
> a_t=\pi_\theta\!\left(\hat{\bar{o}}_t+\eta_t,\;g_t\right),\qquad \eta_t\sim\rho_\eta
> $$

#### 策略目标（式 (2)）

$$
\hat{J}(\theta)
=\mathbb{E}_{s_0\sim\rho_0,\;\pi_\theta}\left[
\sum_{t=0}^{H-1}\gamma^{t}\,\hat{c}_t\,\hat{r}_{t+1}
+\gamma^{H}\,\hat{c}_H\,V_\psi(\hat{s}_H,\,g_H,\,\xi)
\right]
\qquad (2)
$$

> 「对于动作跟踪，奖励 $r_{t+1}$ 是 BeyondMimic 的动作跟踪奖励，去掉其中的自碰撞惩罚（附录 A）。」

#### BPTT vs PPO（重要的一段对比，逐句译）

> 「世界模型可微，所以 BPTT 通过世界模型对 $\hat{s}_t$ 和 $a_t$ 的**雅可比**，在这 $H$ 步想象上算出一阶梯度 $\nabla_\theta \hat{J}$。相比之下，PPO 把动力学当黑盒：它用**采样动作的优势**去加权随机策略的 score $\nabla_\theta\log\pi_\theta$，这是**似然比（likelihood-ratio）估计**。BPTT 则直接对每个动作求回报的导数，所以在短视界上梯度方差低得多。」

**💡 讲解**：这段是全文的技术核心之一，值得展开。

- PPO 的梯度：$\nabla_\theta J\approx\mathbb{E}\big[A(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big]$。它只通过「动作的概率」间接利用动力学，动力学本身不可微，所以必须靠大量采样把方差压下去。方差 $\propto 1/N$，且随视界增长。
- BPTT 的梯度：$\nabla_\theta J=\mathbb{E}\big[\sum_t(\partial\hat{r}/\partial a_t)(\partial a_t/\partial\theta)+\cdots\big]$，动力学以雅可比 $\partial f_\omega/\partial a_t$ 的形式进入。它是**解析**的（对代理目标而言），单条轨迹就能给出低方差估计。
- 代价：**偏差**。这是「surrogate 的梯度」，不是「真实仿真期望回报的梯度」。论文在 §3.2 明确承认："The resulting analytic policy gradient is the gradient of a surrogate for the expected return in simulation."

#### critic（式 (3)）

$$
\begin{aligned}
\mathcal{L}_V(\psi) &= \mathbb{E}\sum_{t=0}^{H-1}\big(V_t-\hat{R}_t\big)^2,\\
\hat{R}_t &= \hat{r}_{t+1}+\gamma\,(1-\hat{d}_{t+1})\Big[(1-\lambda)\,V_{t+1}+\lambda\,\hat{R}_{t+1}\Big],\\
V_t &= V_\psi(\hat{s}_t,\,g_t,\,\xi),\qquad \hat{R}_H=V_H
\end{aligned}
\qquad (3)
$$

> 「critic $V_\psi$ 用广义优势估计（GAE）回报拟合，这些回报**只在想象 transition 上**反向递推计算。递推从 $\hat{R}_H=V_H$ 开始，即 critic 对视界之外回报的估计。目标 $\hat{R}_t$ 每条 rollout 只算一次，在拟合 $\psi$ 期间保持不变。」

**💡 讲解**：$(1-\lambda)V_{t+1}+\lambda\hat{R}_{t+1}$ 就是 GAE($\lambda$) 的递推形式（$\lambda=1$ 退化为 Monte-Carlo，$\lambda=0$ 退化为一步 TD）。注意 $(1-\hat{d}_{t+1})$ 出现在**递推里**而不是只乘在奖励上 —— 它同时截断了自举路径，这和式 (2) 里的 $\hat{c}$ 是同一个「软存活」思想的两处体现。目标固定（不随 $\psi$ 更新而重算）是为了避免「追着自己的尾巴跑」。

#### 与 SHAC / 前人工作的关系（逐句译）

> 「策略与 critic 的更新合起来构成一个短视界的 actor–critic，思路与 SHAC 一致 —— SHAC 也是在可微仿真器上反传。SuperMimic 用存活权重 $\hat{c}_t$ 取代了 SHAC 的硬回合终止，且它的 critic **没有** EMA target network。此前的解析梯度动作跟踪控制器常常省掉 critic，只优化 $H$ 步奖励和。没有 critic 时训练也能达到相近的**标称**跟踪回报，但这个不自举的目标更**短视**。critic 在**受扰动之后**才显出价值 —— 恢复或平衡的奖励可能落在想象窗口之外（§4.3）。」

**💡 讲解**：这就是 §5.3 那个推人实验的理论预告。$H=32\times 0.02\ \text{s}=0.64\ \text{s}$ 的想象窗口，对「被推之后 2 秒才站稳」这种任务根本看不到回报，必须靠 critic 把视界外的价值 bootstrap 进来。实验也确实显示：无 critic 时 $H\le 16$ 直接训不出来；有 critic 时 $H=16$ 就能追平无 critic 的 $H=32$；而且在 $0.5$–$4$ m/s 的推力下，有 critic 的策略 $0.5$–$2.4$ s 内稳住，无 critic 的永远在踉跄，$4$ m/s 直接倒。

### 3.4 跨 goal 接口联合训练（附录 B，式 (A.1)）

> 每个 goal 接口实例化一个 POMDP $\mathcal{M}_i$（全参考动作、速度指令、末端位姿、SMPL 体姿……），各有自己的 goal 空间 $\mathcal{G}_i$、奖励 $r_i$、终止 $d_i$，各有一个 critic $V_{\psi_i}$。单个策略网络用**每接口一个 goal encoder** + **共享 decoder**：

$$
z=\operatorname{enc}_{\theta,i}(g)\in\mathcal{Z},
\qquad
a=\operatorname{dec}_\theta(z,\,o),
\qquad
g\in\mathcal{G}_i
\qquad (A.1)
$$

$$
\max_\theta\ \mathbb{E}_{i\sim p(\mathcal{I})}\big[\hat{J}_i(\theta)\big]
$$

> 每个接口因此有自己的策略
>
> $$
> \pi_i(o,g):=\operatorname{dec}_\theta\!\left(\operatorname{enc}_{\theta,i}(g),\,o\right)
> $$
>
> 速度 encoder 还额外接收观测历史的特征。每个 $\hat{J}_i$ 就是式 (2) 把 $\pi_\theta$ 换成 $\pi_i$、奖励/终止换成 $\mathcal{M}_i$ 的版本、critic 换成 $V_{\psi_i}$。**所有接口都通过同一个动力学模型 $f_\omega$ 训练，$f_\omega$ 不按接口重训。**

下游接口按奖励类型分两种训练方式：

- **蒸馏**：用动作跟踪奖励，指令从参考动作读取 $\to$ 学会「从这个指令复现参考动作」。
- **RL**：直接优化接口自己的任务奖励（如速度跟踪），穿过世界模型反传。

> 「在下游接口的 rollout 上，回报梯度穿过共享 decoder 传回该接口的 encoder，**decoder 的权重上有 stop-gradient**；decoder 仍由动作跟踪目标训练。」

**💡 讲解**：这是「一个 decoder 服务多个 encoder」的模块化设计，等价于多任务学习里的 shared trunk + task head。stop-gradient 是为了防止下游任务把共享 decoder 拉坏（灾难性遗忘）。论文在局限里也承认：因为速度奖励只更新 encoder，**「并发训练」其实等价于「先训跟踪器、再训 encoder」**，并发本身可能没必要 —— 这个自我批评很到位。

### 3.5 训练算法（附录 D，算法 A.1）

**Require**：世界模型 $f_\omega$；跨接口共享的策略 $\pi_\theta$；replay buffer $\mathcal{B}$。

**for** 每次迭代：

1. $\triangleright$ **Collect**：在仿真里用带噪动作 rollout $\pi_\theta$，transition 推入 $\mathcal{B}$（替换最旧的）；重置起点按 return-based sampling 抽取（附录 D.1）。
2. $\triangleright$ **Fit world model**：从 $\mathcal{B}$ 采一个 $k$ 步日志窗口，沿日志动作自回归滚动 $f_\omega$，
   $$
   \omega \leftarrow \omega-\alpha_\omega\,\nabla_\omega \mathcal{L}_{\text{WM}}
   $$
3. $\triangleright$ **Improve policy**：采样接口 $i\sim p(\mathcal{I})$ 与 $\mathcal{B}$ 里的一个起始状态，在冻结的 $f_\omega$ 上 on-policy 想象 $H$ 步：
   $$
   \hat{s}_{t+1}=f_\omega\!\left(\hat{s}_t,\;\pi_i(\hat{\bar{o}}_t+\eta_t,\,g_t),\;\xi\right)
   $$
   然后
   $$
   \theta\leftarrow\theta+\alpha_\theta\,\nabla_\theta \hat{J}_i
   $$
   （解析梯度穿过 $f_\omega$，见 §6.7。）并用这条想象 rollout 的 GAE 目标拟合 critic $V_{\psi_i}$。

> 「$\omega$ 和 $\theta$ 每次迭代都更新；$\omega$ 只在策略反传时被固定（stop-gradient）。对下游接口，$\hat{J}_i$ 只更新它的 encoder。」

---

## 四、工程实例化（附录 A/C/D）

### 4.1 观测 $o_t$

| 观测项 | 维度 | 噪声 $\eta_t$（逐维均匀分布） | 单位 |
|---|---|---|---|
| 关节位置（相对默认姿态）$q-q^{0}$ | 29 | $\mathcal{U}[-0.01,\,0.01]$ | rad |
| 关节速度 $\dot{q}$ | 29 | $\mathcal{U}[-0.5,\,0.5]$ | rad/s |
| 基座角速度（IMU） | 3 | $\mathcal{U}[-0.2,\,0.2]$ | rad/s |
| 投影重力（基座系） | 3 | $\mathcal{U}[-0.05,\,0.05]$ | — |
| 上一时刻动作 $a_{t-1}$ | 29 | 无 | — |

单帧 93 维；策略接收**最近 10 帧**（推算 930 维）。**注意没有线速度、没有高度、没有基座姿态四元数** —— 只有角速度 + 投影重力，这是经典的「廉价 IMU + 编码器」配置，真机可得的量。

### 4.2 目标 $g_t$

- 动作跟踪接口：从参考动作读取；与 BeyondMimic 不同，encoder 的条件是 $K=10$ 帧未来参考序列，而不是单帧目标。
- 其他接口用各自的指令表示（如速度指令 $(v_x,\,v_y,\,\omega_z)$）。

### 4.3 动作 $a_t$

$$
q^{\text{sp}}=\alpha\,a_t+q^{0}
$$

> 「动作映射为关节设定点 $q^{\text{sp}}=\alpha a_t+q^{0}$（相对默认姿态 $q^{0}$）。沿用 BeyondMimic 的做法，**低 PD 增益**让这个设定点成为**关节力矩的代理**，而不是期望位置。」

**💡 讲解**：这一句很关键，容易被忽略。PD 增益低 $\Rightarrow$ 设定点与实际位置的偏差决定力矩 $\Rightarrow$ 设定点 $\approx$ 力矩指令。所以策略输出虽然长得像「目标关节角」，实际作用更接近「力矩」。$\alpha$ 是动作缩放系数（BeyondMimic 里通常 0.25），论文未给具体值。

### 4.4 奖励（附录 A，表 A.2）

**任务项（跟踪）**，$\mathcal{T}$ 是被跟踪的 body 集合，$\lvert\mathcal{T}\rvert$ 是其基数：

| 项 | 公式 | 权重 |
|---|---|---|
| body 位置 | $\exp\!\Big(-\frac{1}{\vert\mathcal{T}\vert}\sum_{b\in\mathcal{T}}\frac{\lVert p_b^{\text{des}}-p_b\rVert^{2}}{0.3^{2}}\Big)$ | $1.0$ |
| body 姿态 | $\exp\!\Big(-\frac{1}{\vert\mathcal{T}\vert}\sum_{b\in\mathcal{T}}\frac{\lVert \log(R_b^{\text{des}}R_b^{\top})\rVert^{2}}{0.4^{2}}\Big)$ | $1.0$ |
| body 线速度 | $\exp\!\Big(-\frac{1}{\vert\mathcal{T}\vert}\sum_{b\in\mathcal{T}}\frac{\lVert v_b^{\text{des}}-v_b\rVert^{2}}{1.0^{2}}\Big)$ | $1.0$ |
| body 角速度 | $\exp\!\Big(-\frac{1}{\vert\mathcal{T}\vert}\sum_{b\in\mathcal{T}}\frac{\lVert \omega_b^{\text{des}}-\omega_b\rVert^{2}}{3.14^{2}}\Big)$ | $1.0$ |
| root 位置 | $\exp\!\Big(-\lVert p_{\text{root}}^{\text{des}}-p_{\text{root}}\rVert^{2}/0.3^{2}\Big)$ | $0.5$ |
| root 姿态 | $\exp\!\Big(-\lVert \log(R_{\text{root}}^{\text{des}}R_{\text{root}}^{\top})\rVert^{2}/0.4^{2}\Big)$ | $0.5$ |

**正则项**：

| 项 | 公式 | 权重 |
|---|---|---|
| 动作平滑 | $\lVert a_t-a_{t-1}\rVert^{2}$ | $-0.1$ |
| 关节限位 | $\sum_j\big[\max(0,\,q_j-q_j^{\max})+\max(0,\,q_j^{\min}-q_j)\big]$ | $-2$ |

> 「用 BeyondMimic 的动作跟踪奖励，去掉自碰撞惩罚。那个惩罚是为了防随机策略的 reward hacking：当参考动作让两个 body 靠近但不相接触时，把它们压在一起会**抑制动作噪声**并抬高速度跟踪奖励。我们的策略在想象 rollout 里是确定性的，这个 exploit 没有收益，所以该项不必要。关节限位惩罚从 $-10$ 降到 $-2$。」

**💡 讲解**：这段是很好的「奖励设计随算法改变」的案例。自碰撞惩罚在 PPO 里必须存在，因为随机策略会通过「挤压 + 噪声平均」骗奖励；但这里的策略在想象中确定性执行，噪声只在模型外重加，所以 exploit 消失。**换了算法，奖励的防作弊项就可以删** —— 这种耦合关系在复现时很容易被忽略。

三个量纲说明：$\exp(-\text{MSE}/\sigma^{2})$ 的值域是 $(0,1]$，完美跟踪 $=1$，是「软准确度」，天然 dense；分母 $0.3/0.4/1.0/3.14$ 就是各量的容差尺度（角速度用了 $\pi^{2}\approx 9.87$，约等于 $180^\circ/\text{s}$ 的平方）；所有项最后再乘控制周期 $\Delta t=0.02\ \text{s}$。

### 4.5 终止

$$
\lvert e_{p,z}\rvert>0.25\ \text{m}
\qquad\text{or}\qquad
e_{\text{tilt}}>0.8
$$

> 「$e_{\text{tilt}}$ 是**参考系与机器人 root（pelvis）系中重力方向 $z$ 分量之差的绝对值**。对竖直参考，$e_{\text{tilt}}$ 在约 $78^\circ$ 倾角时达到 $0.8$。与 BeyondMimic 不同，我们**不**因末端执行器偏差而终止。」

**💡 推导**：取重力单位向量在体系下 $u_z=-\cos\theta$（直立时 $u=(0,0,-1)$），则

$$
e_{\text{tilt}}=\lvert u_z^{\text{ref}}-u_z\rvert=\lvert 1-\cos\theta\rvert
$$

令 $=0.8$ 得 $\cos\theta=0.2$，$\theta=78.5^\circ$ ✓。这也解释了速度任务里 $1+u_z>0.7$ 的写法：$1+u_z=1-\cos\theta$，阈值 $0.7$ $\Rightarrow$ $\theta\approx 72.5^\circ\approx 73^\circ$。

### 4.6 域随机化 = sim context $\xi$（附录 A，表 A.3）

| 参数 | 采样分布 |
|---|---|
| 摩擦系数 $\mu$ | $\mathcal{U}[0.3,\,1.2]$ |
| 关节标定偏置 $\Delta q^{0}_{j}$ [rad] | $\mathcal{U}[-0.01,\,0.01]$ |
| 躯干质心偏置 [m] | $\Delta x\sim\mathcal{U}[-0.025,\,0.025]$，$\ \Delta y,\Delta z\sim\mathcal{U}[-0.05,\,0.05]$ |

> 「我们**去掉**了 BeyondMimic 的 root 速度扰动：这类推力会注入状态**不连续**，确定性的世界模型无法表达，而实验发现只对物理参数做随机化就足以迁移到硬件。」

**💡 讲解**：这是全文最值得玩味的设计决策之一。**为了可微性牺牲了一类扰动**（推人扰动），然后靠 critic + 真实世界的鲁棒性来补。这既是优点（模型干净），也是局限（世界模型学不到「被推」这种事件，只能通过 $\xi$ 里的静态参数变化来间接泛化）。

### 4.7 网络结构（附录 C，表 A.4）

| 网络 | 输入 | 输出 | 结构（隐层宽度） |
|---|---|---|---|
| 动力学 $f_\omega$ | $\hat{s}_t,\ a_t,\ \xi$ | $\hat{s}_{t+1}$ | MLP (2048, 2048, 2048) |
| 动作 encoder $\operatorname{enc}_{\theta,\text{motion}}$ | $K=10$ 帧未来参考 | $z$ | MLP (2048, 1024, 512, 256) |
| 速度 encoder $\operatorname{enc}_{\theta,\text{vel}}$ | 归一化指令 $(v_x,v_y,\omega_z)$ + 最近 10 帧 $o$ 的特征 | $z$ | MLP (256, 256) |
| 共享 decoder $\operatorname{dec}_\theta$ | $z$ + 最近 10 帧 $o$ | $a_t$ | MLP (2048, 1024, 512, 256) |
| critic $V_{\psi_i}$ | $\hat{s}_t,\ g_t,\ \xi$ | 标量 | MLP (256, 128)，每接口一个 |

潜在 $z$ 是 **32 维**，用 $\tanh$ 限幅。

**💡 注意**：critic 的输入是**状态** $\hat{s}_t$（不是观测 $o_t$），而且带 $\xi$ —— 它是「上帝视角」的价值函数，因为在想象里状态是全知的。部署时只需要 actor。

### 4.8 超参（附录 D，表 A.5）

| 类别 | 超参 | 值 |
|---|---|---|
| 世界模型 | 窗口长度 $k$ | $8$ |
| | 优化器 / 学习率 | Adam；$\alpha_\omega=2\times 10^{-4}$（1 卡），2 卡时 $\sqrt{2}\times 2\times 10^{-4}$ |
| | batch | 每次更新 10 个 batch × 2048 窗口 |
| | 梯度裁剪 | 全局范数 $1.0$ |
| 想象与策略 | 视界 $H$ | $32$ |
| | 优化器 | Adam，$\beta=(0.9,\,0.95)$ |
| | 学习率 $\alpha_\theta$ | 同 $\alpha_\omega$ |
| | batch | 每次更新 4096 条想象 rollout |
| | 梯度裁剪 | 全局范数 $1.0$，再逐元素 $0.1$ |
| | 折扣 $\gamma$ | $0.99$ |
| | 传感器噪声 | 见 §4.1 表 |
| critic | GAE $\lambda$ | $0.95$ |
| | 每次更新 epoch 数 | 5，每个 epoch 4 个打乱的小 batch |
| 数据采集 | 每 GPU 仿真环境数 | 4096 |
| | 动作噪声 $\epsilon_t$ | 每动作维 $\mathcal{N}(0,\,0.1^{2})$ |
| | buffer 大小 | 每个环境 1000 步 |
| 调度 | update-to-data 比 | $0.4$ |
| | 每次世界模型更新配几次策略更新 | $1$ |
| | 学习率调度 | $250{,}000$ 次迭代内 cosine 衰减到 $0.03\times$，之后恒定 |
| | 训练长度 | $320{,}000$ 次迭代 |
| | 多卡 | 数据并行，梯度取平均 |

### 4.9 return-based reset 采样（附录 D.1）

> 仿真回合重置到动作库中的某些时间点，**偏向策略当前回报低的点**。动作时间轴切成 1 秒的 bin，每个 bin $b$ 维护一个折扣回报的运行估计 $G_b$，用策略自己的 rollout 更新。重置按三个分布混合：

| 比例 | 采样分布 |
|---|---|
| 50% | 在 bin 上均匀 |
| 40% | $\operatorname{softmax}_b\!\left(-25\,G_b/G_{\max}\right)$，其中 $G_{\max}$ 是理论最大折扣回报 |
| 10% | 按各 bin 过去 softmax 权重的比例（学好的难动作仍保留一些练习） |

**💡 讲解**：这是「hard example mining」的轻量实现，但第三项（10% 的惯性）很聪明 —— 纯 softmax 会导致「学会了就再也不练」，然后慢慢遗忘；保留 10% 的历史权重相当于给难动作一个不会归零的下限。对动作库这种「长尾分布 + 会遗忘」的场景，这个 trick 值得抄。

### 4.10 动作数据过滤（附录 E）

原始 BONES-SEED（G1 retarget）**142,220** 条 $\to$ 名称过滤后 **129,667** $\to$ 可行性过滤后 **129,514**。

1. **名称过滤**（去掉 12,553 条）：clip 名字里含 `stair` / `ladder` / `box` / `chair` / `sit` / `climb` / `jump_off_50cm` / `look_over_fence` 等「需要地形或物体」的词（部分借用了 SONIC 发布的数据处理代码里的关键词表）。
2. **可行性过滤**（再去掉 153 条）：对每条动作跑 MuJoCo 正运动学，看身体 link 与足部几何的轨迹，定义
   $$
   z_{\min}(t)=\min_{b}\,z_b(t)
   $$
   （最低 body link 的高度，地面 $z=0$），标记三类：
   - **悬空支撑**：$z_{\min}$ 在 $0.20$ m 以上持续 $\ge 0.6$ s（站在/爬上箱子）$\to$ 不可行。
   - **穿地**：$z_{\min}<-0.12$ m，或整条 clip 有 80% 以上时间 $<-0.08$ m（retarget 错误把腿压进地面）。
   - **双脚悬空**：至少 $0.5$ s 内**两只脚**都离地 $>0.02$ m（用 MuJoCo 精确的 geom-to-plane 距离 `mj_geomDistance`，不是 link 原点代理），同时骨盆竖直、处于站立高度、身体近乎静止。自由站立的人形在平地上不可能双脚离地还静止不动 $\to$ 无论原因是低平台、retarget 浮空还是缓慢起身，都判为不可行（弹道式跳跃会被「静止」这一条件排除）。干净的地面站立是 $-0.004\pm 0.004$ m，所以 $0.02$ m 的门限离地面有好几个 $\sigma$。

> 所有阈值都是为 G1 retarget @ 50 Hz 调的。

**💡 讲解**：这段工程价值很高，但也是最「不可移植」的部分 —— 换机器人、换控制频率、换 retarget 工具，阈值全要重调。注意它用的是**几何距离**而不是 link 原点，这个细节决定了「脚底贴地」判得准不准。

---

## 五、实验结果（§4）逐节精译

### 5.1 主结果（§4.1）

实验设置：仿真器 **MuJoCo Warp**（MuJoCo 的 GPU 实现）；所有成本在 NVIDIA H100（80 GB SXM5，Lambda 按需实例）上测。跑了多个随机种子，训练非常稳定（归因于动作跟踪的 dense reward），所以图上不画种子方差。

> **归一化回报（normalized return）**：「整条回合内每步奖励的**未折扣**和，除以这个和可能取到的最大值。所以提前终止的策略得分低。」
>
> $$
> \text{normalized return}=\frac{\sum_{t=0}^{T-1} r_{t+1}}{\max_{\pi}\ \sum_{t=0}^{T-1} r_{t+1}}
> $$

主结果（129,514 条动作的共享基准）：

| 指标 | SONIC（released ckpt） | SuperMimic 1×H100, 3 h | SuperMimic 2×H100, 9.8 h |
|---|---|---|---|
| 归一化回报（129,514 条基准） | $0.90$ | $0.936$ @ 3.16 GPU·h | $0.964$ @ 19.6 GPU·h |
| 失败动作数 | $179$ | $25$ | $\mathbf{0}$ |

> 「我们保守地说 3 小时：此后每个 checkpoint 在回报和失败数上都至少领先 SONIC 与 SuperMimic 最佳结果之间差距的一半。」

**按动作组拆失败数**（表 A.8）：

| 动作组（条数） | SONIC | SM 1×3h | SM 2×9.8h |
|---|---|---|---|
| 侧滚（22） | **22** | 2 | 0 |
| 空翻：$360^\circ$/$180^\circ$/$90^\circ$（44） | **44** | 1 | 0 |
| fire-in-the-hole 转身（20） | 3 | 3 | 0 |
| 交叉腿（10） | 9 | 0 | 0 |
| 躺/歇/起身（99） | 28 | 8 | 0 |
| 跳远（24） | 15 | 0 | 0 |
| 舞蹈 routine（8） | **8** | 0 | 0 |
| 其余（129,287） | 50 | 11 | 0 |
| **合计（129,514）** | **179** | **25** | **0** |

> 「SONIC 失败于整组的动作（全部 44 个空翻、全部 22 个侧滚、某舞蹈 routine 的全部 8 条）以及 24 个跳远中的 15 个。它的 v1.1 checkpoint 失败 239 条……3 小时后 SuperMimic 的失败集中在少数几组难动作：表里前五组共 195 条，占了 25 次失败中的 14 次。**在任何一组里它的失败数都不多于 SONIC**。9.8 小时后零失败。」

> 「基准动作对两种方法都是**训练动作**，所以这些计数衡量的是各自策略对这些训练动作的覆盖程度，而**不是**对新动作的泛化。」

**💡 这句话必须加粗记住**。媒体标题式的「零失败」是「把训练集全部动作都学会了」，不是「没见过的新动作也能做」。

**受控对比**（图 4a，LAFAN1 40 条、2.45 h）：

- 基线 **PPO-Sim**：同样的动作、观测、奖励、终止、DR、actor/critic 规模、评测协议，用 BeyondMimic 的 PPO 超参、每卡 8192 并行环境（是 SuperMimic 的两倍）。
- SuperMimic 单卡 **1.2 小时**达到 $0.82$ 回报；PPO-Sim 在 18 小时预算内**达不到**（最终：1 卡 $0.60$，4 卡 $0.70$，8 卡 $0.72$ —— 4 $\to$ 8 卡几乎无收益）。

### 5.2 世界模型的三个性质（§4.2）

**(a) 一个模型摊销整个动作库。**

> 「数据集从 1 h 增长到 264 h，世界模型的 **8 步预测误差**始终在 $0.007$–$0.011$ 之间。每条动作都由同一具人形执行，所以新增 clip 提供的是关于共享身体与接触动力学的**更多证据**，而不是要求更大容量。这与「达到目标回报所需算力增长很慢」一致。」

**(b) 想象让 rollout 变快**（图 5a）。

| 并行环境数 | 世界模型 (steps/s) | MuJoCo Warp (steps/s) | 加速 |
|---|---|---|---|
| 2,048 | $8.43\times 10^{5}$ | $7.84\times 10^{4}$ | $11\times$ |
| 8,192 | $3.35\times 10^{6}$ | $1.97\times 10^{5}$ | $17\times$ |
| 16,384 | $6.71\times 10^{6}$ | $2.61\times 10^{5}$ | $26\times$ |
| 65,536 | $2.00\times 10^{7}$ | $3.22\times 10^{5}$ | $62\times$ |

> 一个 step = 一个 50 Hz 控制步，MuJoCo Warp 用 4 个 5 ms 子步计算；世界模型跑在训练所用的 TF32 精度上。实践中用的 batch 区间（2k–16k）里就快 $10\times$ 以上。

**(c) 一阶梯度需要的想象步数少得多**（图 5b，机制消融）。

> 「SuperMimic 用它的 actor–critic 更新，以约 **$64\times$ 更少**的 imagined steps 达到 PPO-WM（在世界模型 rollout 上跑 PPO）的最好训练回报。」

**💡 讲解**：这三个性质正好对应三种成本：**样本成本**（预测误差稳定 $\to$ 数据不用指数增长）、**时间成本**（rollout 快 $10$–$60\times$）、**梯度效率**（解析梯度少 $64\times$ 步）。三者相乘就是那 $1000\times+$ 的来源。这个「把收益分解到三个可测量维度」的写法，是这篇论文方法学上最漂亮的地方。

### 5.3 critic 消融与抗推（§4.3，图 4b/4c）

| 视界 $H$ | 有 critic：达到回报 $0.82$ 的小时数 | 无 critic |
|---|---|---|
| $8$ | 未达到 | 未达到 |
| $16$ | $1.19$ | 未达到 |
| $\mathbf{32}$（默认） | $\mathbf{0.88}$ | $1.37$ |
| $64$ | $1.70$ | $1.34$ |
| $128$ | $4.62$ | $3.80$ |

- 无 critic 时 $H\le 16$ 训练失败；有 critic 时 $H=16$ 追平无 critic 的 $H=32$；$H=32$ 最快达到 $0.82$。
- $H$ 从 $64$ 加到 $128$，时间几乎三倍、最终回报更低；$H=128$ 是能在 H100 上放下的最长视界。
- **同样回报下行为也不同**：在仿真里保持静态 T-pose、施加 $0.5$–$4$ m/s 的推力，有 critic 的策略在 $0.5$–$2.4$ s 内稳下来；无 critic 的策略**从不**稳住 —— $0.5$ m/s 就一直在踉跄，$4$ m/s 直接倒。

**💡 讲解**：$H=32\to 0.88$ h，$H=64\to 1.70$ h，$H=128\to 4.62$ h：成本随 $H$ 超线性增长（显存 $O(H\cdot B)$ + 反向图变长），而收益在 $H=32$ 附近饱和。这就是「短视界 + critic」的工程最优解。

### 5.4 真机迁移（§4.4，图 6）

> 一个通用动作跟踪器，**不做逐动作调参**，部署到真实 Unitree G1。四个动作覆盖：移动（跳跃式 skip）、接触丰富的全身动作（躯干与四肢贴地的侧滚）、杂技（$90^\circ$ 空翻、原地 $360^\circ$ 空翻）。

### 5.5 下游任务：RL encoder vs 蒸馏（§4.5，图 7，附录 G）

协议：从站立开始给固定指令 5 s，每个指令测 4 次；**相对稳态误差** = 最后 2 s 的平均速度误差的范数 / 指令范数（root 速度在 heading 系里量）；每条指令取「没摔倒的那些 trial」的中位数：

$$
e_{\text{ss}}=\frac{\big\lVert \overline{v}_{[T-2,\,T]}-v^{\text{cmd}}\big\rVert}{\lVert v^{\text{cmd}}\rVert}
$$

672 条 step response $=3\times(15\times15-1)=3\times224$（三个平面，每个平面去掉 1 条站立指令）。误差 $>0.9$ 按 $0.9$ 画；四次全摔的策略画在 cap 上。

结果：

- RL encoder 在 **650** 条上误差更低，蒸馏 encoder 在 **15** 条上更低（剩下 **7** 条是蒸馏「四次全摔」、无法比较的快转弯指令，且动作库里几乎没有这类数据）。
- 中位相对误差：**0.16**（RL）vs **0.49**（蒸馏）。
- 速度指令范围：$v_x\in[-1.5,\,2]$，$v_y\in[-1.32,\,1.32]$ m/s，$\omega_z\in[-3.35,\,3.35]$ rad/s，10% 概率置零（站立）。
- 速度任务的奖励表（表 A.6）：

$$
\begin{aligned}
r_{\text{lin}} &= \exp\!\Big(-\lVert v_{xy}-v_{xy}^{\text{cmd}}\rVert^{2}/0.5^{2}\Big), && w=2.0\\
r_{\text{yaw}} &= \exp\!\Big(-(\omega_z-\omega_z^{\text{cmd}})^{2}/0.5\Big), && w=2.0\\
r_{\text{smooth}} &= \lVert a_t-a_{t-1}\rVert^{2}, && w=-0.3\\
r_{\text{limit}} &= \sum_j\big[\max(0,q_j-q_j^{\max})+\max(0,q_j^{\min}-q_j)\big], && w=-2\\
r_{\text{lean}} &= \sin^{2}\phi_{\text{torso}}, && w=-4\\
r_{\text{rate}} &= \omega_{\text{roll}}^{2}+\omega_{\text{pitch}}^{2}, && w=-0.05
\end{aligned}
$$

  终止：$h_{\text{root}}<0.35$ m 或 $1+u_z>0.7$（约 $73^\circ$ 倾角）。
- latent 对齐：在有参考指令的 rollout 上，以 $0.1$ 权重的 MSE 把速度 latent 拉向同一时刻动作 encoder 的 latent（动作 latent 上有 stop-gradient）：
  $$
  \mathcal{L}_{\text{align}}=0.1\cdot\big\lVert z_{\text{vel}}-\operatorname{sg}(z_{\text{motion}})\big\rVert^{2}
  $$
- **每个策略只有一个 seed** —— 下游结论的统计强度有限，论文也把它定位为 "preliminary study"。

### 5.6 结论与局限（§5）

结论：SuperMimic 把算力从 scaling 的瓶颈上移除。一个学到的可微世界模型抓住所有动作共享的动力学，BPTT 把整个数据集摊销成一个策略。单卡训练、超过 128 卡集群训出的策略、迁移到真机 G1。成本随数据量增长很慢。**「测量动作跟踪的 scaling law 现在只需一次一张卡」**。

局限（三条原文）：

1. **多接口训练研究不足**。速度奖励只更新速度 encoder、不更新共享 decoder（附录 B），所以「并发训练」等价于「先训跟踪器再训 encoder」，并发可能不必要；若让速度奖励也更新 decoder，并发是否有帮助**尚未验证**。
2. **只有仿真动力学**。世界模型只从仿真 transition 学；真机数据可以 refine 它，它的可微性还能支持系统辨识。
3. **没有地形和物体交互**。只在平地上训练；扩展到地形与物体交互是正在进行的工作。

---

## 六、公式全集：逐条精解

### 6.1 编号索引

| 编号 | 内容 | 位置 |
|---|---|---|
| （无编号，成组） | 世界模型四个头 $f_\omega,\,r_\omega,\,g_\omega,\,d_\omega$ | §3.2 |
| **(1)** | 世界模型损失 $\mathcal{L}_{\text{WM}}$ | §3.2 |
| (无编号，正文) | 存活权重 $c_0=1,\ c_{t+1}=c_t(1-d_{t+1})$；$J(\theta)$（详解见 §3.1「存活权重 $c_t$ 到底是什么」） | §3.1 |
| (无编号，正文) | 观测分解 $o_t=\bar{o}_t+\eta_t$；动作噪声 $a_t+\epsilon_t$ | §3.1 |
| **(2)** | 想象策略目标 $\hat{J}(\theta)$ | §3.3 |
| **(3)** | critic 损失与 GAE 目标 $\mathcal{L}_V(\psi)$、$\hat{R}_t$ | §3.3 |
| **(A.1)** | 多接口编码/解码与提升目标 | 附录 B |
| (无编号，表 A.2) | 动作跟踪奖励 6 项 + 2 个正则项 | 附录 A |
| (无编号，表 A.3) | 域随机化 3 组分布（$=\xi$） | 附录 A |
| (无编号，表 A.6) | 速度跟踪奖励 6 项 + 终止 | 附录 G |
| (无编号，正文) | $q^{\text{sp}}=\alpha a_t+q^{0}$；$e_{\text{tilt}}$；$z_{\min}(t)$；return-based reset 的三个分布 | 附录 A/D |

### 6.2 式 (1)：世界模型损失

$$
\mathcal{L}_{\text{WM}}(\omega)
=\mathbb{E}_{\mathcal{B}}\sum_{j=1}^{k}\Big[
\lVert \hat{s}_j-s_j\rVert^2
+(\hat{r}_j-r_j)^2
+\lVert \hat{\bar{o}}_j-\bar{o}_j\rVert^2
+\operatorname{BCE}(\hat{d}_j,\,d_j)
\Big]
\qquad (1)
$$

**逐项**：

- $\mathbb{E}_{\mathcal{B}}$：对 replay buffer 里采样的 **$k$ 步窗口**取期望（窗口起点随机）。
- $\lVert\hat{s}_j-s_j\rVert^{2}$：**多步**状态预测误差。注意 $\hat{s}_j$ 是把模型自己上一步的输出 $\hat{s}_{j-1}$ 喂回去得到的（自回归），不是单步预测。这是整篇论文最关键的训练细节 —— 它让模型学会「在自身误差上继续前进」，从而支持 $H=32>k=8$ 的想象。
- $(\hat{r}_j-r_j)^{2}$：奖励标量误差；若奖励解析计算则此项不参与（无梯度）。
- $\lVert\hat{\bar{o}}_j-\bar{o}_j\rVert^{2}$：无噪观测的重建误差。因为 $\bar{o}$ 是 $s$ 的子集，这一项在实践中通常解析给出（相当于状态损失的一部分），只有当真值 $\bar{o}$ 与状态不完全一致（如含滤波/延迟）时才需要单独学。
- $\operatorname{BCE}(\hat{d}_j,d_j)$：终止的二元交叉熵，$d_j\in\{0,1\}$。它让终止成为**概率**而不是判决，是「软存活」得以可微的前提。
- **无权重系数**：字面四项等权相加。实际复现时通常要给 BCE 一个系数（否则量纲不一致）。

**为什么不加 KL / 不确定性项？** 这是确定性模型，没有隐变量，所以没有 ELBO/KL；也没有 ensemble 或 epistemic uncertainty 惩罚（对比 RWM-U）。这让训练简单，但也失去了「知道自己不知道」的能力（见 §9.2）。

### 6.3 式 (2)：想象策略目标

$$
\hat{J}(\theta)
=\mathbb{E}_{s_0\sim\rho_0,\;\pi_\theta}\left[
\sum_{t=0}^{H-1}\gamma^{t}\,\hat{c}_t\,\hat{r}_{t+1}
+\gamma^{H}\,\hat{c}_H\,V_\psi(\hat{s}_H,\,g_H,\,\xi)
\right]
\qquad (2)
$$

**逐项**：

- $s_0\sim\rho_0$：起始状态取自 replay buffer（其 $\xi$ 一同取出）。$\rho_0$ 不是均匀分布 —— 是 §4.9 那个 return-based 混合（偏向低回报的时间 bin）。
- $\mathbb{E}_{\cdot,\,\pi_\theta}$：期望覆盖策略自身的随机性 —— 但策略是确定性的，实际随机性只来自每步重采的 $\eta_t$（以及 buffer 起点），所以这个期望在实现上是对 $\eta$ 和起点的期望。
- $\sum_{t=0}^{H-1}\gamma^{t}\hat{c}_t\hat{r}_{t+1}$：$H$ 步的**折扣、存活加权**回报。其中
  $$
  \hat{c}_t=\prod_{i=1}^{t}\big(1-\hat{d}_i\big),\qquad \hat{c}_0=1
  $$
  是「活到第 $t$ 步」的概率（软指示函数）。定义、下标约定与实现的完整推导见 §3.1「存活权重 $c_t$ 到底是什么」。
- $\gamma^{H}\hat{c}_H V_\psi(\hat{s}_H,g_H,\xi)$：**bootstrap 项**，用 critic 估计视界之外的回报。$\hat{c}_H$ 乘在这里，等价于「如果已经（软）终止，就不再自举」。
- $g_H$：第 $H$ 步的目标。动作跟踪时它是参考动作在第 $H$ 步的帧，所以自举要用**未来**的目标 —— 这是 goal-conditioned critic 必须显式接收 $g$ 的原因。
- $H=32$（$0.64$ s）、$\gamma=0.99$（有效视界 $\approx 100$ 步，但被 $H$ 截断，所以 critic 必不可少）。

**💡 与标准 actor-critic 的差异**：标准做法是

$$
\sum_t \gamma^t r_{t+1}+\gamma^H V(s_H)
$$

这里把 $c$ 插进每一项。$c$ 可微（因为 $\hat{d}$ 是概率），所以梯度能穿过「将要失败」这件事，传回到导致失败的动作上。这是 SHAC 的硬终止做不到的。

### 6.4 式 (3)：critic 与 GAE

$$
\begin{aligned}
\mathcal{L}_V(\psi) &= \mathbb{E}\sum_{t=0}^{H-1}\big(V_t-\hat{R}_t\big)^2,\\
\hat{R}_t &= \hat{r}_{t+1}+\gamma\,(1-\hat{d}_{t+1})\Big[(1-\lambda)\,V_{t+1}+\lambda\,\hat{R}_{t+1}\Big],\\
V_t &= V_\psi(\hat{s}_t,\,g_t,\,\xi),\qquad \hat{R}_H=V_H
\end{aligned}
\qquad (3)
$$

**逐项**：

- $V_t$：critic 在想象轨迹第 $t$ 步的预测。
- $\hat{R}_t$：GAE($\lambda$) 的**回报**形式（不是优势）。$\lambda=0.95$ 控制「bootstrap 与 Monte-Carlo 的混合比例」。
- $(1-\hat{d}_{t+1})$：软终止因子。若 $\hat{d}=1$，则 $\hat{R}_t=\hat{r}_{t+1}$，不再向后自举 —— 语义正确。
- 递推从 $\hat{R}_H=V_H$ 开始（视界末端的自举）。
- 目标**每条 rollout 只算一次并冻结**，然后对 $\psi$ 做 5 个 epoch × 4 个小 batch 的回归。冻结是为了避免「目标随预测移动」的稳定性问题（类似 DQN 的 target network，但这里是「一次性冻结」而不是 EMA —— 论文明确说 "no exponential-moving-average target network"）。

**💡 一个容易读错的下标**：式 (3) 里 $\hat{d}_{t+1}$ 需要 $\hat{s}_{t+1}$，而式 (1) 定义的是 $\hat{d}_t=d_\omega(\hat{s}_t,g_t)$。自洽的读法是：每一步先算 $\hat{s}_{t+1}$，再算 $\hat{d}_{t+1}$，再算 $\hat{c}_{t+1}=\hat{c}_t(1-\hat{d}_{t+1})$，然后进入下一步。论文在正文里没把这个时序写死，复现时要自己定一个一致约定（我建议：状态 $\to$ 终止 $\to$ 存活权重 $\to$ 奖励 $\to$ 策略）。

### 6.5 式 (A.1)：多接口

$$
z=\operatorname{enc}_{\theta,i}(g)\in\mathcal{Z},
\quad a=\operatorname{dec}_\theta(z,o),
\quad g\in\mathcal{G}_i,
\qquad
\max_\theta\ \mathbb{E}_{i\sim p(\mathcal{I})}\big[\hat{J}_i(\theta)\big]
\qquad (A.1)
$$

$$
\pi_i(o,g):=\operatorname{dec}_\theta\big(\operatorname{enc}_{\theta,i}(g),\,o\big)
$$

- $i$：接口索引，$\mathcal{I}$ 是接口集合，$p(\mathcal{I})$ 是其上的分布（速度实验里 $p$ 均匀，且每个想象 batch 混合两个接口）。
- $\mathcal{Z}$：32 维 $\tanh$ 限幅的共享潜在空间。
- 提升目标（lifted objective）：对所有接口的期望。每个 $\hat{J}_i$ 用自己的奖励/终止/critic。
- 关键：**$f_\omega$ 只有一份**，不按接口重训。「同一个动力学，多个目标语言」—— 这是它相对「每接口一个端到端策略」的结构性优势。

### 6.6 奖励公式的通用形式

所有任务项都是**指数核**：

$$
r=\exp\!\left(-\frac{\lVert e\rVert^{2}}{\sigma^{2}}\right)\in(0,1]
$$

- $\sigma$ 是容差尺度：$\sigma$ 小 $\to$ 奖励对误差敏感、梯度尖锐；$\sigma$ 大 $\to$ 平滑但区分度低。
- 用指数而不是负平方，是为了让奖励**有界**（不会因为一次大误差就把整条回合的回报打成负无穷），且「接近完美时梯度自然变小」，数值上更稳。
- 所有项最后乘 $\Delta t=0.02$ s，所以单步奖励量级是 $O(0.01\text{–}0.1)$，整条回合（假设 10 s = 500 步）的回报量级是 $O(10)$，再除以理论上限得到 $[0,1]$ 的归一化回报。
- 姿态误差用 $\log(R^{\text{des}}R^{\top})$：旋转矩阵的 **log map**，得到轴角向量，其范数就是测地角（弧度）。所以 $\lVert\log(\cdot)\rVert^{2}$ 是「角度平方」。
- 关节限位用 $\max(0,\,q-q^{\max})$ 的**单边** hinge：只有越界才罚，且是线性的（不是平方），对小幅越界更宽容。

### 6.7 补：BPTT 展开（我的推导，非论文原文）

论文只说「BPTT 通过 $f_\omega$ 对 $\hat{s}_t$ 和 $a_t$ 的雅可比算一阶梯度」。把它写开有助于理解为什么它省样本、为什么显存是瓶颈。

记

$$
A_t=\frac{\partial f_\omega}{\partial \hat{s}_t},\quad
B_t=\frac{\partial f_\omega}{\partial a_t},\quad
G_t=\frac{\partial g_\omega}{\partial \hat{s}_t},\quad
P_t=\frac{\partial \pi_\theta}{\partial o},\quad
Q_t=\frac{\partial \pi_\theta}{\partial \theta}
$$

（均在 $(\hat{s}_t,a_t,\xi)$ 处求值），并记 $J_t=\dfrac{\partial \hat{s}_t}{\partial\theta}\in\mathbb{R}^{n_s\times\lvert\theta\rvert}$，$J_0=0$（因为 $s_0$ 来自 buffer、与 $\theta$ 无关）。

**前向**（每步）：

$$
\begin{aligned}
\hat{\bar{o}}_t &= g_\omega(\hat{s}_t,\xi), &\quad o_t&=\hat{\bar{o}}_t+\eta_t\\
a_t &= \pi_\theta(o_t,g_t), &\quad \frac{\partial a_t}{\partial\theta}&=P_t\,G_t\,J_t+Q_t\\
\hat{s}_{t+1} &= f_\omega(\hat{s}_t,a_t,\xi), &\quad J_{t+1}&=A_t\,J_t+B_t\big(P_t\,G_t\,J_t+Q_t\big)
\end{aligned}
$$

最后一行就是「雅可比沿轨迹传播」。

**反向**（对式 (2) 求导，先忽略 $\hat{c}$ 的梯度以简洁）：

$$
\nabla_\theta\hat{J}
=\mathbb{E}\sum_{t=0}^{H-1}\gamma^{t}\hat{c}_t\left[
\frac{\partial \hat{r}_{t+1}}{\partial a_t}\frac{\partial a_t}{\partial\theta}
+\frac{\partial \hat{r}_{t+1}}{\partial \hat{s}_t}J_t
\right]
+\gamma^{H}\hat{c}_H\left[\frac{\partial V_H}{\partial \hat{s}_H}J_H+\frac{\partial V_H}{\partial\theta}\right]
$$

（$V$ 若不以 $\theta$ 为输入，末项为 $0$。）

$\hat{c}_t$ 也依赖 $\theta$（通过 $\hat{d}$），完整版再多一项 $\hat{r}_{t+1}\,\partial\hat{c}_t/\partial\theta$，其中

$$
\frac{\partial\hat{c}_t}{\partial\theta}
=-\sum_{i\le t}\hat{c}_{i-1}\frac{\partial\hat{d}_i}{\partial\hat{s}_i}J_i\prod_{j\ne i}\big(1-\hat{d}_j\big)
$$

**成本**：前向 $H$ 次 MLP + 一次反向，显存 $O(H\cdot B\cdot n_s)$（要存每步的激活和雅可比链）。这解释了为什么 $H=128$ 是 H100 的上限、$H$ 从 32 到 128 时间涨 $5\times$。

**为什么方差低**：$\nabla_\theta\hat{J}$ 是**对动作的解析导数**，不是对采样动作的加权平均。PPO 的 $A(s,a)\nabla\log\pi$ 里 $A$ 本身是估计量、$\nabla\log\pi$ 是随机的，两者相乘方差大；这里 $\partial\hat{r}/\partial a$ 和 $\partial f/\partial a$ 都是确定的光滑函数，方差主要来自 $s_0$ 与 $\eta$ 的采样。

### 6.8 其他散落的公式

| 公式 | 含义 |
|---|---|
| $c_0=1,\ c_{t+1}=c_t(1-d_{t+1})$ | 硬终止下的存活指示；$c_t=\prod_{i\le t}(1-d_i)$ |
| $J(\theta)=\mathbb{E}\big[\sum_{t\ge 0}\gamma^{t}c_t r_{t+1}\big]$ | 真实（仿真）目标；与式 (2) 的 $\hat{J}$ 区别只在「仿真 vs 模型」「硬 vs 软终止」 |
| $o_t=\bar{o}_t+\eta_t$ | 观测 = 无噪观测 + 已知噪声 |
| $a_t+\epsilon_t$ | 仿真实际执行的动作；buffer 存这个 |
| $q^{\text{sp}}=\alpha a_t+q^{0}$ | 动作 $\to$ 关节设定点（低 PD 增益 $\Rightarrow$ 力矩代理） |
| $e_{\text{tilt}}=\lvert u_z^{\text{ref}}-u_z^{\text{robot}}\rvert$ | 倾角误差；直立参考下 $=1-\cos\theta$，$0.8\Rightarrow 78.5^\circ$ |
| $z_{\min}(t)=\min_b z_b(t)$ | 数据过滤用的最低 link 高度 |
| $\operatorname{softmax}_b(-25\,G_b/G_{\max})$ | reset 采样的难例权重 |
| $\text{NR}=\sum_t r_t\,/\,\max\sum_t r_t$ | 评测指标（未折扣、整条回合） |
| $\exp\!\big(-\lVert v_{xy}-v_{xy}^{\text{cmd}}\rVert^{2}/0.5^{2}\big)$ | 速度跟踪奖励（平面速度） |

---

## 七、符号字典（逐符号）

### 7.1 时间、索引、集合

| 符号 | 含义 | 取值/维度 |
|---|---|---|
| $t$ | 时间步（控制步，50 Hz，$\Delta t=0.02$ s） | 整数 |
| $j$ | 世界模型损失里的窗口内步索引 | $1,\dots,k$ |
| $k$ | 世界模型训练的自回归窗口长度 | $8$ |
| $H$ | 想象视界（BPTT 展开步数） | $32$（默认） |
| $i$ | goal 接口索引 | 属于 $\mathcal{I}$ |
| $\mathcal{I}$ | goal 接口集合；$p(\mathcal{I})$ 是其上的分布 | — |
| $b$ | 被跟踪的 body（奖励里）/ 时间 bin（reset 采样里） | — |
| $\mathcal{T}$ | 被跟踪 body 的集合；$\lvert\mathcal{T}\rvert$ 其基数 | — |
| $\mathcal{B}$ | replay buffer（固定大小，新 transition 替换最旧） | 每环境 $1000$ 步 |
| $\mathcal{M},\ \mathcal{M}_i$ | goal-conditioned POMDP（第 $i$ 个接口） | — |
| $\mathcal{G}_i$ | 第 $i$ 个接口的 goal 空间 | — |
| $\mathcal{Z}$ | 共享潜在空间 | 32 维，$\tanh$ 限幅 |

### 7.2 状态、观测、动作

| 符号 | 含义 | 维度 |
|---|---|---|
| $s_t$ | 仿真器状态（真值）；提供 $o_t$ 与 $d_t$ | 论文未明说；应含 $q,\dot{q}$、基座位姿/速度 |
| $\hat{s}_t$ | 世界模型预测的状态（想象的当前状态） | 同 $s_t$ |
| $s_0$ | 想象 rollout 的起点（取自 buffer） | — |
| $\rho_0$ | 起点分布（return-based reset 混合） | — |
| $o_t$ | 本体感受观测 | $93$（10 帧 $\to 930$） |
| $\bar{o}_t$ | 无噪观测（模型预测 $\hat{\bar{o}}_t$） | $93$ |
| $\eta_t\sim\rho_\eta$ | 传感器噪声（模型外重加） | 见表 A.1 |
| $a_t$ | 动作（归一化关节设定点） | $29$ |
| $a_{t-1}$ | 上一动作（观测的一部分） | $29$ |
| $\epsilon_t\sim\rho_\epsilon$ | 动作噪声（只用于数据采集） | $\mathcal{N}(0,0.1^{2})$ |
| $q,\ \dot{q}$ | 关节位置/速度 | $29$ |
| $q^{0}$ | 默认关节姿态 | $29$ |
| $q^{\text{sp}}$ | 关节设定点 | $29$ |
| $\alpha$ | 动作缩放系数 | 未给（BeyondMimic 常用 $0.25$） |
| $q_j^{\max},\ q_j^{\min}$ | 第 $j$ 关节的限位 | 标量 |
| $p_b,\ R_b,\ v_b,\ \omega_b$ | body $b$ 的位置/姿态/线速度/角速度 | $3$ / $3\times 3$ / $3$ / $3$ |
| $(\cdot)^{\text{des}}$ | 上标：参考动作给出的期望值 | — |
| $u_z$ | 体系下重力单位向量的 $z$ 分量 | $[-1,1]$ |
| $h_{\text{root}}$ | root 高度 | m |

### 7.3 目标与奖励

| 符号 | 含义 |
|---|---|
| $g_t$ | 目标（动作跟踪时 = 参考动作第 $t$ 步） |
| $K$ | 未来参考帧数（encoder 条件）$=10$ |
| $r_{t+1}$ | 仿真器奖励（真值）；$\hat{r}_{t+1}$ 模型预测 |
| $d_t$ | 硬终止标志 $\{0,1\}$；$\hat{d}_t\in[0,1]$ 终止概率 |
| $c_t$ | 硬存活权重 $=\prod(1-d)$；$\hat{c}_t$ 软存活权重 $=\prod(1-\hat{d})$ |
| $\gamma$ | 折扣因子 $=0.99$ |
| $\lambda$ | GAE 参数 $=0.95$ |
| $\phi_{\text{torso}}$ | 躯干 $z$ 轴的侧倾角（仅速度任务） |
| $\omega_{\text{roll}},\ \omega_{\text{pitch}}$ | 躯干 roll/pitch 角速度（仅速度任务） |
| $v_{xy},\ \omega_z$ | 平面 root 速度、世界系偏航角速度（速度任务） |

### 7.4 世界模型与优化

| 符号 | 含义 |
|---|---|
| $\omega$ | 世界模型参数（动力学 + 各头） |
| $f_\omega$ | 动力学头 $\hat{s}_{t+1}=f_\omega(\hat{s}_t,a_t,\xi)$ |
| $r_\omega$ | 奖励头 |
| $g_\omega$ | 观测头（**与 goal $g_t$ 无关**） |
| $d_\omega$ | 终止头 |
| $\xi$ | 每回合的仿真上下文（摩擦、标定偏置、躯干 CoM 偏置） |
| $\mathcal{L}_{\text{WM}}$ | 世界模型损失（式 (1)） |
| $\alpha_\omega$ | 世界模型学习率 $=2\times 10^{-4}$ |
| $\mu$ | 摩擦系数 $\sim\mathcal{U}[0.3,1.2]$ |
| $\Delta q^{0}_{j}$ | 关节标定偏置 $\sim\mathcal{U}[-0.01,0.01]$ rad |
| $\Delta x,\Delta y,\Delta z$ | 躯干 CoM 偏置（m） |

### 7.5 策略与值函数

| 符号 | 含义 |
|---|---|
| $\theta$ | 策略参数（含所有 encoder + 共享 decoder） |
| $\pi_\theta$ | 确定性策略 $a_t=\pi_\theta(o_t,g_t)$ |
| $\pi_i$ | 第 $i$ 接口的策略 $\operatorname{dec}_\theta(\operatorname{enc}_{\theta,i}(g),o)$ |
| $\operatorname{enc}_{\theta,i},\ \operatorname{dec}_\theta$ | 接口 $i$ 的 goal encoder / 共享 decoder |
| $z$ | 32 维 goal 潜在 |
| $\psi,\ \psi_i$ | critic 参数（每接口一个） |
| $V_\psi$ | critic $V_\psi(\hat{s}_t,g_t,\xi)$ |
| $V_t$ | 简写 $V_\psi(\hat{s}_t,g_t,\xi)$ |
| $\hat{J}(\theta),\ \hat{J}_i(\theta)$ | 想象目标（式 (2)） |
| $J(\theta)$ | 真实（仿真）目标 |
| $\hat{R}_t$ | GAE 回报目标 |
| $\mathcal{L}_V$ | critic 损失（式 (3)） |
| $\alpha_\theta$ | 策略学习率（同 $\alpha_\omega$） |

### 7.6 易混符号对照（重要）

| 看起来像 | 但其实是 | 区别 |
|---|---|---|
| $g_\omega$（观测头） | $g_t$（目标） | 一个是网络，一个是条件输入 |
| $\alpha$（动作缩放） | $\alpha_\omega$、$\alpha_\theta$（学习率） | 上下标不同，含义完全不同 |
| $\mathcal{G}_i$（goal 空间） | $G_b$（reset 采样的回报估计）、$G_{\max}$ | 三处 $G$ |
| $\mathcal{T}$（被跟踪 body 集合） | $H$（视界） | 论文用 $H$ 表示视界，$\mathcal{T}$ 只用于 body 集合 |
| $\hat{d}_t$（终止概率） | $d_\omega$（终止头） | — |
| $\hat{r}$（预测奖励） | $r_\omega$（奖励头） | — |
| $\rho_\epsilon$（动作噪声分布） | $\rho_\eta$（观测噪声）、$\rho_0$（起点分布） | 三个 $\rho$ |
| $K$（未来帧数 $=10$） | $k$（自回归窗口 $=8$） | **大小写敏感，极易混** |

---

## 八、复现清单与 mjlab 映射

### 8.1 最小复现路径

1. **仿真侧（mjlab / MuJoCo Warp）**：G1，29 DoF，控制 50 Hz，物理 200 Hz（$4\times 5$ ms 子步）；BeyondMimic 奖励（去掉自碰撞、关节限位 $\to -2$）；DR 三组（摩擦/标定/CoM），**不要**加 root 速度扰动。
2. **数据落盘**：每个 env step 存 $(s_t,\ a_t+\epsilon_t,\ \xi,\ r_{t+1},\ d_{t+1},\ \bar{o}_t)$。关键是 $s_t$ 要用**完整仿真状态**（不只是观测），因为世界模型预测的是状态；观测只是它的子集。buffer 1000 步/env。
3. **世界模型**：PyTorch/JAX 里的 $3\times 2048$ MLP（$n_s+29+\lvert\xi\rvert\to n_s$），Adam $2\times 10^{-4}$，TF32；训练时 8 步自回归、状态反馈、动作取日志。
4. **想象 + BPTT**：$H=32$，batch 4096，冻结 $\omega$，`torch.autograd` 直接反传（MLP 的 BPTT 不需要手写雅可比）；critic 单独 $256\times 128$；GAE 目标每条 rollout 算一次。
5. **调度**：每次迭代 1 次 WM 更新（$10\times 2048$ 窗口）+ 1 次策略更新；UTD $0.4$；320k 迭代；cosine 到 $0.03\times$。
6. **已知的模糊点**（论文没写死，复现要自己定）：
   - $s_t$ 的**确切定义**（是否含接触力/摩擦状态？基座位姿用四元数还是 6D？）；
   - $\xi$ 的**向量化方式**（摩擦是按脚还是按 geom？维度多少？）；
   - $\alpha$（动作缩放）的具体值；
   - $\hat{d}$ 头的结构（网络表里没列）；
   - 奖励/观测头是解析还是学习（从表 A.4 看应是解析）；
   - $\hat{d}$ 与 $\hat{c}$ 的严格时序（见 §6.4）。

### 8.2 与 mjlab 的对应关系

| 论文组件 | mjlab 里对应 |
|---|---|
| MuJoCo Warp | mjlab 的后端就是 MuJoCo Warp（`mujoco_warp`） |
| G1 29 DoF / 50 Hz / $4\times5$ ms | mjlab 的 G1 环境配置可直接对齐 |
| BeyondMimic 奖励 + 终止 | mjlab 的 tracking 任务（奖励项与 $e_{\text{tilt}}$/高度阈值） |
| 4096 并行环境 | mjlab 的 `num_envs`（需按显存调） |
| replay buffer / 世界模型 / BPTT | **不在 mjlab 内**，需在 PyTorch/JAX 侧另建 |

**💡 一个现实的工程判断**：mjlab 负责「快速产生高质量 transition」，论文的收益主要来自「把 transition 的利用率提高 $1000\times$」。所以复现的重点不在仿真器，而在 (a) buffer 的吞吐（4M transitions 的读写）、(b) BPTT 的显存（$H=32\times B=4096\times n_s$）、(c) 世界模型的训练稳定性（自回归 8 步 + 4 项损失等权，容易在某些状态维度上失衡，建议做归一化）。

### 8.3 已有的开源复现（外部资源，未验证）

- `DiaoOrg/awesome-supermimic`（汇总）
- `CuboiLeo/SuperMimic-Reproduced`（声称基于 mjlab / MuJoCo Warp 复现 G1 动作跟踪）
- `zita-ch/supermimic_vibed`（声称 1.5 GPU·h 跑通 LAFAN）

（以上为搜索到的第三方仓库，质量与论文一致性需自行验证。）

---

## 九、我的评价

### 9.1 强在哪

1. **问题重构得漂亮**。「同一具身体的动力学被重复学习」这个观察一旦说破，就很容易接受；而它同时给出了三个可独立测量的收益来源（预测误差 / rollout 速度 / 梯度效率），使论证不是黑箱的「我们更快」。
2. **工程细节诚实**。算力审计主动列出数据量差 $2.3\times$、GPU 型号差 $5.5\times$；评测明确说是**训练集动作**；下游实验自称 preliminary；局限里承认「并发训练其实等价于串行」。这种自我拆台在 RL 论文里不多见。
3. **可微性设计克制**。没有去可微化接触，而是学一个光滑代理；把随机性（$\epsilon,\ \eta,\ \xi$）全部显式外置。这让整个系统在数学上非常干净：确定性模型 + 显式噪声注入 = 可复现的 BPTT。
4. **软终止 + critic 的配合**。这两个改动单独看都很小，但合起来解决了「短视界 + 稀疏失败信号」的老问题，并且在推人实验里给出了**行为层面**的证据（不只是回报数字）。

### 9.2 可疑 / 值得追问的地方

1. **代理梯度偏差没有任何界**。论文只给「8 步预测误差 $0.007$–$0.011$」这一个间接指标，但**策略梯度对模型误差的敏感度**没有被测量。一个模型可以在平均 MSE 上很好，却在「决定成败的那几个状态维度」（如脚接触瞬间）上很差。缺少 per-dimension 误差分解或「模型误差 $\to$ 回报差」的敏感性分析，是这篇论文理论上的最大空洞。
2. **零失败 $\ne$ 泛化**。129,514 条全部是训练动作；论文自己说了，但「0 failures」这个数字在传播中一定会被误读。
3. **单 seed**。主实验说多 seed 稳定但不画方差；下游实验明确只有一个 seed。对一个「$1000\times$ 加速」的强声明，seed 数是审稿人会追问的点。
4. **baseline 的公平性**。PPO-Sim 用了 BeyondMimic 的超参、每卡 8192 envs（是 SM 的两倍），但没有为单卡/8 卡重新调参；SONIC 是 128 卡的工程实现。论文的「$>1000\times$」有真实成分，但「单卡 3 h vs 128 卡 7 天」更像是「算法效率 × 工程规模」的复合结果，不宜简单归因于 MBRL。
5. **世界模型学不到「事件」**。去掉 root 速度扰动是为了确定性，但这意味着世界模型对「被推」「踩到异物」这类**状态跳变**完全没有概念；真机鲁棒性靠的是 DR 出来的静态参数泛化 + critic 的保守性，而不是模型本身。这在更动态的环境里会成为瓶颈。
6. **$H$ 的上限是显存**。$H=128$ 已经是 H100 极限，而实验显示 $H=128$ 反而更差 —— 说明「更长视界」这条路在当前架构下走不通，必须靠 critic。这是一个有价值的负面结果，但论文没有深挖「为什么 128 变差」（是优化问题还是模型误差累积？）。
7. **奖励的可移植性**。自碰撞惩罚被删掉的理由（确定性策略没有 reward hacking）在**多接口/RL encoder** 的场景下是否仍成立？速度任务的策略也是确定性的，但它的探索来自哪里（$\epsilon$ 只在采集时加）？这一点论文没讨论。

### 9.3 如果我要跟进，会做什么

- **把「模型误差 $\to$ 策略回报」的敏感度量化**：按状态维度/接触相位分解误差，看哪一维真正影响梯度。
- **不确定性感知的 world model**：ensemble 或 heteroscedastic head，在想象时对高不确定区域降权（RWM-U 的思路），可能让 $H$ 用得更长。
- **把 $\epsilon$ 从「数据增强」升级为「探索策略」**：目前它只服务模型拟合，若让它随策略熵自适应，可能改善多接口 RL 的探索。
- **地形/物体**：世界模型输入加 heightmap patch 或物体状态，$\xi$ 加地形参数 —— 这是论文自己列的方向，也是最难的一步（接触模式会变得多模态，确定性模型可能不够）。

---

## 十、延伸阅读（按重要性）

1. **BeyondMimic**（Liao et al., 2025）—— 奖励、终止、动作缩放、低 PD 增益的直接来源，读这篇才能看懂附录 A。
2. **SONIC**（Luo et al., 2025）—— baseline，128 卡 PPO，理解「贵」在哪。
3. **SHAC**（Xu et al., 2022）—— 短视界 actor-critic + 可微仿真，本文的直系祖先；对比硬终止 vs 软存活。
4. **Dyna**（Sutton, 1991）/ **Dreamer**（Hafner et al.）—— 「用模型数据改进策略」的两种范式：前者在状态空间、后者在 latent 空间。
5. **RWM / RWM-U**（Li et al., 2025）—— 同样用学到的神经仿真器 + 真机迁移，但用 PPO 而非 BPTT；对照读能看清「解析梯度」的独特价值。
6. **DeepMimic**（Peng et al., 2018）—— 物理动作模仿的原点。
7. **ControlVAE / SuperTrack / SuperDyno** —— 学到的可微动力学用于角色控制的前史。

---

### 附：一页纸速查

$$
\begin{aligned}
& \min_{\theta}\ -\hat{J}(\theta)\\[2pt]
& \hat{J}(\theta)=\mathbb{E}\Big[\sum_{t<H}\gamma^{t}\hat{c}_t\hat{r}_{t+1}+\gamma^{H}\hat{c}_H V_\psi(\hat{s}_H,g_H,\xi)\Big]\\[2pt]
& \hat{s}_{t+1}=f_\omega(\hat{s}_t,a_t,\xi),\qquad
a_t=\pi_\theta(\hat{\bar{o}}_t+\eta_t,\,g_t),\qquad
\hat{\bar{o}}_t=g_\omega(\hat{s}_t,\xi)\\[2pt]
& \hat{c}_0=1,\qquad \hat{c}_{t+1}=\hat{c}_t\,(1-\hat{d}_{t+1}),\qquad
\hat{d}_t=d_\omega(\hat{s}_t,g_t)\\[2pt]
& \mathcal{L}_{\text{WM}}=\mathbb{E}_{\mathcal{B}}\sum_{j\le k}\Big[\lVert\hat{s}_j-s_j\rVert^{2}+(\hat{r}_j-r_j)^{2}+\lVert\hat{\bar{o}}_j-\bar{o}_j\rVert^{2}+\operatorname{BCE}(\hat{d}_j,d_j)\Big]\\[2pt]
& \mathcal{L}_V=\mathbb{E}\sum_{t<H}\big(V_t-\hat{R}_t\big)^{2},\qquad
\hat{R}_t=\hat{r}_{t+1}+\gamma(1-\hat{d}_{t+1})\big[(1-\lambda)V_{t+1}+\lambda\hat{R}_{t+1}\big]
\end{aligned}
$$

**超参**：$k=8$，$H=32$，$\gamma=0.99$，$\lambda=0.95$，$\alpha_\omega=\alpha_\theta=2\times 10^{-4}$，$4096$ envs/GPU，buffer $=1000$/env，动作噪声 $\mathcal{N}(0,0.1^{2})$，UTD $=0.4$，320k iters，cosine $\to 0.03\times$ @ 250k。

**规模**：$129{,}514$ 条动作 / $264.56$ h；$1\times$H100 $\times$ 3 h $\to$ 25 失败；$2\times$H100 $\times$ 9.8 h $\to$ 0 失败。
