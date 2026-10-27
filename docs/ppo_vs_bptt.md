# PPO vs. BPTT：两种策略梯度估计的完整对比

> **配套**：与 [supermimic_paper_reading_zh.md](supermimic_paper_reading_zh.md) 配套阅读——那篇是论文精读，这篇只谈「梯度是怎么来的」。
>
> **范围**：本文对比的是「**常规 model-free PPO**」与「**在学到的可微世界模型上做 BPTT**」这两种策略改进方式，以 SuperMimic（arXiv:2610.28480）和 SHAC / Dreamer / PPO-WM 为参照。
>
> **数学约定**：行内 `$...$`、行间 `$$...$$`，符号与 SuperMimic 论文一致（$\hat{s}_t$ 表示世界模型的量）。带 💡 的是我的分析或工程经验，不是论文原文。

---

## 0. 一句话

| | PPO | BPTT |
|---|---|---|
| 本质 | 用**采样到的动作**的似然比给回报加权 | 用**回报对动作的解析导数**穿过动力学反传 |
| 一句话 | 「这个动作比平均好多少？那就让它更常出现」 | 「如果当时这个动作大一点，之后每一步的回报会怎么变？」 |
| 梯度来源 | score function（REINFORCE 家族） | pathwise derivative（重参数化家族） |
| 需要 | 环境可采样 | 动力学可微（或可学一个 surrogate） |

两者估计的是**同一个东西**——期望回报对参数的梯度——但走的是数学上两条根本不同的路。

---

## 1. 两种梯度到底在算什么

### 1.1 目标

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\Big[\sum_{t\ge 0}\gamma^{t}r_{t+1}\Big],
\qquad
p_\theta(\tau)=\rho_0(s_0)\prod_{t}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)
$$

难点在于：**期望的分布本身依赖 $\theta$**。$p_\theta(\tau)$ 里既有 $\pi_\theta$（可微），也有 $p(s'\mid s,a)$（PPO 假设不可微）。

### 1.2 PPO：对「概率」求导（score function）

关键恒等式是 $\nabla_\theta p_\theta=\;p_\theta\,\nabla_\theta\log p_\theta$，于是

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\Big[\Big(\sum_t\gamma^{t}r_{t+1}\Big)\nabla_\theta\log p_\theta(\tau)\Big]
$$

而 $\log p_\theta(\tau)$ 里只有策略项依赖 $\theta$（动力学项 $\log p(s'\mid s,a)$ 被消掉了），所以

$$
\nabla_\theta J=\mathbb{E}\Big[\sum_t \gamma^{t}\,A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big]
$$

PPO 在此基础上做了三件事（都是工程上的方差/稳定性处理）：

$$
\mathcal{L}_{\text{PPO}}(\theta)=\mathbb{E}\Big[\min\big(\rho_tA_t,\ \operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big],
\qquad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

1. 用 **critic + GAE** 把 $A_t$ 的方差压下来；
2. 用 **clip** 把策略更新限制在信赖域内；
3. 用**多个 epoch 的小 batch** 复用同一批数据（但仍然近似 on-policy）。

**结构性的东西**：梯度只通过 $\log\pi_\theta$ 进入，**动力学被完全当作黑盒**。这就是「model-free」的含义——它连一步 $s_{t+1}$ 都不需要求导。

### 1.3 BPTT：对「路径」求导（pathwise derivative）

把动力学换成一个可微模型 $\hat{s}_{t+1}=f_\omega(\hat{s}_t,a_t,\xi)$，动作由一个确定性策略产生 $a_t=\pi_\theta(o_t,g_t)$，整条想象轨迹就是 $\theta$ 的一个**确定性函数**。于是可以直接用链式法则：

$$
\hat{J}(\theta)=\mathbb{E}_{s_0}\Big[\sum_{t=0}^{H-1}\gamma^{t}\hat{r}_{t+1}\Big],
\qquad
\nabla_\theta\hat{J}=\mathbb{E}_{s_0}\Big[\sum_{t=0}^{H-1}\gamma^{t}\frac{\partial \hat{r}_{t+1}}{\partial \theta}\Big]
$$

把 $\partial\hat{r}/\partial\theta$ 展开，就得到两项：**本步动作的直接影响** + **通过未来状态传播的间接影响**：

$$
\nabla_\theta\hat{J}
=\mathbb{E}\sum_{t=0}^{H-1}\gamma^{t}\Big[
\underbrace{\frac{\partial \hat{r}_{t+1}}{\partial a_t}}_{\text{direct}}
+\underbrace{\sum_{t'=t+1}^{H-1}\gamma^{t'-t}\frac{\partial \hat{r}_{t'+1}}{\partial \hat{s}_{t'}}\frac{\partial \hat{s}_{t'}}{\partial a_t}}_{\text{via }H\text{ future steps}}
\Big]\frac{\partial a_t}{\partial\theta}
$$

其中后一项的核是**雅可比沿轨迹的连乘**：

$$
\frac{\partial \hat{s}_{t'}}{\partial a_t}
=\Big(\prod_{k=t+1}^{t'-1}A_k\Big)B_t,
\qquad
A_k=\frac{\partial f_\omega}{\partial \hat{s}_k},\quad
B_t=\frac{\partial f_\omega}{\partial a_t}
$$

**💡 这就是两种方法最本质的差别**：PPO 只能给「**已经发生过**的动作」重新加权；BPTT 能回答「**假如**当时动作改成另一个值，后面 $H$ 步会怎样」——后者是**反事实**信息，前者只是**重加权**。

### 1.4 一个积分的两种求法（直觉）

想估计 $\nabla_\theta\,\mathbb{E}_{a\sim\pi_\theta}[R(a)]$：

- **score function**：把梯度搬到概率上，$\mathbb{E}[R\,\nabla\log\pi]$。需要对 $R$ **采样**，方差大，但 $R$ 可以是任意黑盒（不可微、离散、稀疏）。
- **pathwise derivative**：写 $a=\mu_\theta+\sigma\epsilon$，$\epsilon\sim\mathcal{N}(0,I)$，把随机性挪到与 $\theta$ 无关的 $\epsilon$ 上，于是 $\nabla_\theta\mathbb{E}[R(a)]=\mathbb{E}[\nabla_aR\cdot\nabla_\theta\mu_\theta]$。需要 $R$ **对 $a$ 可微**，但方差小得多。

BPTT 就是 pathwise derivative 在**时间维度上的递归版**：不仅 $R$ 对 $a$ 可微，连"下一个状态"也对 $a$ 可微。

### 1.5 显式的 credit assignment 对比

| | PPO | BPTT |
|---|---|---|
| 第 $t$ 步的动作 | 只有采样到的那个 $a_t$ | $a_t$ 是一个连续函数 $\pi_\theta(o_t)$ |
| 第 $t'$ 步奖励对 $a_t$ 的依赖 | 通过学习出来的 $A_t$ 隐式体现 | 显式的雅可比连乘 $\partial\hat{s}_{t'}/\partial a_t$ |
| 长时信用分配 | 靠 critic 的 bootstrap（可能不准） | 靠链式法则（准确，但会被模型误差污染） |
| 视界外 | 折扣 + GAE 自然衰减 | 硬截断，必须 critic 自举 |

---

## 2. 偏差与方差

这是选择两者的**唯一实质理由**，值得单独拆开。

### 2.1 PPO 的方差从哪来

$$
\operatorname{Var}\big[\hat{g}_{\text{PPO}}\big]\approx\frac{1}{N}\operatorname{Var}\big[A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big]
$$

三个放大因子：

1. **动作采样**：单条轨迹只看到一个动作，$\nabla\log\pi$ 的符号和幅度随采样剧烈跳动；
2. **$A_t$ 本身是估计量**：$A_t=\sum_{k}\gamma^{k}r-V(s_t)$，回报的和越长，方差越大（$\propto H$）；
3. **策略越接近确定性，$\nabla\log\pi$ 越尖**（$\pi\to\delta$ 时似然比退化）——这也是为什么 PPO 必须保持一个"足够随机"的策略，而随机性又直接推高方差。

**数量级直觉**：要让梯度估计的标准差降到 $1/10$，样本要 ×100。这就是 PPO 在长视界、高维动作上"怎么加 batch 都不够"的原因。

### 2.2 PPO 的偏差从哪来（它并不是无偏的）

很多人以为 PPO 无偏。实际上：

- **clip** 引入偏差（超出信赖域的部分梯度被截断）；
- **GAE($\lambda<1$)** 用 $V$ 自举，偏差 $\propto(1-\lambda)$；
- **函数逼近**（critic 不完美）把误差注入 $A_t$；
- **多 epoch 复用**使数据不再 on-policy，重要性采样比偏离 1。

所以准确说法是：**PPO 的偏差不来自动力学建模**（它不建模），而来自值函数、裁剪和 off-policy 复用。这个区别很重要——它意味着 PPO 的偏差**不会随动作库变大而变差**。

### 2.3 BPTT 的偏差从哪来

$$
\mathbb{E}\big[\hat{g}_{\text{BPTT}}\big]-\nabla_\theta J
=\underbrace{O\big(\lVert f_\omega-f\rVert\big)}_{\text{model error}}
+\underbrace{O\big(\gamma^{H}\big)}_{\text{truncation}}
+\underbrace{O\big(\lVert \hat{r}-r\rVert\big)}_{\text{reward-head error}}
$$

一阶 Taylor 展开更能看清机制：模型误差 $\delta f=f_\omega-f$ 会通过**二阶项**污染梯度，

$$
\nabla_\theta\hat{J}-\nabla_\theta J
\approx\mathbb{E}\Big[\sum_t\frac{\partial^{2}J}{\partial f_t\,\partial\theta}\,\delta f_t\Big]
$$

**两个致命推论**：

1. 模型在**平均 MSE** 上很准，不代表梯度准——误差要落在「雅可比大的方向」上才致命（比如脚接触瞬间）；
2. 误差会随 $H$ **累积**（最坏情况指数放大），所以 BPTT 天然偏好短视界 + critic 自举。

SuperMimic 的应对：$k=8$ 步自回归训练（让模型学会纠正自身漂移）、$H=32$ 短视界、critic bootstrap、梯度裁剪。论文的实测是 8 步预测误差在数据量 $264\times$ 增长下稳定在 $0.007$–$0.011$。

### 2.4 偏差–方差定位

```
方差
 ▲
 │  REINFORCE（无偏，方差极大）
 │        ●
 │              ● PPO + GAE（有偏，方差中）
 │                    ● PPO-WM（同一个 PPO，但在模型上跑）
 │                          ● BPTT H=8（有偏小，方差小）
 │                                ● BPTT H=32（有偏中，方差很小）
 │                                      ● BPTT H=128（偏差↑，方差仍小）
 └──────────────────────────────────────────────▶ 偏差（模型误差）
```

**💡 结论**：BPTT 不是"更好的 PPO"，而是**在偏差–方差曲线上换了一个位置**。它把方差换成了对模型精度的依赖。模型准的时候它碾压；模型不准的时候它会**非常高效地优化错误的目标**。

---

## 3. 计算与工程成本

### 3.1 显存

| | 存什么 | 量级 |
|---|---|---|
| PPO | 观测、动作、$\log\pi$、$V$、奖励、终止（$B\times T$ 个数） | 与 $H$ 无关，几 GB 以内 |
| BPTT | 每一想象步的**全部中间激活**（反向要用） | $O(H\cdot B\cdot n_{\text{hidden}})$ |

具体估算（SuperMimic 的配置：$H=32$，batch $4096$，MLP $3\times 2048$）：

$$
32\times 4096\times 2048\times 4\ \text{B}\approx 1.07\ \text{GB}\ \text{per hidden layer}
$$

三层 + 输入输出 $\approx 3$–$4$ GB；$H=128$ 时 $\approx 13$–$16$ GB。这就是论文说"$H=128$ 是 H100 上能放下的最长视界"的原因——**不是算不动，是存不下**。

### 3.2 计算

| | 每次更新 |
|---|---|
| PPO | 采样 $B\times T$ 步（贵，因为跑仿真）+ 前向若干 epoch |
| BPTT | $H$ 步 MLP 前向 + **1 次反向**（贵，但 MLP 比仿真快得多） |

关键数字（SuperMimic 图 5a，H100）：世界模型在 16k 并行环境下 $6.71\times10^{6}$ steps/s，MuJoCo Warp $2.61\times10^{5}$ steps/s —— **26×**。所以"在模型上跑 PPO"（PPO-WM）也不慢，但仍然比 BPTT 差 64× 的样本效率（见 §5）。

### 3.3 并行扩展性

- **PPO**：吞吐受**仿真器**限制。MuJoCo Warp 在 64k 环境时已经平坦（$3.22\times10^{5}$ steps/s），加卡也只能线性扩。
- **BPTT**：吞吐受 **GPU FLOPs** 限制，且可以用大 batch 吃满 tensor core；世界模型在 64k 时还能到 $2.00\times10^{7}$ steps/s（62×）。

**💡 这条差异常被忽略**：PPO 的瓶颈是"产生数据"，BPTT 的瓶颈是"消费数据"。前者靠堆机器，后者靠堆算力密度。

---

## 4. 二十维大表

| # | 维度 | PPO（model-free） | BPTT（through learned model） |
|---|---|---|---|
| 1 | 需要的环境 | 任意黑盒，能采样即可 | 可微动力学，或能学一个 surrogate |
| 2 | 数据来源 | 当前策略在真实/仿真环境采样（on-policy） | 学到的模型想象（起点 off-policy） |
| 3 | 梯度信号 | 采样动作的似然比 $\times$ 优势 | 回报对动作的解析导数 |
| 4 | 策略类型 | 必须随机 | 确定性（外加噪声） |
| 5 | 动作空间 | 离散 / 连续都行 | 连续（离散需 Gumbel / 重参数化） |
| 6 | 奖励要求 | 任意（稀疏、不可微、事件型都行） | 需可微或至少平滑 |
| 7 | 方差 | 高（动作采样 + $A$ 估计） | 低（给定 $s_0$ 时接近 0） |
| 8 | 偏差 | 裁剪、GAE、函数逼近 | 模型误差 + 截断（随 $H$ 累积） |
| 9 | 样本效率 | 低 | 高（SuperMimic 实测 64×） |
| 10 | 显存 | $O(B\cdot T)$，无计算图 | $O(H\cdot B\cdot n_{\text{hidden}})$，有图 |
| 11 | 更新算力 | 采样贵、反传便宜 | 采样便宜、反传贵 |
| 12 | 扩展瓶颈 | 仿真器吞吐 | GPU FLOPs / 显存 |
| 13 | 视界 | GAE + 折扣，天然长视界 | 硬截断 $H$，必须 critic 自举 |
| 14 | 稳定性机制 | clip 提供信赖域 | 无内建信赖域，靠梯度裁剪 + 小 lr |
| 15 | 超参敏感度 | 中（clip、lr、GAE $\lambda$） | 高（$H$、lr、模型精度、噪声尺度） |
| 16 | 探索 | 策略熵（内生） | 外部动作噪声 $\epsilon$（外置） |
| 17 | 接触/不连续 | 无影响（黑盒） | 靠学到的光滑模型绕开；可微仿真会炸 |
| 18 | 典型失败模式 | 样本低效、方差大、调参难 | 高效优化错误目标、梯度爆炸、长视界发散 |
| 19 | 代表工作 | PPO、SONIC、BeyondMimic、RWM | SHAC、SuperMimic、SuperTrack、ControlVAE |
| 20 | 调试难度 | 低（工具链成熟） | 高（要同时看模型、梯度、策略） |

---

## 5. 实测数字（SuperMimic 的两个消融）

### 5.1 同一世界模型上，BPTT vs PPO-WM

固定世界模型、固定数据、固定网络，只换策略改进方式（图 5b，LAFAN1 的 walk/run/sprint）：

| 方法 | 达到最好训练回报所需的 imagined steps |
|---|---|
| PPO-WM（4k×32、8k×24、8k×32、16k×32 四种配置） | $5.96\times10^{10}$ |
| **SuperMimic（BPTT + critic）** | $9.32\times10^{8}$ |

$$
\frac{5.96\times10^{10}}{9.32\times10^{8}}\approx 64
$$

**这是最干净的对比**：同一个模型、同一批数据，只有梯度估计方式不同，差 $64\times$。所以收益**不只是"rollout 快"**，而是一阶梯度本身带来了更高的样本效率。

### 5.2 端到端：BPTT vs PPO-Sim（真仿真器）

| | 设置 | 到达归一化回报 $0.82$ 的时间 | 18 h 后的最终回报 |
|---|---|---|---|
| PPO-Sim, 1 GPU | LAFAN1 40 clips | 未达到 | $0.60$ |
| PPO-Sim, 4 GPU | 同上 | 未达到 | $0.70$ |
| PPO-Sim, 8 GPU | 同上 | 未达到 | $0.72$ |
| **SuperMimic, 1 GPU** | 同上 | **1.2 h** | — |

**⚠️ 归因警告**：这个对比里混了三个因素——(a) 在模型上想象而不是在仿真里采样；(b) 解析梯度；(c) 更快的 rollout。5.1 那个消融才是把 (b) 单独拎出来的。

### 5.3 规模端到端

| | SONIC（PPO, 128×L40, 7 天） | SuperMimic（BPTT, 1×H100, 3 h） |
|---|---|---|
| GPU·h | $\approx 21{,}000$ | $3$ |
| 129,514 条动作的失败数 | 179 | 25 |
| 归一化回报 | 0.90 | 0.936 |

---

## 6. 什么时候用哪个

### 6.1 决策树

```
动力学对动作可微吗（或能学一个准的可微模型）？
├─ 不能 / 状态有随机跳变（碰撞、被推、物体掉落）
│     └─ PPO
└─ 能
      ├─ 奖励可微 / 平滑（dense shaping）
      │     ├─ 动作连续 → BPTT（短视界 + critic）
      │     └─ 动作离散 → PPO（或 Gumbel-softmax 版 BPTT，实验性）
      └─ 奖励稀疏 / 事件型（成功=1，失败=0）
            └─ PPO（或先做 reward shaping 再考虑 BPTT）

额外判据：
- 仿真很便宜、真机数据很贵 → 学模型 + BPTT（把真机数据用在模型上）
- 仿真已经很便宜、模型难学 → PPO
- 需要绝对稳、可复现的 baseline → PPO
- 动作库巨大、要压 GPU·h → BPTT
```

### 6.2 场景表

| 场景 | 推荐 | 理由 |
|---|---|---|
| 人形动作跟踪（dense 指数核奖励、连续动作） | **BPTT** | SuperMimic 的场景，正是 BPTT 的主场 |
| 灵巧手操作 + 稀疏成功信号 | PPO | 奖励不可微，BPTT 无从下手 |
| 离散动作（Atari、棋盘） | PPO | BPTT 需要重参数化，收益不确定 |
| 真机在线学习（不敢乱试） | 学模型 + BPTT / Dreamer | 模型里试错便宜 |
| 接触丰富、可微仿真会炸 | 学 surrogate + BPTT | 这正是 SuperMimic 的动机 |
| 需要可解释的稳定收敛曲线 | PPO | 成熟、方差大但可控 |
| 要测 scaling law | BPTT | 成本随数据量增长慢，能跑很多点 |

---

## 7. 混合与中间地带

| 方法 | 模型 | 梯度 | 特点 |
|---|---|---|---|
| **PPO-WM** | 学到的世界模型 | 似然比（在想象上） | 对模型误差更鲁棒，但样本效率低（本文 $-64\times$） |
| **BPTT + critic**（SuperMimic） | 学到的世界模型 | pathwise + bootstrap | 当前性价比最高 |
| **SHAC** | **可微仿真器**（不学模型） | pathwise + critic | 无模型偏差，但接触不可微就崩 |
| **Dreamer(V3)** | latent 动力学 | pathwise（在 latent 里） | 高维观测友好，但 latent 误差难诊断 |
| **TD-MPC** | 学到的模型 | 以**规划**（MPC）为主，策略先验是辅助 | 短视界很强，长视界靠 value |
| **SVG(0)** | 学到的模型 | 重参数化 + value gradient | PPO 与 BPTT 的中间态 |
| **PPO + 模型辅助值估计** | 学到的模型 | 似然比 + 模型 rollout 做 target | 只借模型的"预测"，不借"梯度" |

**💡 一个实用折中**：先用 PPO 冷启动（模型还没训好时 BPTT 的梯度不可信），等模型 8 步误差稳定后再切到 BPTT。SuperMimic 没这么做（它一开始就交替训练），因为它的模型从第一步就在学，且奖励 dense。

---

## 8. 实现细节

### 8.1 PPO 骨架

```python
for it in range(iters):
    # 1) 采样：on-policy，跑仿真
    batch = collect(env, policy, n_steps=T, n_envs=B)
    #    -> obs, act, logp_old, rew, done, val
    # 2) 算 GAE
    adv, ret = gae(batch.rew, batch.val, batch.done, gamma, lam)
    # 3) 多 epoch 更新
    for epoch in range(K):
        for mb in minibatches(batch):
            ratio = exp(policy.logp(mb.obs, mb.act) - mb.logp_old)
            loss = -min(ratio * mb.adv,
                        clip(ratio, 1-eps, 1+eps) * mb.adv).mean()
            loss += vf_coef * mse(value(mb.obs), mb.ret)
            loss -= ent_coef * policy.entropy(mb.obs)
            opt.step(loss)
```

要点：**不需要动力学可微、不需要存计算图、不需要 $H$ 这个超参**。

### 8.2 BPTT 骨架

```python
for it in range(iters):
    # 1) 拟合世界模型（与策略更新交替）
    for _ in range(wm_steps):
        s0, a_seq, xi, s_seq, r_seq, d_seq, o_seq, g_seq = sample_kstep(buffer, k=8)
        s = s0
        loss = 0
        for j in range(k):
            s_next = f_wm(s, a_seq[j], xi)      # 自回归：喂回自己的输出
            loss += mse(s_next, s_seq[j]) + mse(r_wm(s, a_seq[j], g_seq[j]), r_seq[j]) \
                  + mse(o_wm(s, xi), o_seq[j]) + bce(d_wm(s, g_seq[j]), d_seq[j])
            s = s_next
        opt_wm.step(loss / k)

    # 2) 在冻结的模型上想象 H 步，反传
    s = sample_start(buffer)                    # s0 ~ rho_0（return-based）
    o_hat, a_list, s_list, r_list, d_list = [], [], [], [], []
    for t in range(H):
        o_hat = o_wm(s, xi).detach() + noise()  # 噪声在模型外重加
        a = policy(o_hat, g_t)
        r = r_wm(s, a, g_t)
        s_next = f_wm(s, a, xi)                 # 不 detach：要梯度
        d = d_wm(s_next, g_{t+1})
        a_list.append(a); s_list.append(s); r_list.append(r); d_list.append(d)
        s = s_next

    J = sum(gamma**t * c_t * r_t for ...) + gamma**H * c_H * V(s_H, g_H, xi)
    # 关键：只更新策略/值函数，不更新模型
    loss_pi = -J
    opt_pi.step(loss_pi)
    # 3) critic 用同一条 rollout 的 GAE 目标回归（目标冻结）
```

要点：**`f_wm` 不 detach（要梯度），`o_wm` 输出 detach（噪声重加），模型参数必须 stop-gradient**。

### 8.3 两者共同的坑 / BPTT 特有的坑

| 坑 | PPO | BPTT |
|---|---|---|
| 忘记 stop-gradient 模型参数 | 不适用 | **致命**：模型会学着让策略好看，而不是预测准 |
| 长视界梯度爆炸 | 不适用 | 用短 $H$ + 梯度裁剪（全局 1.0 + 逐元素 0.1） |
| 动作尺度不一致 | 归一化 | 归一化 + 奖励量纲对齐 |
| 终止处理 | mask 掉 bootstrap | 硬终止断梯度；软终止（概率）保持可微 |
| 观察噪声 | 可选 | **必须**在模型外重加，否则策略会利用模型的确定性 |
| 混合精度 | 可用 | 慎用 fp16（BPTT 对精度敏感），TF32 可以 |
| batch 太大 | 收益递减 | 显存爆炸（$O(H\cdot B)$） |
| 学习率 | $3\times10^{-4}$ 量级 | 更小（$2\times10^{-4}$ 且 cosine 衰减） |

---

## 9. 五个常见误解

1. **「BPTT 一定更省样本」**
   只在**模型准**的时候成立。模型不准时，BPTT 会用极少的样本高效地优化一个错误目标——这比 PPO 慢更糟，因为你看不出它在错。

2. **「PPO 是无偏的」**
   错。clip、GAE 自举、函数逼近、多 epoch 复用都引入偏差。准确说法是：PPO 的偏差**不来自动力学建模**。

3. **「BPTT 不需要 critic」**
   错。截断在 $H$ 就丢掉了视界外的回报，必须 bootstrap。SuperMimic 的消融显示：无 critic 时 $H\le16$ 直接训不出来；有 critic 时 $H=16$ 能追平无 critic 的 $H=32$；抗推能力也完全不同（有 critic 在 $0.5$–$2.4$ s 稳住，无 critic 永不稳）。

4. **「BPTT 能全面替代 PPO」**
   不能。离散动作、稀疏奖励、不可微动力学、随机状态跳变（被推、物体掉落）这四类场景，BPTT 要么做不了，要么要先做大量改造。

5. **「BPTT 的梯度方差是 0」**
   给定起点 $s_0$ 且模型/策略都确定性时，条件方差确实是 0；但 $s_0$ 是采样的、SuperMimic 还每步注入 $\eta_t$，所以边缘方差非零。只是它**远小于** PPO 里"动作采样 + 优势估计"带来的方差。

---

## 10. 速查

$$
\begin{aligned}
&\text{PPO:}\quad \nabla_\theta J=\mathbb{E}\Big[\sum_t\gamma^{t}A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big]\\[2pt]
&\text{BPTT:}\quad \nabla_\theta\hat{J}=\mathbb{E}\sum_t\gamma^{t}\Big[\frac{\partial\hat{r}_{t+1}}{\partial a_t}+\sum_{t'>t}\gamma^{t'-t}\frac{\partial\hat{r}_{t'+1}}{\partial\hat{s}_{t'}}\frac{\partial\hat{s}_{t'}}{\partial a_t}\Big]\frac{\partial a_t}{\partial\theta}\\[2pt]
&\text{Jacobian chain:}\quad \frac{\partial\hat{s}_{t'}}{\partial a_t}=\Big(\prod_{k=t+1}^{t'-1}\frac{\partial f_\omega}{\partial\hat{s}_k}\Big)\frac{\partial f_\omega}{\partial a_t}\\[2pt]
&\text{cost:}\quad \text{PPO}=O(B\cdot T)\ \text{(storage)};\quad \text{BPTT}=O(H\cdot B\cdot n_h)\ \text{(activations)}
\end{aligned}
$$

| | PPO | BPTT |
|---|---|---|
| 一句话 | 重加权已发生的动作 | 反事实地求导 |
| 赢在 | 通用、稳、无模型假设 | 样本效率、rollout 速度、梯度质量 |
| 输在 | 样本效率、方差 | 模型误差、显存、离散/稀疏奖励 |
| 关键超参 | clip、$\lambda$、lr | $H$、lr、模型精度 |
| 典型数字 | SONIC: 21k GPU·h | SuperMimic: 3 GPU·h、64× 更少 imagined steps |
