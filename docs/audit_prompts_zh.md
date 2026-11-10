# 让 Agent 审这份代码的提示词

> 针对本仓库（SuperMimic-mjlab）的正确性审计。核心不是"帮我 review"，而是
> **先声明不变量 → 逐消费者对账 → 构造反例 → 固化成回归测试**。
> 一次只给一个范围；"能跑通、loss 有限"不算证据。

## 0. 三条使用规则

1. **先跑 P1 拿到地图**，再按风险挑 P2–P6，最后用 P7 把发现固化成测试，P8 做对抗复核。
2. 每个提示词都自带"只审计不改码"的约束。要修就另开一轮，并且要求贴修改前后数字。
3. 不要问"这里是不是写错了"——那是求附和。要问"哪条不变量不成立，反例是什么"。

## 通用前置块（粘在每个提示词前面）

```text
只审计，不修改代码。范围严格限定在我给的文件/函数，不要顺手评论风格。
每条结论必须带：违反点(文件:行) + 数值证据 + 影响面 + 最小修法。
没有证据的写「未验证」，不要写「看起来没问题」「整体实现正确」。
区分「已用数值/运行确认」和「仅静态推断(给置信度)」。
```

---

## P1 总纲：先画数据流，再列不变量

```text
你是训练算法审阅者。仓库 /home/zy0028/mjlab_project/supermimic-mjlab，范围 <src/supermimic/algo + src/supermimic/mdp>。

不要先读细节评价代码。按顺序做三件事：
1) 画出数据流：mjlab env → state/goal 提取 → replay buffer → world model 拟合 →
   想象 rollout(actor) → critic 拟合 → 日志/评估/ONNX 导出。标出每一步的张量形状与单位。
2) 对每个组件写出数学上必须成立的不变量（写成公式，明确 dt、gamma、lam、尺度、参考系）。
3) 逐条核对代码，输出表格：不变量 | 违反点(文件:行) | 数值反例 | 影响面 | 最小修法 | 回归断言。

特别注意：任何加工过的张量（*_normed / *_scaled / *_clip / *_rel / *_b）都要追到所有消费者，
跨文件也要追。最后按 🔴改变训练语义 / 🟡影响量级 / 🟢次要 分级，🔴必须有可运行的复现。
```

## P2 单位、尺度与归一化对账

```text
对 src/supermimic/algo/{state,networks,trainer,reward}.py 做「加工张量 × 消费者」对账。

1) 列出所有 RunningNorm 实例（state_norm / goal_norm / obs_norm / critic_norm）以及 reward 里的
   std、dt、gamma；说明每个实例拟合的是什么分布（真实转移？想象转移？原始观测？）。
2) 对每个被 normalize() 过的张量回答：原始还是无量纲？生产点(文件:行)？所有消费者(含跨文件)？
   消费者期望哪种尺度？是否存在「把归一化后的量又喂回 update()」或「用 A 的统计量归一化 B 的数据」？
3) 检查：norm_interval 期间用陈旧统计的后果、count 增长是否让统计冻结、eps/clip 是否在
   统计建立前就截断、BPTT 反向穿过归一化层是否被当成常数（detach）。

输出表格：张量 | 生产点 | 单位/尺度 | 消费者 | 期望 | 匹配?。不匹配的给 numpy 复现数字。
```

## P3 想象循环与 BPTT 语义

```text
审计 trainer.py 的 _sample_imagination_starts / _history_from_buffer / _update_policy /
_update_critic（约 443–685 行），以及 networks.py 的 WorldModel.forward、ActorCritic.act/value。

必须成立的不变量：
- 想象阶段 world model 与 critic 参数 frozen，且梯度不写进它们的 .grad；
- actor 梯度只沿 action → (fk, reward, termination) → 回报 这条链；
- critic 目标在 no_grad 下算，bootstrap V(s_H) 的输入已 detach；
- survival 权重与 termination 概率定义一致（1-d 还是 (1-d)^k，别混用）；
- 收集端与想象端的 dt、gamma、lam、reward 尺度完全相同；
- policy_chunk_size 改变时目标值不变（只允许浮点误差）。

验证方式：给每步画计算图并标 requires_grad；用 H=1、单样本手算对照代码输出；
把 chunk_size 从 4096 改成 512 跑同一批起点，比较 reward_sum 与梯度范数。
输出：梯度路径图 + 多余/缺失 detach 的位置 + 数值差异。
```

## P4 Buffer、采样与 episode 边界

```text
审计 algo/buffer.py、algo/sampler.py、trainer.py 的 _collect / _finish_episodes。

不变量：
- 环形缓冲的读写索引不跨 episode 边界（除非显式允许 burn-in 跨 reset）；
- 长度 wm_window 的窗口索引偏移正确（s_t … s_{t+k} 是 k 步还是 k+1 步）；
- done 处理、reset、buffer 写入三者的顺序与「最后一个 reward 是否计入」一致；
- return 分箱按 dt 换算成 1 秒边界，末箱不丢、边界样本不重复；
- warmup 完成前不采样；count 增长不导致统计/采样冻结。

构造最小反例：3 个 episode，长度 [2,5,3]，逐窗口打印 (env_id, t0, t1, episode_id)，
指出越界、错配或重复计数的具体那一条。
```

## P5 参考系、四元数与部署契约

```text
审计 state.py、mdp/commands.py、mdp/features.py、algo/deploy_features.py、algo/export.py、
scripts/export_onnx.py 的坐标/约定一致性。

逐项核对：四元数存储顺序(wxyz vs xyzw)、quat_mul 的乘法顺序、yaw 消除是否用对、
anchor-relative 位置公式 R(q_ref(t))ᵀ(p_ref(t+1+k)-p_ref(t))、goal 帧索引 t+1+k、
速度表达在哪个坐标系、关节顺序、action_scale/limit 的作用顺序、ONNX 中 baked 的
normalization 与 tanh 是否与 PyTorch 逐位一致（eps、clip、dtype、cast）。

然后用同一个 clip 手工算一帧 goal（写出中间数值），与 deploy_features 的输出对比；
再比较 export.py 的 onnxruntime 输出与 PyTorch actor 的差。任何 >1e-6 的偏差都要解释。
```

## P6 "对齐论文/上游"注释对账

```text
搜索所有声称对齐外部实现的注释与文档（"match", "following", "same as upstream",
"paper Eq.", "Table A.5", "BeyondMimic", docs/*.md 里的算法描述）。

对每条只给三态结论：成立 / 不成立 / 无法验证。上游不在仓库里 = 无法验证，不得默认成立。
重点核对：
- Algorithm A.1 的 collect/fit/imagine 顺序与比例（config.py 里 num_steps_per_env=2.5 与
  "UTD 0.4" 的说法是否自洽，UTD 的定义是「每环境步更新几次」还是「数据/更新」）；
- Table A.5 的超参与 rl_cfg.py 默认值逐项对齐；
- reward 的两个改动（去自碰撞、joint_limit 权重 -2）是否真的生效；
- termination 的条件与论文/文档描述是否一致。

给 git log -S / blame 证据说明这些行是移植还是原样搬运；注释与代码矛盾处直接引用原文。
```

## P7 反例 → 回归测试 → 变异验证

```text
把 P1–P6 里每条 🔴/🟡 发现固化成不依赖训练环境的 CPU 单测（pytest + torch/numpy）：

1) 构造能触发错误的最小输入（数据分布、batch、步数）；
2) 测试断言的是「数学不变量」，不是「函数能跑」；
3) 变异验证：把代码临时改回错误写法，确认测试失败并贴输出；改回，确认通过并贴输出；
4) 测试必须能在
   MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src /home/zy0028/miniconda3/envs/mjlab_130/bin/python -m pytest tests -q
   下跑通，且不依赖 GPU、不依赖日志文件。

输出：新增测试列表 + 每个测试失败/通过的两段输出 + 覆盖了哪条不变量。
```

## P8 对抗复核（第二意见）

```text
下面是另一轮审计给出的发现清单：<粘贴报告>。

你的任务不是确认它，而是尽力推翻它：
1) 对每条发现，构造一个能证明它「不是 bug」的场景（合法调用路径、配置、数据分布）；
2) 找出报告里把「静态推断」写成「已确认」的地方；
3) 指出遗漏的更高风险问题（跨文件、归一化、参考系、单位）；
4) 给每条发现一个最终裁决：真 bug / 仅有条件成立 / 误报，并附证据。
禁止因为「作者是这么写的」「注释这么说」而接受任何一条。
```

---

## 顺序与产出

建议一次会话只跑 P1 + 一个专题，避免审阅者摊薄注意力：

```text
P1（地图与不变量） → P2 或 P3（最可疑的语义层）→ P5（部署契约，如果改了 obs/goal）
→ P7（固化成测试）→ P8（对抗复核）
```

每轮结束后把「确认的 bug」写进 `docs/` 或 issue，附文件:行与复现命令；下一轮直接用 P8 复核，
不要重新问一遍「帮我看看有没有问题」。

## 反模式

- ❌「帮我 review 一下这个文件」：局部视角，跨文件的尺度错配看不见。
- ❌「检查有没有 bug」：目标太大，会去挑命名和格式。
- ❌「确认实现是否正确」：引导它证明"对"，而不是证伪。
- ❌「这里是不是标定错了？」：给出结论等附和。
- ❌ 相信注释里的"对齐上游"、相信 docs 里的算法描述——它们是要被审计的断言，不是证据。
- ❌ 用"能跑通 / loss 有限 / 指标不 NaN"当作正确性证据。

## 本机环境事实（写进提示词可减少幻觉）

- 当前 shell 无 GPU：`/dev/nvidia*` 为空，`torch.cuda.is_available()` 为 `False`；
  训练类结论只能静态推断或 CPU 复现，**不得声称跑过训练**。
- 可用的解释器：`/home/zy0028/miniconda3/envs/mjlab_130/bin/python`（有 torch + mjlab；
  仓库未 pip 安装，需 `PYTHONPATH=src`）。
- 建议前缀：`MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src`（否则 matplotlib 缓存告警）。
