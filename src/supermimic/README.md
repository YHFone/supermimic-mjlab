# SuperMimic package

An implementation of [SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model](https://supermimic.github.io) on top of mjlab's
MuJoCo Warp simulator.  The project-level README lives in the repository root;
this file documents the module layout.

The training loop follows Algorithm A.1 of the paper: collect simulator
transitions with action noise, fit a differentiable world model on short
autoregressive replay windows, then improve a goal-conditioned actor and a
critic by back-propagating an imagined return through the frozen world model.

## Quick start

```bash
# From the repository root.
pip install -e ".[dev]"

# 1. Preprocess BONES-SEED G1 clips (1200 clips = 2.24 h).
python -m supermimic.scripts.preprocess \
    --dataset-root /path/to/bones-seed \
    --output-dir /path/to/supermimic/g1_1200 \
    --max-clips 1200 --min-duration-s 2.0 --max-duration-s 30.0

# 2. Train with 4096 parallel environments.
python -m supermimic.scripts.train \
    --motion-dir /path/to/supermimic/g1_1200 \
    --num-envs 4096

# 3. Watch the policy (Viser on http://localhost:8091). The viewer overlays a
#    translucent reference robot; toggle it under "SuperMimic reference".
python -m supermimic.scripts.play \
    --checkpoint logs/supermimic/g1_supermimic/<timestamp>/checkpoint_final.pt \
    --motion-dir /path/to/supermimic/g1_1200 \
    --viewer viser --viser-port 8091

# 4. Evaluate deterministically, one rollout per motion.
python -m supermimic.scripts.evaluate \
    --checkpoint logs/supermimic/g1_supermimic/<timestamp>/checkpoint_final.pt \
    --motion-dir /path/to/supermimic/g1_1200 \
    --num-envs 64 --episodes-per-env 4
```

The G1 defaults are in `config/g1/rl_cfg.py`
(`unitree_g1_supermimic_cfg()`), so the command above uses them automatically.
Every algorithm hyperparameter is a `--sm.<field>` override; see
`algo/config.py` for the full list.

## What is implemented

- `algo/kinematics.py` — batched differentiable forward kinematics for
  hinge/slide trees. Link poses, linear velocities, and angular velocities
  match MuJoCo to `1e-6`.
- `algo/networks.py` — residual MLP world model, goal encoder, shared decoder,
  and critic.
- `algo/trainer.py` — data collection, world-model fitting, BPTT actor update
  with survival weighting, GAE critic targets, cosine LR schedule, logging,
  and checkpointing.
- `algo/buffer.py` — per-environment ring buffer with episode-boundary-aware
  window sampling.
- `algo/sampler.py` — return-based reset sampling over one-second bins
  (Appendix D.1).
- `algo/reward.py` — differentiable BeyondMimic tracking reward (self-collision
  removed, joint-limit weight `-2`) and anchor termination.
- `mdp/commands.py` — multi-motion command with per-environment clip/frame
  state and reference resets.
- `scripts/preprocess.py` — BONES-SEED G1 CSV to mjlab NPZ with the Appendix E
  keyword filter.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/play.py`.

## Validation performed

- Differentiable FK matches MuJoCo link poses, linear velocities, and angular
  velocities to numerical precision (`tests/test_supermimic_kinematics.py`).
- The analytic reward matches mjlab's tracking reward terms to `1e-7` on
  simulator states.
- Unit tests for the replay buffer window invariants and the networks.
- End-to-end smoke test: two training iterations, finite losses, checkpoint
  written (`tests/test_supermimic_smoke.py`, marked slow).

The defaults now match Table A.5 exactly: 4,096 environments, UTD 0.4
(``num_steps_per_env=2.5``), a 1,000-step-per-environment replay buffer,
10 x 2,048 world-model windows, 4,096 imagined rollouts, 320,000 iterations,
and cosine decay over 250,000 iterations. With the FK optimizations (vectorized
level-wise kinematics, one FK shared by reward and termination, TF32, compiled
FK) the steady-state iteration on an RTX 4090 is about 0.96 s, versus 20 s
before. The paper's H100 run is 0.11 s per iteration; the remaining ~8x is the
GPU difference. A full run is still ~85 hours on one 4090.

## Differences from the paper

- Analytic observation, reward, and termination heads are used instead of
  learned heads. The paper allows analytic heads and this removes reward-model
  bias.
- The paper batches are preserved. ``policy_chunk_size`` defaults to 4,096
  (no chunking); lower it only if memory is tight.
- The velocity-command interface, multi-GPU training, and real-robot deployment
  are not implemented.
- The full 129,514-clip library is not bundled. `scripts/preprocess.py` can
  build it, but loading 47 GB of reference frames requires a larger host or a
  streaming motion library.
