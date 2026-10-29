.. _supermimic:

SuperMimic: model-based humanoid motion tracking
================================================

This page documents an implementation of `SuperMimic: Scaling Humanoid
Whole-Body Control with One Differentiable World Model
<https://supermimic.github.io>`_ (CoRL 2026) inside mjlab. The implementation
trains a goal-conditioned motion-tracking policy for the Unitree G1 by
back-propagating an imagined return through a learned, differentiable world
model instead of running model-free PPO in the simulator.

What is implemented
-------------------

The training loop mirrors Algorithm A.1 of the paper:

1. **Collect.** The current policy acts in MuJoCo Warp with Gaussian action
   noise. Transitions are stored in a per-environment replay ring buffer.
   Resets are sampled from a return-based mixture over one-second motion bins
   (Appendix D.1).
2. **Fit the world model.** A residual MLP dynamics model is trained on
   ``k = 8`` step autoregressive windows of logged states and actions. The
   world model conditions on a per-episode simulator context
   ``xi = [friction, encoder bias, torso COM offset]``.
3. **Improve the policy.** The actor and critic are optimized on ``H = 32``
   imagined steps. The world model is frozen during this backward pass, and
   back-propagation through time supplies analytic policy gradients. The
   critic bootstraps the return beyond the horizon with GAE targets
   (Eq. 3).

The observation, reward, and termination are computed analytically from the
predicted state, so the world model only has to learn the dynamics. The
tracking reward matches the BeyondMimic instantiation in mjlab, including the
two changes of Appendix A: no self-collision penalty and a joint-limit weight
of ``-2``.

Implemented:

- multi-motion ``MotionLibraryCommand`` with per-environment clip/frame state,
  reference-state resets, and return-based start-bin sampling;
- differentiable G1 forward kinematics (hinge/slide trees) validated against
  MuJoCo for link poses, linear velocities, and angular velocities;
- world model, deterministic goal-conditioned actor, and critic;
- BPTT actor--critic update, GAE, survival weighting, gradient clipping;
- BONES-SEED G1 CSV preprocessing with the Appendix E name filter;
- deterministic evaluation and a Viser/video play script;
- unit and smoke tests.

Not implemented: the velocity-command interface, learned reward/termination
heads, multi-GPU training, real-robot deployment, and the full 129,514-clip
library as one managed dataset.

Installation
------------

From the repository root:

.. code-block:: bash

   pip install -e . --no-deps

If ``pip`` reports a missing ``uv_build`` backend, run ``pip install uv_build``
first. The scripts are run as modules:

.. code-block:: bash

   python -m supermimic.scripts.<script> ...

Motion data
-----------

The paper trains on the G1-retargeted `BONES-SEED
<https://huggingface.co/datasets/bones-studio/seed>`_ release. Download and
extract ``g1.tar.gz`` into a dataset root, then convert a subset:

.. code-block:: bash

   python -m supermimic.scripts.preprocess \
       --dataset-root /data/bones-seed \
       --output-dir /data/supermimic/g1_1200 \
       --max-clips 1200 \
       --min-duration-s 2.0 --max-duration-s 30.0

A 1,200-clip / 2.24 h subset takes about 20 seconds and 0.7 GB on disk. Pass
``--skip-existing`` to resume.

Training
--------

.. code-block:: bash

   python -m supermimic.scripts.train \
       --motion-dir /data/supermimic/g1_1200 \
       --num-envs 4096

The G1 defaults live in
``src/supermimic/config/g1/rl_cfg.py``
(``unitree_g1_supermimic_cfg()``) and are **exactly** Table A.5 of the paper:
4,096 environments, update-to-data ratio 0.4 (``num_steps_per_env=2.5``
vectorized simulator steps per collect/fit/imagine round), a replay buffer of
1,000 steps per environment, ``k=8``, 10 world-model batches of 2,048 windows,
``H=32``, 4,096 imagined rollouts, 320,000 iterations, cosine decay to 0.03x
over 250,000 iterations, and Adam at 2e-4 everywhere.

``num_steps_per_env=2.5`` is the number of vectorized simulator steps
collected per update round (the reciprocal of the 0.4 update-to-data ratio),
**not** the replay-buffer length.  The buffer is 1,000 steps per environment;
the trainer accumulates the 0.5 fractional step, so it collects 2 and 3 steps
alternately and averages 2.5. The observation
is the 93-dimensional Table A.1 vector with 10 frames of history; the goal is
the K=10 future reference sequence, 73 dimensions per frame (joint positions
and velocities, anchor position and orientation, plus the reference root's
linear and angular velocity in the reference frame); the root
reward/termination is measured on the pelvis. Every field of
``SuperMimicCfg`` can still be overridden with ``--sm.<field>``.

``policy_chunk_size`` (default 4,096) is not a paper parameter: it splits the
4,096 imagined rollouts into gradient-accumulation chunks so that the
*effective* batch and gradient remain the paper's 4,096. After the FK
optimizations below, the full 4,096 batch fits on a 24 GB GPU, so no chunking
is needed by default; lower it only if the policy update runs out of memory.

Iteration speed
---------------

The first version of this implementation took about 20 s per iteration on an
RTX 4090. Profiling one 512-rollout imagination chunk gave:

.. list-table::
   :header-rows: 1

   * - Component
     - Time per 32-step chunk
   * - Differentiable FK inside the reward
     - 723 ms
   * - Differentiable FK inside the termination
     - 390 ms
   * - Goal features
     - 172 ms
   * - World model
     - 75 ms
   * - Policy network
     - 23 ms
   * - Proprioception / reference gather
     - 14 ms

The FK was recomputed twice per imagined step (once by the reward, once by
the termination), for all 30 robot bodies, with a Python loop per body and an
autograd graph through every quaternion op. Eight chunks of 512 rolled out
serially, with no TF32 and no compiler.

The fixes are:

- **vectorized level-wise FK**: one batched pass per kinematic-tree depth
  instead of a Python loop per body;
- **one FK per step**: the reward and termination share the same body poses
  and velocities;
- **binary, detached termination**: the paper's ``d_t`` is a discrete event,
  so the survival product ``c_t`` should not back-propagate through it;
- **full 4,096 rollout batch** after the FK memory reduction;
- **TF32 matmuls**, matching the precision the paper uses for world-model
  rollouts;
- **``torch.compile`` on the FK**, which is the single hottest kernel.

Measured on the same RTX 4090, with the exact paper hyperparameters and
4,096 environments, the steady-state iteration is now about **0.96 s**:
collection 0.36 s, world model 0.22 s, policy + critic 0.38 s. The first timed
iteration is slightly slower while CUDA warms up; the one-off FK compilation
is done before the timer and takes 2-50 s depending on the Inductor cache.

The paper reports 0.11 s per iteration on an H100 node. The remaining ~8x is
hardware, not a missing paper parameter: the H100 has roughly 3x the TF32
throughput and 3x the memory bandwidth of a 4090, and the paper's reference
run uses two GPUs with data-parallel gradients. At 0.96 s per iteration the
full 320,000-iteration schedule is about 85 hours on one 4090, versus the
paper's 9.8 hours on two H100s.

.. note::

   ``torch.compile`` caches per batch shape. If you change
   ``--sm.policy-batch-size``, the first iteration compiles again.

The world model and actor are saved as ``checkpoint_<iteration>.pt`` and
``checkpoint_final.pt``. Ctrl+C saves ``checkpoint_interrupt.pt``. Logs are
printed as an rsl-rl-style table and written to TensorBoard.

Each run writes to its own timestamped directory,
``<log-dir>/<experiment-name>/<YYYY-MM-DD_HH-MM-SS>/`` (for example
``logs/supermimic/g1_supermimic/2026-10-29_15-04-05/``), so repeated runs never
share TensorBoard event files. Point TensorBoard at the parent:

.. code-block:: bash

   tensorboard --logdir logs/supermimic

Play and visualize
------------------

.. code-block:: bash

   python -m supermimic.scripts.play \
       --checkpoint logs/supermimic/g1_supermimic/<timestamp>/checkpoint_final.pt \
       --motion-dir /data/supermimic/g1_1200 \
       --viewer viser --viser-port 8091

The Viser viewer overlays a translucent light-blue copy of the reference robot
(45% opacity), synchronized to the current reference frame. Toggle
it with **SuperMimic reference → Show reference ghost** in the GUI. Use
``--viewer native`` on a machine with a display. For a headless video:

.. code-block:: bash

   python -m supermimic.scripts.play \
       --checkpoint ... --motion-dir /data/supermimic/g1_walk1 \
       --video rollout.mp4 --video-length 300 --compare-reference

``--compare-reference`` is only needed for the side-by-side video; Viser shows
the ghost by default.

``--clip`` accepts a clip name or index, and ``--agent zero``/``random`` runs
sanity-check policies without a checkpoint.

Evaluation
----------

.. code-block:: bash

   python -m supermimic.scripts.evaluate \
       --checkpoint logs/supermimic/g1_supermimic/<timestamp>/checkpoint_final.pt \
       --motion-dir /data/supermimic/g1_1200 \
       --num-envs 64 --episodes-per-env 4

Evaluation uses deterministic physics, no domain randomization, and no
observation or action noise. A motion counts as failed when the training
termination fires. The script reports success rate and normalized return; the
normalizer is the maximum possible per-step reward (``0.1``) times the episode
length.

Implementation map
------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Module
   * - Training loop
     - ``supermimic/algo/trainer.py``
   * - World model / actor / critic
     - ``supermimic/algo/networks.py``
   * - Differentiable G1 kinematics
     - ``supermimic/algo/kinematics.py``
   * - Replay buffer
     - ``supermimic/algo/buffer.py``
   * - Return-based resets
     - ``supermimic/algo/sampler.py``
   * - Tracking reward and termination
     - ``supermimic/algo/reward.py``
   * - Multi-motion command
     - ``supermimic/mdp/commands.py``
   * - Observation and goal features
     - ``supermimic/mdp/features.py``
   * - G1 environment configuration
     - ``supermimic/config/g1/env_cfgs.py``
   * - G1 algorithm/training configuration
     - ``supermimic/config/g1/rl_cfg.py``

Tests
-----

.. code-block:: bash

   pytest tests/test_supermimic_kinematics.py tests/test_supermimic_buffer.py
   pytest tests/test_supermimic_networks.py tests/test_supermimic_config.py
   pytest tests/test_supermimic_smoke.py -m slow
