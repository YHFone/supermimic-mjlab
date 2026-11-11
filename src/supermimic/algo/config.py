"""Configuration for the SuperMimic training loop.

Defaults follow the official implementation (``HybridRobotics/supermimic``,
``runners/supermimic_runner.py`` and ``learners/*``): GenCoord representation,
``actor_obs="rep_state"`` with a single-frame history, 10 goal frames at stride
5, analytic soft termination, Bryson-weighted world-model loss, Muon optimizer,
constant learning rate and a 1e6-iteration budget.

Where the official code and the paper differ, the official value wins and the
paper value is noted in the field docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SuperMimicCfg:
  """Hyperparameters of the collect → fit → imagine loop."""

  # Data collection.
  buffer_steps: int = 1000
  """Replay-buffer length in simulator steps per environment."""
  num_steps_per_env: float = 2.5
  """Simulator steps collected per collect/fit/imagine round.

  The official runner derives the same value from
  ``N / num_batches / batch_size / horizon * replay_ratio``
  (= 4096/10/2048/8 × 16 = 0.4 updates per step, i.e. 2.5 steps per update).
  The trainer accumulates the fractional part, collecting 2 and 3 steps
  alternately.
  """
  utd: float | None = None
  """Update-to-data ratio; overrides ``num_steps_per_env`` when set."""
  warmup_steps: int = 64
  action_noise: float = 0.1
  sensor_noise_joint_pos: float = 0.01
  sensor_noise_joint_vel: float = 0.5
  sensor_noise_base_ang_vel: float = 0.2
  sensor_noise_projected_gravity: float = 0.05

  # Actor observation.
  actor_obs_mode: Literal["rep_state", "proprio"] = "rep_state"
  """``rep_state`` (official default): the 68-dim GenCoord state, noiseless.

  ``proprio``: the 93-dim deployable sensor observation with uniform noise
  (paper Table A.1).  Only ``proprio`` can be exported for the real robot.
  """
  obs_history: int = 1
  """Actor history frames; official default 1.  ``>1`` requires ``proprio``."""

  # Goal.
  goal_frames: int = 10
  goal_stride: int = 5
  """Reference-frame spacing of the goal window (official ``GOAL_FRAME_STRIDE``)."""

  # Termination.
  termination_softness: float = 0.05
  termination_pos_threshold: float = 0.25
  termination_ori_threshold: float = 0.8

  # World model.
  wm_window: int = 8
  wm_batch_size: int = 2048
  wm_updates_per_iter: int = 10
  wm_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  wm_lr: float = 2.0e-4
  wm_grad_clip: float = 1.0
  wm_nominal_error_root_pos: float = 0.01
  wm_nominal_error_root_ori: float = 0.02
  wm_nominal_error_joint_pos: float = 0.1
  wm_nominal_error_joint_vel: float = 3.14

  # Policy and critic.
  horizon: int = 32
  policy_batch_size: int = 4096
  policy_chunk_size: int = 4096
  """Imagination chunk size; the effective batch stays ``policy_batch_size``."""
  encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_dim: int = 32
  actor_lr: float = 2.0e-4
  critic_lr: float = 2.0e-4
  gamma: float = 0.99
  lam: float = 0.95
  critic_epochs: int = 5
  critic_minibatches: int = 4
  grad_clip: float = 1.0
  grad_clip_elementwise: float = 0.1
  use_muon: bool = True
  """Muon on the hidden ``Linear`` weights of every ``nn.Sequential``."""

  # Normalizer updates.
  norm_interval: int = 10
  norm_batch_size: int = 65_536

  # Schedule.
  num_iterations: int = 1_000_000
  lr_decay_iters: int = 0
  """0 = constant learning rate (official)."""
  lr_final_ratio: float = 0.03

  # Evaluation, logging, and checkpointing.
  eval_interval: int = 0
  eval_batch_size: int = 256
  log_interval: int = 10
  save_interval: int = 1000
  log_dir: str = "logs/supermimic"
  experiment_name: str = "g1_supermimic"
  seed: int = 0
  device: str = "cuda:0"
  compile: bool = False
  """Compile the differentiable-kinematics layer with ``torch.compile``."""
  use_tf32: bool = True
  extra: dict = field(default_factory=dict)

  def __post_init__(self) -> None:
    if self.utd is not None:
      if self.utd <= 0:
        raise ValueError("utd must be positive")
      self.num_steps_per_env = 1.0 / self.utd
    elif self.num_steps_per_env <= 0:
      raise ValueError("num_steps_per_env must be positive")
    else:
      self.utd = 1.0 / self.num_steps_per_env
    if self.obs_history < 1:
      raise ValueError("obs_history must be >= 1")
    if self.obs_history > 1 and self.actor_obs_mode != "proprio":
      raise ValueError(
        "obs_history > 1 is only supported with actor_obs_mode='proprio' "
        "(the official rep_state actor uses a single frame)."
      )
    if self.termination_softness <= 0:
      raise ValueError("termination_softness must be positive")
