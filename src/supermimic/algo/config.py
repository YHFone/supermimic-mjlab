"""Configuration for the SuperMimic training loop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SuperMimicCfg:
  """Hyperparameters of Algorithm A.1.

  Defaults follow Table A.5 of the paper.  The batch sizes are smaller by
  default because a single 24 GB consumer GPU is the reference machine for
  this implementation; every value is configurable from the command line.
  """

  # Data collection.
  buffer_steps: int = 1000
  """Replay-buffer length in simulator steps per environment (Table A.5)."""
  num_steps_per_env: float = 2.5
  """Vectorized simulator steps collected per collect/fit/imagine round.

  Table A.5 reports this as an update-to-data ratio of 0.4, i.e.
  ``1 / 0.4 = 2.5`` steps per round.  The trainer accumulates the fractional
  part, so it collects 2 and 3 steps alternately.
  """
  utd: float | None = None
  """Update-to-data ratio.  If set, it overrides ``num_steps_per_env``."""
  warmup_steps: int = 64
  action_noise: float = 0.1
  sensor_noise_joint_pos: float = 0.01
  sensor_noise_joint_vel: float = 0.5
  sensor_noise_base_ang_vel: float = 0.2
  sensor_noise_projected_gravity: float = 0.05

  # Observation.
  obs_history: int = 10
  goal_frames: int = 10

  # World model.
  wm_window: int = 8
  wm_batch_size: int = 2048
  wm_updates_per_iter: int = 10
  wm_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  wm_term_hidden_dims: tuple[int, ...] = (256, 128)
  wm_term_loss_weight: float = 1.0
  """Weight of the termination BCE in the world-model loss (Eq. 1)."""
  wm_lr: float = 2.0e-4
  wm_grad_clip: float = 1.0

  # Policy and critic.
  horizon: int = 32
  policy_batch_size: int = 4096
  policy_chunk_size: int = 4096
  """Imagination chunk size.  The effective batch is ``policy_batch_size``;
  chunks are only a gradient-accumulation device for a 24 GB GPU."""
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

  # Normalizer updates.
  norm_interval: int = 10
  norm_batch_size: int = 65_536
  critic_norm_batch_size: int = 8192
  """Batch size for the critic input normalizer.

  Smaller than ``norm_batch_size`` because each sample also gathers
  ``goal_frames + 1`` reference frames from the motion library.
  """

  # Schedule.
  num_iterations: int = 320_000
  lr_decay_iters: int = 250_000
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
  """Compile the differentiable-kinematics layer with ``torch.compile``.

  The FK is the hottest part of an imagined step; compiling it gives roughly a
  7x forward+backward speedup on an RTX 4090 and does not change the model.
  """
  use_tf32: bool = True
  """Use TF32 matmuls, as the paper does for world-model rollouts."""
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
