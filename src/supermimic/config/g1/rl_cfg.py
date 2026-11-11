"""Training configuration for the Unitree G1 SuperMimic task.

Defaults follow the official implementation: GenCoord representation,
``actor_obs="rep_state"`` with one frame of history, 10 goal frames at stride 5,
analytic termination, constant learning rate and a 1e6-iteration budget.
"""

from __future__ import annotations

from supermimic.algo.config import SuperMimicCfg


def unitree_g1_supermimic_cfg() -> SuperMimicCfg:
  """Create the training configuration for the Unitree G1 SuperMimic task."""
  return SuperMimicCfg(
    buffer_steps=1000,
    num_steps_per_env=2.5,
    warmup_steps=64,
    action_noise=0.1,
    actor_obs_mode="rep_state",
    obs_history=1,
    wm_loss_mode="bryson",
    use_muon_actor=True,
    goal_frames=10,
    goal_stride=5,
    termination_softness=0.05,
    wm_window=8,
    wm_batch_size=2048,
    wm_updates_per_iter=10,
    wm_hidden_dims=(2048, 2048, 2048),
    wm_lr=2.0e-4,
    wm_grad_clip=1.0,
    horizon=32,
    policy_batch_size=4096,
    policy_chunk_size=4096,
    compile=True,
    use_tf32=True,
    encoder_hidden_dims=(2048, 1024, 512, 256),
    decoder_hidden_dims=(2048, 1024, 512, 256),
    critic_hidden_dims=(256, 128),
    latent_dim=32,
    actor_lr=2.0e-4,
    critic_lr=2.0e-4,
    gamma=0.99,
    lam=0.95,
    critic_epochs=5,
    critic_minibatches=4,
    grad_clip=1.0,
    grad_clip_elementwise=0.1,
    use_muon=True,
    norm_interval=1,
    norm_batch_size=65_536,
    num_iterations=1_000_000,
    lr_decay_iters=0,
    log_interval=10,
    save_interval=1000,
    log_dir="logs/supermimic",
    experiment_name="g1_supermimic",
    seed=0,
    device="cuda:0",
  )
