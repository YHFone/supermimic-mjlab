"""Training configuration for the Unitree G1 SuperMimic task."""

from __future__ import annotations

from supermimic.algo.config import SuperMimicCfg


def unitree_g1_supermimic_cfg() -> SuperMimicCfg:
  """Create the training configuration for the Unitree G1 SuperMimic task."""
  return SuperMimicCfg(
    buffer_steps=1000,
    num_steps_per_env=2.5,
    warmup_steps=64,
    action_noise=0.1,
    obs_history=10,
    goal_frames=10,
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
    norm_interval=10,
    norm_batch_size=65_536,
    num_iterations=320_000,
    lr_decay_iters=250_000,
    lr_final_ratio=0.03,
    log_interval=1,
    save_interval=500,
    log_dir="logs/supermimic",
    experiment_name="g1_supermimic",
    seed=0,
    device="cuda:0",
  )
