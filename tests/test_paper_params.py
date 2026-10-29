"""Pin every SuperMimic hyperparameter to the paper (Tables A.1-A.5)."""

import torch

from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.networks import (
  ActorCriticCfg,
  WorldModelCfg,
)
from supermimic.algo.reward import TrackingRewardCfg
from supermimic.config.g1.env_cfgs import (
  unitree_g1_supermimic_env_cfg,
)
from supermimic.config.g1.rl_cfg import (
  unitree_g1_supermimic_cfg,
)
from supermimic.mdp.features import (
  GoalFeatures,
  ProprioFeatures,
)
from supermimic.scripts.train import TrainArgs


def test_table_a5_schedule_and_collection():
  cfg = unitree_g1_supermimic_cfg()
  assert isinstance(cfg, SuperMimicCfg)
  # Data collection.
  assert cfg.buffer_steps == 1000
  assert cfg.num_steps_per_env == 2.5
  assert cfg.utd == 0.4
  assert cfg.action_noise == 0.1
  # Observation / goal.
  assert cfg.obs_history == 10
  assert cfg.goal_frames == 10
  # Schedule.
  assert cfg.num_iterations == 320_000
  assert cfg.lr_decay_iters == 250_000
  assert cfg.lr_final_ratio == 0.03


def test_table_a5_world_model():
  cfg = unitree_g1_supermimic_cfg()
  assert cfg.wm_window == 8
  assert cfg.wm_batch_size == 2048
  assert cfg.wm_updates_per_iter == 10
  assert cfg.wm_lr == 2.0e-4
  assert cfg.wm_grad_clip == 1.0
  assert cfg.wm_hidden_dims == (2048, 2048, 2048)


def test_table_a5_imagination_policy_critic():
  cfg = unitree_g1_supermimic_cfg()
  assert cfg.horizon == 32
  assert cfg.policy_batch_size == 4096
  assert cfg.policy_chunk_size <= cfg.policy_batch_size
  assert cfg.actor_lr == 2.0e-4
  assert cfg.critic_lr == 2.0e-4
  assert cfg.gamma == 0.99
  assert cfg.lam == 0.95
  assert cfg.critic_epochs == 5
  assert cfg.critic_minibatches == 4
  assert cfg.grad_clip == 1.0
  assert cfg.grad_clip_elementwise == 0.1
  assert cfg.latent_dim == 32
  assert cfg.encoder_hidden_dims == (2048, 1024, 512, 256)
  assert cfg.decoder_hidden_dims == (2048, 1024, 512, 256)
  assert cfg.critic_hidden_dims == (256, 128)
  assert cfg.sensor_noise_joint_pos == 0.01
  assert cfg.sensor_noise_joint_vel == 0.5
  assert cfg.sensor_noise_base_ang_vel == 0.2
  assert cfg.sensor_noise_projected_gravity == 0.05


def test_table_a1_a2_a3_env():
  cfg = unitree_g1_supermimic_env_cfg(["dummy.npz"], num_envs=4096)
  assert cfg.scene.num_envs == 4096
  command = cfg.commands["motion"]
  assert command.goal_frames == 10
  assert command.anchor_body_name == "pelvis"
  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.25
  assert cfg.terminations["anchor_ori"].params["threshold"] == 0.8
  assert "ee_body_pos" not in cfg.terminations
  assert cfg.events["foot_friction"].params["ranges"] == (0.3, 1.2)
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.01, 0.01)
  com_ranges = cfg.events["base_com"].params["ranges"]
  assert com_ranges[0] == (-0.025, 0.025)
  assert com_ranges[1] == (-0.05, 0.05)
  assert com_ranges[2] == (-0.05, 0.05)

  reward = TrackingRewardCfg()
  assert reward.root_pos_weight == 0.5
  assert reward.root_ori_weight == 0.5
  assert reward.body_pos_weight == 1.0
  assert reward.body_ori_weight == 1.0
  assert reward.body_lin_vel_weight == 1.0
  assert reward.body_ang_vel_weight == 1.0
  assert reward.action_rate_weight == -0.1
  assert reward.joint_limit_weight == -2.0
  assert reward.root_pos_std == 0.3
  assert reward.root_ori_std == 0.4
  assert reward.body_pos_std == 0.3
  assert reward.body_ori_std == 0.4
  assert reward.body_lin_vel_std == 1.0
  assert reward.body_ang_vel_std == 3.14
  assert reward.dt == 0.02


def test_table_a4_dimensions():
  proprio = ProprioFeatures(29, default_joint_pos=torch.zeros(29))
  assert proprio.dim == 93
  goal = GoalFeatures(29, torch.zeros(29), anchor_index=0, horizon=10)
  assert goal.frame_dim == 73
  assert goal.dim == 730
  assert WorldModelCfg().hidden_dims == (2048, 2048, 2048)
  assert WorldModelCfg().context_dim == 33
  assert ActorCriticCfg().context_dim == 33
  assert ActorCriticCfg().latent_dim == 32


def test_env_and_train_defaults():
  assert TrainArgs().num_envs == 4096
  assert TrainArgs().sm.utd == 0.4
