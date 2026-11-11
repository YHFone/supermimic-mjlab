"""Contract pins: official GenCoord shapes and hyperparameters.

The paper (Table A.4/A.5) specifies 67-dim goal frames and a 93-dim proprio
observation; the official implementation differs in three places (1-frame
rep_state actor, 640-dim actor goal, analytic termination).  These tests pin the
official values, which is what this package now implements.
"""

import torch

from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.networks import ActorCriticCfg, WorldModelCfg
from supermimic.algo.representation import GenCoordCfg, GenCoordRepresentation
from supermimic.algo.reward import TrackingRewardCfg
from supermimic.config.g1.env_cfgs import unitree_g1_supermimic_env_cfg
from supermimic.config.g1.rl_cfg import unitree_g1_supermimic_cfg
from supermimic.mdp.features import GoalFeatures, ProprioFeatures


def test_schedule_and_collection():
  cfg = unitree_g1_supermimic_cfg()
  assert cfg.buffer_steps == 1000
  assert cfg.num_steps_per_env == 2.5 and cfg.utd == 0.4
  assert cfg.action_noise == 0.1
  assert cfg.obs_history == 1
  assert cfg.goal_frames == 10 and cfg.goal_stride == 5
  assert cfg.num_iterations == 1_000_000
  assert cfg.lr_decay_iters == 0


def test_world_model_contract():
  cfg = unitree_g1_supermimic_cfg()
  assert cfg.wm_window == 8
  assert cfg.wm_batch_size == 2048
  assert cfg.wm_updates_per_iter == 10
  assert cfg.wm_lr == 2.0e-4
  assert cfg.wm_hidden_dims == (2048, 2048, 2048)
  assert WorldModelCfg().nominal_error_root_pos == 0.01
  assert WorldModelCfg().nominal_error_root_ori == 0.02
  assert WorldModelCfg().nominal_error_joint_pos == 0.1
  assert WorldModelCfg().nominal_error_joint_vel == 3.14


def test_actor_critic_contract():
  cfg = unitree_g1_supermimic_cfg()
  assert cfg.horizon == 32
  assert cfg.policy_batch_size == 4096
  assert cfg.actor_lr == 2.0e-4 and cfg.critic_lr == 2.0e-4
  assert cfg.gamma == 0.99 and cfg.lam == 0.95
  assert cfg.critic_epochs == 5 and cfg.critic_minibatches == 4
  assert cfg.grad_clip == 1.0 and cfg.grad_clip_elementwise == 0.1
  assert cfg.latent_dim == 32
  assert cfg.encoder_hidden_dims == (2048, 1024, 512, 256)
  assert cfg.critic_hidden_dims == (256, 128)
  assert ActorCriticCfg().goal_dim == 640
  assert ActorCriticCfg().wm_goal_dim == 670


def test_representation_and_goal_dims():
  rep = GenCoordRepresentation(GenCoordCfg(num_joints=29))
  assert rep.rep_state_dim == 68
  assert rep.prediction_dim == 35
  assert rep.aux_state_dim == 7
  goal = GoalFeatures(num_joints=29, anchor_index=0)
  assert goal.wm_dim == 670 and goal.actor_dim == 640
  assert ProprioFeatures(29, torch.zeros(29)).dim == 93


def test_env_thresholds_and_reward():
  cfg = unitree_g1_supermimic_env_cfg(["dummy.npz"], num_envs=4)
  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.25
  assert cfg.terminations["anchor_ori"].params["threshold"] == 0.8
  assert cfg.events["foot_friction"].params["ranges"] == (0.3, 1.2)
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.01, 0.01)
  reward = TrackingRewardCfg()
  assert (reward.root_pos_weight, reward.root_ori_weight) == (0.5, 0.5)
  assert reward.joint_limit_weight == -2.0
  assert reward.dt == 0.02
