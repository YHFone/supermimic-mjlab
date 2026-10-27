"""G1 SuperMimic configuration wiring."""

from supermimic.algo.config import SuperMimicCfg
from supermimic.config.g1.env_cfgs import (
  unitree_g1_supermimic_env_cfg,
)
from supermimic.config.g1.rl_cfg import (
  unitree_g1_supermimic_cfg,
)
from supermimic.mdp.commands import MotionLibraryCommandCfg


def test_g1_rl_cfg_matches_paper_defaults():
  cfg = unitree_g1_supermimic_cfg()
  assert isinstance(cfg, SuperMimicCfg)
  assert cfg.horizon == 32
  assert cfg.wm_window == 8
  assert cfg.obs_history == 10
  assert cfg.goal_frames == 10
  assert cfg.wm_hidden_dims == (2048, 2048, 2048)
  assert cfg.latent_dim == 32
  assert cfg.gamma == 0.99
  assert cfg.lam == 0.95
  assert cfg.num_iterations == 320_000
  assert cfg.lr_decay_iters == 250_000
  assert cfg.utd == 0.4
  assert cfg.num_steps_per_env == 2.5
  assert cfg.buffer_steps == 1000
  assert cfg.wm_batch_size == 2048
  assert cfg.wm_updates_per_iter == 10
  assert cfg.policy_batch_size == 4096


def test_train_script_uses_g1_cfg_by_default():
  from supermimic.scripts.train import TrainArgs

  args = TrainArgs()
  assert args.sm.num_iterations == 320_000
  assert args.sm.horizon == 32
  assert args.num_envs == 4096


def test_g1_env_cfg_uses_multi_motion_command():
  cfg = unitree_g1_supermimic_env_cfg(["dummy.npz"], num_envs=4)
  assert cfg.scene.num_envs == 4
  assert cfg.auto_reset is False
  assert cfg.rewards == {}
  command = cfg.commands["motion"]
  assert isinstance(command, MotionLibraryCommandCfg)
  assert command.goal_frames == 10
  assert command.motion_files == ("dummy.npz",)
  assert "motion_end" in cfg.terminations
  assert "anchor_pos" in cfg.terminations
  assert "anchor_ori" in cfg.terminations
