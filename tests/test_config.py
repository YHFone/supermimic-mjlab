"""G1 SuperMimic configuration wiring (official defaults)."""

from supermimic.algo.config import SuperMimicCfg
from supermimic.config.g1.env_cfgs import unitree_g1_supermimic_env_cfg
from supermimic.config.g1.rl_cfg import unitree_g1_supermimic_cfg
from supermimic.mdp.commands import MotionLibraryCommandCfg


def test_g1_rl_cfg_matches_official_defaults():
  cfg = unitree_g1_supermimic_cfg()
  assert isinstance(cfg, SuperMimicCfg)
  assert cfg.actor_obs_mode == "rep_state"
  assert cfg.obs_history == 1
  assert cfg.goal_frames == 10 and cfg.goal_stride == 5
  assert cfg.termination_softness == 0.05
  assert cfg.horizon == 32
  assert cfg.wm_window == 8
  assert cfg.wm_hidden_dims == (2048, 2048, 2048)
  assert cfg.latent_dim == 32
  assert cfg.gamma == 0.99 and cfg.lam == 0.95
  assert cfg.num_iterations == 1_000_000
  assert cfg.lr_decay_iters == 0
  assert cfg.use_muon is True
  assert cfg.utd == 0.4 and cfg.num_steps_per_env == 2.5
  assert cfg.buffer_steps == 1000
  assert cfg.wm_batch_size == 2048
  assert cfg.wm_updates_per_iter == 10
  assert cfg.policy_batch_size == 4096


def test_g1_env_cfg_uses_multi_motion_command():
  cfg = unitree_g1_supermimic_env_cfg(["dummy.npz"], num_envs=4)
  assert cfg.scene.num_envs == 4
  assert cfg.auto_reset is False
  assert cfg.rewards == {}
  command = cfg.commands["motion"]
  assert isinstance(command, MotionLibraryCommandCfg)
  assert command.goal_frames == 10
  assert command.goal_stride == 5
  assert command.motion_files == ("dummy.npz",)
  for name in ("motion_end", "anchor_pos", "anchor_ori"):
    assert name in cfg.terminations
