"""End-to-end smoke test: a few SuperMimic iterations on a synthetic motion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.kinematics import (
  DifferentiableKinematics,
  build_kinematic_tree,
)
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.g1.env_cfgs import (
  unitree_g1_supermimic_env_cfg,
)
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul


def _write_synthetic_motion(path: Path, frames: int = 80) -> None:
  entity = Entity(get_g1_robot_cfg())
  model = entity.spec.compile()
  kin = DifferentiableKinematics(
    build_kinematic_tree(model, entity.body_names, entity.joint_names)
  )
  key = torch.tensor(model.key_qpos[0], dtype=torch.float32)
  t = torch.linspace(0, 1, frames)
  joint_pos = key[7:][None] + 0.05 * torch.sin(2 * torch.pi * t)[:, None]
  root_pos = key[:3][None].repeat(frames, 1)
  root_pos[:, 2] += 0.01 * torch.sin(2 * torch.pi * t)
  root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(frames, 1)
  dt = 0.02
  root_lin_vel = torch.gradient(root_pos, spacing=dt, dim=0)[0]
  joint_vel = torch.gradient(joint_pos, spacing=dt, dim=0)[0]
  q_rel = quat_mul(root_quat[2:], quat_conjugate(root_quat[:-2]))
  omega = axis_angle_from_quat(q_rel) / (2 * dt)
  root_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  body_pos, body_quat, body_lin_vel, body_ang_vel = kin.forward_with_velocity(
    root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel
  )
  np.savez(
    path,
    fps=np.array([50.0], dtype=np.float32),
    body_names=np.array(entity.body_names),
    joint_names=np.array(entity.joint_names),
    joint_pos=joint_pos.numpy(),
    joint_vel=joint_vel.numpy(),
    body_pos_w=body_pos.numpy(),
    body_quat_w=body_quat.numpy(),
    body_lin_vel_w=body_lin_vel.numpy(),
    body_ang_vel_w=body_ang_vel.numpy(),
  )


@pytest.mark.slow
def test_training_smoke(tmp_path):
  if not torch.cuda.is_available():
    pytest.skip("SuperMimic training smoke test requires CUDA.")
  motion = tmp_path / "standing.npz"
  _write_synthetic_motion(motion)
  env_cfg = unitree_g1_supermimic_env_cfg([str(motion)], num_envs=2)
  env_cfg.episode_length_s = 0.8
  env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
  cfg = SuperMimicCfg(
    buffer_steps=32,
    num_steps_per_env=1.0,
    warmup_steps=20,
    wm_batch_size=8,
    wm_updates_per_iter=1,
    wm_hidden_dims=(32, 32),
    policy_batch_size=4,
    horizon=4,
    encoder_hidden_dims=(32,),
    decoder_hidden_dims=(32,),
    critic_hidden_dims=(16,),
    critic_epochs=1,
    critic_minibatches=1,
    norm_batch_size=64,
    norm_interval=1,
    num_iterations=2,
    log_interval=1,
    save_interval=0,
    log_dir=str(tmp_path),
    experiment_name="smoke",
    device="cuda:0",
  )
  trainer = SuperMimicTrainer(env, cfg)
  try:
    trainer.train()
    path = trainer.save("final.pt")
    assert path.exists()
  finally:
    trainer.close()
    env.close()
