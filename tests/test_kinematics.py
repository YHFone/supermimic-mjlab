"""Differentiable FK must agree with MuJoCo on the Unitree G1."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from supermimic.algo.kinematics import (
  DifferentiableKinematics,
  build_kinematic_tree,
)
from mjlab.utils.lab_api.math import quat_apply_inverse


@pytest.fixture(scope="module")
def g1_tree():
  entity = Entity(get_g1_robot_cfg())
  model = entity.spec.compile()
  return model, entity.body_names, entity.joint_names


def test_fk_matches_mujoco(g1_tree):
  model, body_names, joint_names = g1_tree
  kin = DifferentiableKinematics(build_kinematic_tree(model, body_names, joint_names))
  batch = 4
  torch.manual_seed(0)
  root_pos = torch.randn(batch, 3) * 0.2
  root_pos[:, 2] += 0.8
  root_quat = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
  joint_pos = torch.randn(batch, len(joint_names)) * 0.4
  root_lin_vel = torch.randn(batch, 3) * 0.3
  root_ang_vel = torch.randn(batch, 3) * 0.3
  joint_vel = torch.randn(batch, len(joint_names)) * 0.5
  body_pos, body_quat, body_lin_vel, body_ang_vel = kin.forward_with_velocity(
    root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel
  )
  data = mujoco.MjData(model)
  joint_q_adr = [model.jnt_qposadr[model.joint(name).id] for name in joint_names]
  joint_v_adr = [model.jnt_dofadr[model.joint(name).id] for name in joint_names]
  body_ids = [model.body(name).id for name in body_names]
  expected_pos, expected_quat, expected_lin, expected_ang = [], [], [], []
  for i in range(batch):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = root_pos[i].numpy()
    data.qpos[3:7] = root_quat[i].numpy()
    data.qpos[joint_q_adr] = joint_pos[i].numpy()
    data.qvel[:3] = root_lin_vel[i].numpy()
    data.qvel[3:6] = quat_apply_inverse(root_quat[i][None], root_ang_vel[i][None])[
      0
    ].numpy()
    data.qvel[joint_v_adr] = joint_vel[i].numpy()
    mujoco.mj_forward(model, data)
    expected_pos.append(data.xpos[body_ids].copy())
    expected_quat.append(data.xquat[body_ids].copy())
    lin, ang = [], []
    root_subtree_com = data.subtree_com[body_ids[0]]
    for body_id in body_ids:
      cvel = data.cvel[body_id]
      offset = root_subtree_com - data.xpos[body_id]
      lin.append(cvel[3:6] - np.cross(cvel[0:3], offset))
      ang.append(cvel[0:3].copy())
    expected_lin.append(lin)
    expected_ang.append(ang)
  expected_pos = torch.tensor(np.asarray(expected_pos), dtype=torch.float32)
  expected_quat = torch.tensor(np.asarray(expected_quat), dtype=torch.float32)
  expected_lin = torch.tensor(np.asarray(expected_lin), dtype=torch.float32)
  expected_ang = torch.tensor(np.asarray(expected_ang), dtype=torch.float32)
  torch.testing.assert_close(body_pos, expected_pos, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(
    torch.minimum(
      (body_quat - expected_quat).abs(), (-body_quat - expected_quat).abs()
    ).max(),
    torch.tensor(0.0),
    atol=1e-5,
    rtol=0.0,
  )
  torch.testing.assert_close(body_lin_vel, expected_lin, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(body_ang_vel, expected_ang, atol=1e-5, rtol=1e-5)


def test_fk_is_differentiable(g1_tree):
  model, body_names, joint_names = g1_tree
  kin = DifferentiableKinematics(build_kinematic_tree(model, body_names, joint_names))
  root_pos = torch.zeros(2, 3, requires_grad=True)
  root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
  joint_pos = torch.zeros(2, len(joint_names), requires_grad=True)
  body_pos, _ = kin(root_pos, root_quat, joint_pos)
  body_pos[:, 5].sum().backward()
  assert root_pos.grad is not None and joint_pos.grad is not None
  assert torch.isfinite(root_pos.grad).all()
