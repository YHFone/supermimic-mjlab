"""Differentiable goal features (official GenCoord contract).

Goal layout follows ``HybridRobotics/supermimic``
(``representations.py:96-116,213-234``):

* a window of ``N = 10`` reference frames spaced ``stride = 5`` control steps
  apart, starting at the current frame ``t``;
* per-frame content ``[joint_pos(J), joint_vel(J)]`` with **absolute** joint
  angles;
* ``rel_pos`` — anchor position relative to the **robot root**, in the root
  frame (world-model goal only);
* ``rel_ori`` — anchor orientation relative to the robot root, as a 6D rotation
  (both goals).

Concatenation is block-major, not frame-major::

    wm_goal    = [content (N*2J), rel_pos (N*3), rel_ori (N*6)]   # 670
    actor_goal = [content (N*2J),              rel_ori (N*6)]   # 640

The actor's proprio layout is also the official one::

    [projected_gravity(3), base_ang_vel(3), joint_pos_rel(J), joint_vel(J), action(J)]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)

NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5
GOAL_REL_POS_DIM = 3 * NUM_GOAL_FRAMES
GOAL_REL_ORI_DIM = 6 * NUM_GOAL_FRAMES


class GoalFeatures:
  """Official goal window: ``N`` strided frames, root-relative anchor."""

  def __init__(
    self,
    num_joints: int,
    anchor_index: int,
    *,
    horizon: int = NUM_GOAL_FRAMES,
  ) -> None:
    self.num_joints = num_joints
    self.anchor_index = anchor_index
    self.horizon = horizon
    self.content_dim = horizon * 2 * num_joints
    self.rel_pos_dim = horizon * 3
    self.rel_ori_dim = horizon * 6
    self.wm_dim = self.content_dim + self.rel_pos_dim + self.rel_ori_dim
    self.actor_dim = self.content_dim + self.rel_ori_dim

  def from_reference(
    self,
    reference: dict[str, torch.Tensor],
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``(B, N, ...)`` reference frames into ``(wm_goal, actor_goal)``."""
    joint_pos = reference["joint_pos"]  # (B, N, J) absolute
    joint_vel = reference["joint_vel"]
    anchor_pos = reference["body_pos"][:, :, self.anchor_index]
    anchor_quat = reference["body_quat"][:, :, self.anchor_index]
    batch, horizon = joint_pos.shape[0], joint_pos.shape[1]

    root_pos_e = root_pos[:, None, :].expand(batch, horizon, 3)
    root_quat_e = root_quat[:, None, :].expand(batch, horizon, 4)
    rel_pos = quat_apply_inverse(root_quat_e, anchor_pos - root_pos_e)
    rel_ori_quat = quat_mul(quat_inv(root_quat_e), anchor_quat)
    rel_ori = matrix_from_quat(rel_ori_quat)[..., :2].reshape(batch, horizon, 6)

    content = torch.cat([joint_pos, joint_vel], dim=-1).reshape(batch, -1)
    rel_pos_flat = rel_pos.reshape(batch, -1)
    rel_ori_flat = rel_ori.reshape(batch, -1)
    wm_goal = torch.cat([content, rel_pos_flat, rel_ori_flat], dim=-1)
    actor_goal = torch.cat([content, rel_ori_flat], dim=-1)
    return wm_goal, actor_goal


def make_state(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> torch.Tensor:
  """Assemble the full state ``[p, q, qj, v, ω, vj]`` (official column order)."""
  return torch.cat(
    [root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel],
    dim=-1,
  )
