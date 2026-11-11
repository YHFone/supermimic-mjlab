"""Differentiable observation and goal features (official GenCoord contract).

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


@dataclass(frozen=True)
class ProprioNoiseCfg:
  """Per-dimension uniform sensor noise (Table A.1 of the paper)."""

  joint_pos: float = 0.01
  joint_vel: float = 0.5
  base_ang_vel: float = 0.2
  projected_gravity: float = 0.05
  actions: float = 0.0

  def bounds(self, num_joints: int) -> torch.Tensor:
    """Half-widths in the official proprio order."""
    return torch.tensor(
      [self.projected_gravity] * 3
      + [self.base_ang_vel] * 3
      + [self.joint_pos] * num_joints
      + [self.joint_vel] * num_joints
      + [self.actions] * num_joints,
      dtype=torch.float32,
    )


class ProprioFeatures:
  """Builds the 93-dimensional sensor proprio in the official term order."""

  def __init__(
    self,
    num_joints: int,
    default_joint_pos: torch.Tensor,
    noise: ProprioNoiseCfg | None = None,
  ) -> None:
    self.num_joints = num_joints
    self.default_joint_pos = default_joint_pos
    self.noise = noise or ProprioNoiseCfg()
    self.dim = 3 * num_joints + 6

  def __call__(
    self,
    state: torch.Tensor,
    last_action: torch.Tensor,
    encoder_bias: torch.Tensor | None = None,
    *,
    add_noise: bool = True,
  ) -> torch.Tensor:
    """``[gravity, ang_vel, qj_rel, vj, action]`` from a 71-dim state."""
    root_quat = state[:, 3:7]
    joint_pos = state[:, 13 : 13 + self.num_joints]
    joint_vel = state[:, 13 + self.num_joints :]
    base_ang_vel_w = state[:, 10:13]
    joint_pos_rel = joint_pos - self.default_joint_pos
    if encoder_bias is not None:
      joint_pos_rel = joint_pos_rel + encoder_bias
    base_ang_vel_b = quat_apply_inverse(root_quat, base_ang_vel_w)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=state.device).expand_as(
      base_ang_vel_w
    )
    projected_gravity_b = quat_apply_inverse(root_quat, gravity_w)
    if add_noise:
      joint_pos_rel = joint_pos_rel + _uniform(self.noise.joint_pos, joint_pos_rel)
      joint_vel = joint_vel + _uniform(self.noise.joint_vel, joint_vel)
      base_ang_vel_b = base_ang_vel_b + _uniform(
        self.noise.base_ang_vel, base_ang_vel_b
      )
      projected_gravity_b = projected_gravity_b + _uniform(
        self.noise.projected_gravity, projected_gravity_b
      )
    return torch.cat(
      [projected_gravity_b, base_ang_vel_b, joint_pos_rel, joint_vel, last_action],
      dim=-1,
    )


def _uniform(scale: float, like: torch.Tensor) -> torch.Tensor:
  return (torch.rand_like(like) * 2.0 - 1.0) * scale


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
  """Assemble the full state ``[p, q, v, ω, qj, vj]`` (local column order)."""
  return torch.cat(
    [root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel],
    dim=-1,
  )
