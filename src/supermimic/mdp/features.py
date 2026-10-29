"""Differentiable observation and goal features for SuperMimic.

The policy observation and goal encoder input are pure functions of the robot
state, the reference motion, and the current root pose.  Keeping them in one
place guarantees that the observation used during simulator data collection is
identical to the one used inside imagined world-model rollouts.
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

# State layout.
ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
ROOT_LIN_VEL = slice(7, 10)
ROOT_ANG_VEL = slice(10, 13)
JOINT_POS = slice(13, None)  # Joint slice end is set by the robot.

# Goal layout: per reference frame.
GOAL_ANCHOR_POS = 3
GOAL_ANCHOR_ORI = 6


@dataclass(frozen=True)
class ProprioNoiseCfg:
  """Per-dimension uniform sensor noise (Table A.1 of the paper)."""

  joint_pos: float = 0.01
  joint_vel: float = 0.5
  base_ang_vel: float = 0.2
  projected_gravity: float = 0.05


class ProprioFeatures:
  """Builds the 93-dimensional proprioceptive observation of the paper."""

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
    """Compute ``o_t`` from a world-model state.

    Args:
      state: ``(B, 13 + 2 * J)`` state.
      last_action: ``(B, J)`` previous raw action.
      encoder_bias: ``(B, J)`` joint calibration offsets, added to the
        measured joint positions exactly as the simulator's noisy encoder does.
      add_noise: whether to add the sensor noise of Table A.1.
    """
    root_quat = state[:, ROOT_QUAT]
    joint_pos = state[:, 13 : 13 + self.num_joints]
    joint_vel = state[:, 13 + self.num_joints :]
    base_ang_vel_w = state[:, ROOT_ANG_VEL]
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
      [joint_pos_rel, joint_vel, base_ang_vel_b, projected_gravity_b, last_action],
      dim=-1,
    )


def _uniform(scale: float, like: torch.Tensor) -> torch.Tensor:
  return (torch.rand_like(like) * 2.0 - 1.0) * scale


class GoalFeatures:
  """Builds the K-frame future-reference goal used by the goal encoder.

  The anchor *position* is expressed in the reference's own local frame: frame
  0 of the gathered window (the reference frame preceding the goal window)
  supplies the origin.  The anchor *orientation* stays relative to the robot's
  current root orientation.
  """

  def __init__(
    self,
    num_joints: int,
    default_joint_pos: torch.Tensor,
    anchor_index: int,
    horizon: int,
  ) -> None:
    self.num_joints = num_joints
    self.default_joint_pos = default_joint_pos
    self.anchor_index = anchor_index
    self.horizon = horizon
    self.frame_dim = 2 * num_joints + GOAL_ANCHOR_POS + GOAL_ANCHOR_ORI
    self.dim = horizon * self.frame_dim

  def from_reference(
    self, reference: dict[str, torch.Tensor], root_quat: torch.Tensor
  ) -> torch.Tensor:
    """Encode ``(B, K + 1, ...)`` reference frames as the goal.

    Frame 0 is the reference frame preceding the goal window (``t - 1``) and
    anchors the position term: frames ``1..K`` are the K goal frames, whose
    anchor positions are relative to the reference anchor body at ``t - 1`` and
    expressed in its frame.  The anchor orientations are relative to
    ``root_quat``, the robot's current root orientation.
    """
    ref_pos = reference["body_pos"][:, :, self.anchor_index]  # (B, K + 1, 3)
    ref_quat = reference["body_quat"][:, :, self.anchor_index]  # (B, K + 1, 4)
    pos_origin = ref_pos[:, 0]
    pos_origin_quat = ref_quat[:, 0]
    ref_pos = ref_pos[:, 1:]
    ref_quat = ref_quat[:, 1:]
    batch, horizon = ref_pos.shape[0], ref_pos.shape[1]

    pos_origin_e = pos_origin[:, None, :]
    pos_origin_quat_e = pos_origin_quat[:, None, :].expand(batch, horizon, 4)
    anchor_pos_b = quat_apply_inverse(
      pos_origin_quat_e, ref_pos - pos_origin_e
    )
    root_quat_e = root_quat[:, None, :].expand(batch, horizon, 4)
    anchor_quat_b = quat_mul(quat_inv(root_quat_e), ref_quat)
    anchor_ori_b = matrix_from_quat(anchor_quat_b)[..., :2].reshape(batch, horizon, 6)
    joint_pos = (
      reference["joint_pos"][:, 1:] - self.default_joint_pos[None, None, :]
    )
    goal = torch.cat(
      [joint_pos, reference["joint_vel"][:, 1:], anchor_pos_b, anchor_ori_b], dim=-1
    )
    return goal.reshape(batch, self.dim)


def make_state(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> torch.Tensor:
  """Assemble the world-model state vector."""
  return torch.cat(
    [root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel],
    dim=-1,
  )
