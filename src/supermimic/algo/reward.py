"""Differentiable motion-tracking reward.

The terms follow BeyondMimic's tracking reward as instantiated by mjlab, with
the two changes SuperMimic makes (Appendix A of the paper): the self-collision
penalty is removed and the joint-limit weight is lowered from -10 to -2.

Everything here is a pure function of the world-model state and a reference
frame, so the same code scores simulator transitions during data collection
and imagined transitions during the policy update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .kinematics import DifferentiableKinematics
from .representation import decompose_full_state

TRACKING_REWARD_TERMS: tuple[str, ...] = (
  "root_pos",
  "root_ori",
  "body_pos",
  "body_ori",
  "body_lin_vel",
  "body_ang_vel",
  "action_rate",
  "joint_limit",
)


@dataclass(frozen=True)
class TrackingRewardCfg:
  """Weights and standard deviations of the tracking reward."""

  dt: float = 0.02
  body_pos_weight: float = 1.0
  body_ori_weight: float = 1.0
  body_lin_vel_weight: float = 1.0
  body_ang_vel_weight: float = 1.0
  root_pos_weight: float = 0.5
  root_ori_weight: float = 0.5
  action_rate_weight: float = -0.1
  joint_limit_weight: float = -2.0
  body_pos_std: float = 0.3
  body_ori_std: float = 0.4
  body_lin_vel_std: float = 1.0
  body_ang_vel_std: float = 3.14
  root_pos_std: float = 0.3
  root_ori_std: float = 0.4


def soft_termination_prob(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  ref_anchor_pos: torch.Tensor,
  ref_anchor_quat: torch.Tensor,
  *,
  pos_threshold: float = 0.25,
  ori_threshold: float = 0.8,
  softness: float = 0.05,
) -> torch.Tensor:
  """Analytic, parameter-free termination probability (official implementation).

  ``d = σ((max(|Δz|/0.25, |Δg_z|/0.8) − 1) / softness)`` where ``Δz`` is the
  reference anchor height minus the root height and ``Δg_z`` is the difference
  of the body-frame gravity z component between the reference and the robot.
  For a unit quaternion ``(w, x, y, z)`` that component is ``2(x² + y²) − 1``.

  Replaces the learned BCE head: it is differentiable in the state, needs no
  goal input, and reproduces the simulator's own height/tilt thresholds with a
  softness band of ``softness`` (default 0.05, i.e. 5 % of the threshold).
  """
  height_error = (ref_anchor_pos[..., 2] - root_pos[..., 2]).abs()
  ref_gravity_z = 2.0 * (ref_anchor_quat[..., 1:3] ** 2).sum(-1) - 1.0
  root_gravity_z = 2.0 * (root_quat[..., 1:3] ** 2).sum(-1) - 1.0
  tilt_error = (ref_gravity_z - root_gravity_z).abs()
  normalized = torch.maximum(
    height_error / pos_threshold, tilt_error / ori_threshold
  )
  return torch.sigmoid((normalized - 1.0) / softness)


def _quat_error_squared(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Squared geodesic angle ``||log(R1^T R2)||^2`` (paper Eq. A.2).

  Matches the official ``rotation_error_magnitude`` (``utils/helpers.py:108-127``):
  ``trace(R1^T R2)`` is clamped to ``1 - 1e-7`` so ``acos`` keeps a finite
  gradient when two orientations coincide in an imagined rollout.
  """
  mat1 = matrix_from_quat(q1)
  mat2 = matrix_from_quat(q2)
  trace = torch.sum(mat1 * mat2, dim=(-2, -1))
  cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
  return torch.acos(cos_angle) ** 2


class TrackingReward(nn.Module):
  """Batched, differentiable tracking reward."""

  def __init__(
    self,
    kinematics: DifferentiableKinematics,
    tracked_body_indices: torch.Tensor,
    anchor_index: int,
    soft_joint_pos_limits: torch.Tensor,
    cfg: TrackingRewardCfg | None = None,
  ) -> None:
    super().__init__()
    self.kinematics = kinematics
    self.cfg = cfg or TrackingRewardCfg()
    self.register_buffer("tracked_body_indices", tracked_body_indices.clone())
    self.register_buffer("soft_joint_pos_limits", soft_joint_pos_limits.clone().float())
    self.anchor_index = anchor_index

  def forward(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    reference: dict[str, torch.Tensor],
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_joints = action.shape[-1]
    root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel = (
      decompose_full_state(state, num_joints)
    )
    body_pos, body_quat, body_lin_vel, body_ang_vel = (
      self.kinematics.forward_with_velocity(
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
      )
    )
    return self.forward_from_bodies(
      state,
      action,
      prev_action,
      reference,
      body_pos,
      body_quat,
      body_lin_vel,
      body_ang_vel,
    )

  def forward_from_bodies(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    reference: dict[str, torch.Tensor],
    body_pos_all: torch.Tensor,
    body_quat_all: torch.Tensor,
    body_lin_vel_all: torch.Tensor,
    body_ang_vel_all: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward from already-computed body poses and velocities."""
    cfg = self.cfg
    num_joints = action.shape[-1]
    _, _, joint_pos, _, _, _ = decompose_full_state(state, num_joints)
    ids = self.tracked_body_indices
    body_pos = body_pos_all[:, ids]
    body_quat = body_quat_all[:, ids]
    body_lin_vel = body_lin_vel_all[:, ids]
    body_ang_vel = body_ang_vel_all[:, ids]
    anchor_pos = body_pos[:, self.anchor_index]
    anchor_quat = body_quat[:, self.anchor_index]

    ref_body_pos = reference["body_pos"]
    ref_body_quat = reference["body_quat"]
    ref_anchor_pos = ref_body_pos[:, self.anchor_index]
    ref_anchor_quat = ref_body_quat[:, self.anchor_index]

    root_pos_err = torch.sum(torch.square(ref_anchor_pos - anchor_pos), dim=-1)
    root_pos_term = torch.exp(-root_pos_err / cfg.root_pos_std**2)
    root_ori_err = _quat_error_squared(ref_anchor_quat, anchor_quat)
    root_ori_term = torch.exp(-root_ori_err / cfg.root_ori_std**2)

    anchor_pos_e = anchor_pos[:, None, :]
    num_bodies = body_pos.shape[1]
    delta_pos = torch.cat(
      [
        anchor_pos_e[..., :2].expand(-1, num_bodies, -1),
        ref_anchor_pos[:, None, 2:3].expand(-1, num_bodies, -1),
      ],
      dim=-1,
    )
    delta_ori = yaw_quat(
      quat_mul(anchor_quat[:, None, :], quat_inv(ref_anchor_quat[:, None, :]))
    ).expand(body_quat.shape[0], num_bodies, 4)
    ref_body_pos_rel = delta_pos + quat_apply(
      delta_ori, ref_body_pos - ref_anchor_pos[:, None, :]
    )
    ref_body_quat_rel = quat_mul(delta_ori, ref_body_quat)

    body_pos_err = torch.sum(torch.square(ref_body_pos_rel - body_pos), dim=-1).mean(
      dim=-1
    )
    body_pos_term = torch.exp(-body_pos_err / cfg.body_pos_std**2)
    body_ori_err = _quat_error_squared(ref_body_quat_rel, body_quat).mean(dim=-1)
    body_ori_term = torch.exp(-body_ori_err / cfg.body_ori_std**2)
    body_lin_vel_err = torch.sum(
      torch.square(reference["body_lin_vel"] - body_lin_vel), dim=-1
    ).mean(dim=-1)
    body_lin_vel_term = torch.exp(-body_lin_vel_err / cfg.body_lin_vel_std**2)
    body_ang_vel_err = torch.sum(
      torch.square(reference["body_ang_vel"] - body_ang_vel), dim=-1
    ).mean(dim=-1)
    body_ang_vel_term = torch.exp(-body_ang_vel_err / cfg.body_ang_vel_std**2)

    action_rate = torch.sum(torch.square(action - prev_action), dim=-1)
    lower = self.soft_joint_pos_limits[:, 0]
    upper = self.soft_joint_pos_limits[:, 1]
    joint_limit = torch.sum(
      torch.clamp(joint_pos - upper, min=0.0) + torch.clamp(lower - joint_pos, min=0.0),
      dim=-1,
    )

    reward = cfg.dt * (
      cfg.root_pos_weight * root_pos_term
      + cfg.root_ori_weight * root_ori_term
      + cfg.body_pos_weight * body_pos_term
      + cfg.body_ori_weight * body_ori_term
      + cfg.body_lin_vel_weight * body_lin_vel_term
      + cfg.body_ang_vel_weight * body_ang_vel_term
      + cfg.action_rate_weight * action_rate
      + cfg.joint_limit_weight * joint_limit
    )
    terms = {
      "root_pos": root_pos_term,
      "root_ori": root_ori_term,
      "body_pos": body_pos_term,
      "body_ori": body_ori_term,
      "body_lin_vel": body_lin_vel_term,
      "body_ang_vel": body_ang_vel_term,
      "action_rate": action_rate,
      "joint_limit": joint_limit,
    }
    return reward, terms

  def term_contributions(
    self, terms: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Weighted, dt-scaled contribution of every reward term."""
    cfg = self.cfg
    return {
      "root_pos": cfg.root_pos_weight * terms["root_pos"] * cfg.dt,
      "root_ori": cfg.root_ori_weight * terms["root_ori"] * cfg.dt,
      "body_pos": cfg.body_pos_weight * terms["body_pos"] * cfg.dt,
      "body_ori": cfg.body_ori_weight * terms["body_ori"] * cfg.dt,
      "body_lin_vel": cfg.body_lin_vel_weight * terms["body_lin_vel"] * cfg.dt,
      "body_ang_vel": cfg.body_ang_vel_weight * terms["body_ang_vel"] * cfg.dt,
      "action_rate": cfg.action_rate_weight * terms["action_rate"] * cfg.dt,
      "joint_limit": cfg.joint_limit_weight * terms["joint_limit"] * cfg.dt,
    }
