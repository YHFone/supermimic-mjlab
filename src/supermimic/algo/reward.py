"""Differentiable motion-tracking reward and termination.

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
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .kinematics import DifferentiableKinematics

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


def _quat_error_squared(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Squared geodesic angle ``||log(R1^T R2)||^2`` (paper Eq. A.2).

  ``dot`` is clamped away from one so that the ``acos`` gradient stays finite
  when two orientations coincide in an imagined rollout.
  """
  dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0 - 1e-6)
  return 4.0 * torch.acos(dot) ** 2


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
    body_pos, body_quat, body_lin_vel, body_ang_vel = (
      self.kinematics.forward_with_velocity(
        state[:, 0:3],
        state[:, 3:7],
        state[:, 7:10],
        state[:, 10:13],
        state[:, 13 : 13 + num_joints],
        state[:, 13 + num_joints :],
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
    joint_pos = state[:, 13 : 13 + num_joints]
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
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
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


@dataclass(frozen=True)
class TrackingTerminationCfg:
  """Termination thresholds of Appendix A."""

  pos_threshold: float = 0.25
  ori_threshold: float = 0.8
  soft_tau: float = 0.02


class TrackingTermination(nn.Module):
  """Anchor height/tilt termination with a differentiable probability."""

  def __init__(
    self,
    kinematics: DifferentiableKinematics,
    tracked_body_indices: torch.Tensor,
    anchor_index: int,
    cfg: TrackingTerminationCfg | None = None,
  ) -> None:
    super().__init__()
    self.kinematics = kinematics
    self.cfg = cfg or TrackingTerminationCfg()
    self.register_buffer("tracked_body_indices", tracked_body_indices.clone())
    self.anchor_index = anchor_index
    self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, -1.0]).unsqueeze(0))

  def forward(
    self, state: torch.Tensor, reference: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    num_joints = (state.shape[-1] - 13) // 2
    body_pos, body_quat = self.kinematics(
      state[:, 0:3],
      state[:, 3:7],
      state[:, 13 : 13 + num_joints],
    )
    ids = self.tracked_body_indices
    anchor_pos = body_pos[:, ids][:, self.anchor_index]
    anchor_quat = body_quat[:, ids][:, self.anchor_index]
    return self.forward_from_anchor(anchor_pos, anchor_quat, reference)

  def forward_from_anchor(
    self,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    reference: dict[str, torch.Tensor],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary termination from an already-computed root anchor pose."""
    ref_anchor_pos = reference["body_pos"][:, self.anchor_index]
    ref_anchor_quat = reference["body_quat"][:, self.anchor_index]

    pos_err = torch.abs(anchor_pos[:, 2] - ref_anchor_pos[:, 2])
    gravity_w = self.gravity_w.expand(anchor_pos.shape[0], 3)
    ref_gravity_z = quat_apply_inverse(ref_anchor_quat, gravity_w)[:, 2]
    robot_gravity_z = quat_apply_inverse(anchor_quat, gravity_w)[:, 2]
    ori_err = torch.abs(ref_gravity_z - robot_gravity_z)

    hard = (pos_err > self.cfg.pos_threshold) | (ori_err > self.cfg.ori_threshold)
    # The paper's d_t is a discrete event: detach the 0/1 probability so the
    # survival product c_t in Eq. 2 carries no gradient through the test.
    return hard, hard.to(anchor_pos.dtype)
