"""GenCoord state representation, ported from the official SuperMimic code.

The official implementation (``HybridRobotics/supermimic``,
``src/supermimic/representations.py``) trains its world model in a
*generalized-coordinate* space rather than on the raw 71-dimensional simulator
state:

===== ==========================================================================
full  ``[root_pos(3), root_quat(4), v_w(3), ω_w(3), qj(J), vj(J)]``  = 13 + 2J
aux   ``[root_pos(3), root_quat(4)]``                                 = 7
rep   ``[z(1), g_b(3), v_b(3), ω_b(3), qj(J), vj(J)]``                = 10 + 2J
pred  ``[Δv_b(3), Δω_b(3), Δvj(J)]``                                  = 6 + J
===== ==========================================================================

``z`` is the root height, ``g_b`` the gravity direction in the root frame and
``v_b`` / ``ω_b`` the root twist in the root frame.  The representation drops
the horizontal position (translation invariance) and the quaternion (the root
pose lives in :attr:`aux`), and the network predicts **acceleration-level
deltas** which :meth:`GenCoordRepresentation.apply_prediction` integrates
semi-implicitly with ``substep_coeff = (d + 1) / (2 d)``.

The *full* state keeps this package's existing column order
``[p, q, v, ω, qj, vj]`` (joints last) instead of the official
``[p, q, qj, v, ω, vj]``; every learned interface (rep, aux, pred, goal) is
byte-identical to the official one, and the reordering keeps the local FK and
reward code unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_mul,
)

DOWN_WORLD = (0.0, 0.0, -1.0)


def quat_integrate(quat: torch.Tensor, angular_velocity: torch.Tensor, dt: float) -> torch.Tensor:
  """Right-multiply ``quat`` by the rotation ``ω dt`` (world frame).

  Args:
    quat: ``(B, 4)`` unit quaternion in wxyz.
    angular_velocity: ``(B, 3)`` world-frame angular velocity.
    dt: integration step.
  """
  angle = torch.linalg.vector_norm(angular_velocity, dim=-1, keepdim=True) * dt
  half = 0.5 * angle
  small = angle < 1e-6
  safe = angle.clamp_min(1e-6)
  # sin(θ/2) / |ω| = sin(θ/2) * dt / angle
  scale = torch.where(small, 0.5 * dt - angle.pow(2) * dt / 48.0, torch.sin(half) / safe)
  delta = torch.cat([torch.cos(half), angular_velocity * scale], dim=-1)
  out = quat_mul(quat, delta)
  return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(1e-8)


def decompose_rep_state(rep: torch.Tensor, num_joints: int) -> tuple[torch.Tensor, ...]:
  """Split ``rep`` into ``(z, g_b, v_b, w_b, qj, vj)``."""
  z = rep[..., 0:1]
  g_b = rep[..., 1:4]
  v_b = rep[..., 4:7]
  w_b = rep[..., 7:10]
  qj = rep[..., 10 : 10 + num_joints]
  vj = rep[..., 10 + num_joints :]
  return z, g_b, v_b, w_b, qj, vj


@dataclass(frozen=True)
class GenCoordCfg:
  num_joints: int = 29
  dt: float = 0.02
  decimation: int = 4
  """Simulator substeps per control step; sets ``substep_coeff``."""


class GenCoordRepresentation(nn.Module):
  """Bidirectional map between the full state and the official GenCoord space."""

  def __init__(
    self,
    cfg: GenCoordCfg,
    *,
    default_joint_pos: torch.Tensor | None = None,
  ) -> None:
    super().__init__()
    self.cfg = cfg
    self.num_joints = cfg.num_joints
    self.full_state_dim = 13 + 2 * cfg.num_joints
    self.aux_state_dim = 7
    self.rep_state_dim = 10 + 2 * cfg.num_joints
    self.prediction_dim = 6 + cfg.num_joints
    self.substep_coeff = (cfg.decimation + 1) / (2.0 * cfg.decimation)
    if default_joint_pos is None:
      self.register_buffer("default_joint_pos", torch.zeros(cfg.num_joints))
    else:
      self.register_buffer("default_joint_pos", default_joint_pos.clone().float())
    self.register_buffer(
      "down_world", torch.tensor(DOWN_WORLD, dtype=torch.float32), persistent=False
    )

  # ------------------------------------------------------------------
  # full -> rep / aux
  # ------------------------------------------------------------------
  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    """``[root_pos(3), root_quat(4)]``."""
    return torch.cat([full_state[..., 0:3], full_state[..., 3:7]], dim=-1)

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    """``[z, g_b, v_b, ω_b, qj, vj]``; horizontal position and quat are dropped."""
    p = full_state[..., 0:3]
    q = full_state[..., 3:7]
    v_w = full_state[..., 7:10]
    w_w = full_state[..., 10:13]
    qj = full_state[..., 13 : 13 + self.num_joints]
    vj = full_state[..., 13 + self.num_joints :]
    down = self.down_world.to(full_state.dtype).expand_as(v_w)
    g_b = quat_apply_inverse(q, down)
    v_b = quat_apply_inverse(q, v_w)
    w_b = quat_apply_inverse(q, w_w)
    return torch.cat([p[..., 2:3], g_b, v_b, w_b, qj, vj], dim=-1)

  # ------------------------------------------------------------------
  # rep + pred -> next rep / aux
  # ------------------------------------------------------------------
  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit integration of an acceleration-level prediction.

    Pose updates use ``v_b + c Δv_b`` (``c = substep_coeff``) to match how the
    simulator composes its substeps; velocities advance by the full delta.  The
    resulting twist is re-expressed in the *new* root frame, exactly as the
    official implementation does.
    """
    dt = self.cfg.dt if dt is None else dt
    c = self.substep_coeff
    n = self.num_joints
    _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state, n)
    dv_b = prediction[..., 0:3]
    dw_b = prediction[..., 3:6]
    dvj = prediction[..., 6 : 6 + n]

    next_v_cur = v_b + dv_b
    next_w_cur = w_b + dw_b
    next_vj = vj + dvj
    next_qj = qj + (vj + c * dvj) * dt

    root_pos = aux_state[..., 0:3]
    root_quat = aux_state[..., 3:7]
    v_pose = quat_apply(root_quat, v_b + c * dv_b)
    w_pose = quat_apply(root_quat, w_b + c * dw_b)
    next_root_pos = root_pos + v_pose * dt
    next_root_quat = quat_integrate(root_quat, w_pose, dt)
    next_root_quat = next_root_quat / torch.linalg.vector_norm(
      next_root_quat, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    v_global = quat_apply(root_quat, next_v_cur)
    w_global = quat_apply(root_quat, next_w_cur)
    next_v_b = quat_apply_inverse(next_root_quat, v_global)
    next_w_b = quat_apply_inverse(next_root_quat, w_global)
    down = self.down_world.to(rep_state.dtype).expand_as(next_v_b)
    next_g_b = quat_apply_inverse(next_root_quat, down)

    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    next_rep = torch.cat(
      [next_root_pos[..., 2:3], next_g_b, next_v_b, next_w_b, next_qj, next_vj],
      dim=-1,
    )
    return next_aux, next_rep

  # ------------------------------------------------------------------
  # rep + aux -> full (for FK / reward)
  # ------------------------------------------------------------------
  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state, self.num_joints)
    root_pos = aux_state[..., 0:3]
    root_quat = aux_state[..., 3:7]
    root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
    v_global = quat_apply(root_quat, v_b)
    w_global = quat_apply(root_quat, w_b)
    return torch.cat([root_pos, root_quat, v_global, w_global, qj, vj], dim=-1)

  # ------------------------------------------------------------------
  # loss / observation helpers
  # ------------------------------------------------------------------
  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """Channel groups scored by the world-model loss (official ``LOSS_WEIGHT_CONFIG``).

    ``z``, ``g_b`` and the root twist are *derived* from the integrated pose, so
    they are not scored directly.  ``root_ori`` uses the rotation matrix scaled
    by ``sqrt(1/2)``: a plain squared difference is then
    ``0.5 ||ΔR||_F² = 2(1 − cos θ) = θ² + O(θ⁴)``, i.e. a nominal in radians.
    """
    _, _, _, _, joint_pos, joint_vel = decompose_rep_state(rep_state, self.num_joints)
    rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    return {
      "root_pos": aux_state[..., 0:3],
      "root_ori": rot * (0.5**0.5),
      "joint_pos": joint_pos,
      "joint_vel": joint_vel,
    }

  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    """``[qj, vj]`` in joint order."""
    return rep_state[..., 10:]
