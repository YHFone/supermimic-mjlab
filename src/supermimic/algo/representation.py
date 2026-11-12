"""GenCoord state representation, ported from the official SuperMimic code.

The official implementation (``HybridRobotics/supermimic``,
``src/supermimic/representations.py``) trains its world model in a
*generalized-coordinate* space rather than on the raw 71-dimensional simulator
state:

===== ==========================================================================
full  ``[root_pos(3), root_quat(4), qj(J), v_w(3), ω_w(3), vj(J)]``  = 13 + 2J
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

The *full* state uses the official column order ``[p, q, qj, v, ω, vj]``
(``utils/gencoord_helpers.py:3``); ``decompose_full_state`` is the single
place that knows the offsets.
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

PROPRIO_NOISE_HALF_WIDTHS: dict[str, float] = {
  "projected_gravity": 0.05,
  "base_ang_vel": 0.2,
  "joint_pos": 0.01,
  "joint_vel": 0.5,
  "actions": 0.0,
}
"""Per-term uniform proprio noise half-widths (official ``representations.py:114``)."""


def quat_integrate(quat: torch.Tensor, angular_velocity: torch.Tensor, dt: float) -> torch.Tensor:
  """Left-multiply ``quat`` by the world-frame rotation ``ω dt``.

  ``angular_velocity`` is expressed in the world frame (the caller passes
  ``quat_apply(root_quat, ω_b + c Δω_b)``), so the delta quaternion acts on the
  left: ``q_next = δ ⊗ q``.  Right-multiplying instead integrates about the
  *body* axes and silently tilts the root whenever yaw and the rotation axis
  are not collinear (up to ~50° after a 32-step imagination rollout).

  Args:
    quat: ``(B, 4)`` unit quaternion in wxyz.
    angular_velocity: ``(B, 3)`` world-frame angular velocity.
    dt: integration step.
  """
  omega = angular_velocity * dt
  norm_sq = (omega * omega).sum(dim=-1, keepdim=True)
  norm_safe = torch.sqrt(norm_sq + 1e-8)
  half = 0.5 * norm_safe
  axis = omega / norm_safe
  delta = torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)
  out = quat_mul(delta, quat)
  return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True)


def full_state_offsets(num_joints: int) -> tuple[int, int, int, int]:
  """Start offsets of ``qj``, root linear/angular velocity and ``vj``.

  Official full state (``utils/gencoord_helpers.py:24-43``):
  ``[root_pos(3), root_quat(4), qj(J), root_lin_vel(3), root_ang_vel(3), vj(J)]``.
  """
  idx_qj = 7
  idx_lin = idx_qj + num_joints
  idx_ang = idx_lin + 3
  idx_vj = idx_ang + 3
  return idx_qj, idx_lin, idx_ang, idx_vj


def decompose_full_state(
  full_state: torch.Tensor, num_joints: int
) -> tuple[torch.Tensor, ...]:
  """Split the official full state into ``(p, q, qj, v_w, ω_w, vj)``."""
  idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(num_joints)
  return (
    full_state[..., 0:3],
    full_state[..., 3:7],
    full_state[..., idx_qj:idx_lin],
    full_state[..., idx_lin:idx_ang],
    full_state[..., idx_ang:idx_vj],
    full_state[..., idx_vj:],
  )


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
    p, q, qj, v_w, w_w, vj = decompose_full_state(full_state, self.num_joints)
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
  def proprio_term_widths(self, num_actions: int) -> list[int]:
    """Widths of the proprio terms, in layout order."""
    return [3, 3, self.num_joints, self.num_joints, num_actions]

  def proprio_obs_dim(self, num_actions: int) -> int:
    return sum(self.proprio_term_widths(num_actions))

  def proprio_noise_bounds(
    self, num_actions: int, device: torch.device | str
  ) -> torch.Tensor:
    """Per-dim uniform-noise half-widths matching the proprio layout."""
    widths = self.proprio_term_widths(num_actions)
    halves = PROPRIO_NOISE_HALF_WIDTHS.values()
    return torch.cat(
      [torch.full((w,), h, device=device) for w, h in zip(widths, halves)]
    )

  def add_proprio_noise(
    self, obs: torch.Tensor, num_actions: int | None = None
  ) -> torch.Tensor:
    """Per-dim uniform proprio jitter (official ``add_proprio_noise``).

    The official implementation owns the noise here (the env observation group
    runs with ``enable_corruption=False``) so that the real-sim history, the
    imagination rollout and the exporter share one spec.
    """
    actions = self.num_joints if num_actions is None else num_actions
    bounds = self.proprio_noise_bounds(actions, obs.device).to(obs.dtype)
    return obs + ((torch.rand_like(obs) * 2 - 1) * bounds).detach()

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state, self.num_joints)
    root_pos = aux_state[..., 0:3]
    root_quat = aux_state[..., 3:7]
    root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
    v_global = quat_apply(root_quat, v_b)
    w_global = quat_apply(root_quat, w_b)
    # Official order: [p, q, qj, v_w, ω_w, vj].
    return torch.cat([root_pos, root_quat, qj, v_global, w_global, vj], dim=-1)

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
