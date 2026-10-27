"""Differentiable forward kinematics for an articulated entity.

The world model of SuperMimic predicts a compact robot state (root pose,
root velocity, joint positions and joint velocities).  To turn that state into
the quantities the motion-tracking reward and the policy observation need
(body poses and velocities), the model needs a differentiable kinematics
layer.  This module implements one for the kinematic trees used by mjlab
robots (hinge and slide joints, one joint per body).

The implementation mirrors MuJoCo's ``mj_kinematics``: a body frame is first
translated/rotated relative to its parent by ``body_pos``/``body_quat``, and
then the joint transformation is applied about the joint anchor ``jnt_pos``.
The velocity recursion is the derivative of the same transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, cast

import mujoco
import torch
from torch import nn

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(frozen=True)
class KinematicTree:
  """Static kinematic-tree tensors extracted from a MuJoCo model.

  All indices are local to the selected body list.  The root body (index 0)
  is driven by an externally supplied root pose; every other body has a parent
  in the same list.
  """

  body_names: tuple[str, ...]
  parent: torch.Tensor  # (N,) long, -1 for root.
  body_pos: torch.Tensor  # (N, 3) child frame in parent frame.
  body_quat: torch.Tensor  # (N, 4) child frame in parent frame.
  joint_index: torch.Tensor  # (N,) long, -1 if the body has no joint.
  joint_pos: torch.Tensor  # (N, 3) joint anchor in child frame.
  joint_axis: torch.Tensor  # (N, 3) joint axis in child frame.
  joint_type: torch.Tensor  # (N,) long, 0 = fixed, 1 = hinge, 2 = slide.


def _resolve_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
  """Resolve a model name, tolerating scene-level entity name prefixes."""
  obj_id = mujoco.mj_name2id(model, obj_type, name)
  if obj_id >= 0:
    return obj_id
  prefix = obj_type.name.lower()
  for i in range(model.nbody if obj_type == mujoco.mjtObj.mjOBJ_BODY else model.njnt):
    candidate = (
      model.body(i).name
      if obj_type == mujoco.mjtObj.mjOBJ_BODY
      else model.joint(i).name
    )
    if candidate.rsplit("/", 1)[-1] == name:
      return i
  raise ValueError(f"Unknown {prefix} {name!r}")


def build_kinematic_tree(
  model: mujoco.MjModel,
  body_names: Sequence[str],
  joint_names: Sequence[str],
) -> KinematicTree:
  """Extract the kinematic tree for ``body_names`` from a MuJoCo model."""
  body_ids = []
  for name in body_names:
    body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    body_ids.append(body_id)
  local_of_global = {gid: i for i, gid in enumerate(body_ids)}

  joint_index_of_name = {name: i for i, name in enumerate(joint_names)}
  n = len(body_ids)
  parent = torch.full((n,), -1, dtype=torch.long)
  body_pos = torch.zeros(n, 3)
  body_quat = torch.zeros(n, 4)
  body_quat[:, 0] = 1.0
  joint_index = torch.full((n,), -1, dtype=torch.long)
  joint_pos = torch.zeros(n, 3)
  joint_axis = torch.zeros(n, 3)
  joint_type = torch.zeros(n, dtype=torch.long)

  for i, body_id in enumerate(body_ids):
    if i > 0:
      parent_global = int(model.body_parentid[body_id])
      if parent_global not in local_of_global:
        raise ValueError(
          f"Body {body_names[i]!r} has parent "
          f"{model.body(parent_global).name!r} outside the requested body set."
        )
      parent[i] = local_of_global[parent_global]
    body_pos[i] = torch.from_numpy(model.body_pos[body_id].copy())
    body_quat[i] = torch.from_numpy(model.body_quat[body_id].copy())

    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    if jnt_num > 1:
      raise ValueError(
        f"Body {body_names[i]!r} has {jnt_num} joints; only one is supported."
      )
    if jnt_num == 0:
      continue
    if i == 0:
      continue
    joint_name = model.joint(jnt_adr).name
    short_name = joint_name.rsplit("/", 1)[-1]
    if short_name not in joint_index_of_name:
      raise ValueError(f"Joint {joint_name!r} is not in the state joint list.")
    joint_index[i] = joint_index_of_name[short_name]
    joint_pos[i] = torch.from_numpy(model.jnt_pos[jnt_adr].copy())
    joint_axis[i] = torch.from_numpy(model.jnt_axis[jnt_adr].copy())
    jnt_type = int(model.jnt_type[jnt_adr])
    if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
      joint_type[i] = 1
    elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
      joint_type[i] = 2
    else:
      raise ValueError(f"Unsupported joint type {jnt_type} on body {body_names[i]!r}.")

  return KinematicTree(
    body_names=tuple(body_names),
    parent=parent,
    body_pos=body_pos,
    body_quat=body_quat,
    joint_index=joint_index,
    joint_pos=joint_pos,
    joint_axis=joint_axis,
    joint_type=joint_type,
  )


class DifferentiableKinematics(nn.Module):
  """Batched differentiable FK for a kinematic tree."""

  def __init__(self, tree: KinematicTree) -> None:
    super().__init__()
    self.body_names = tree.body_names
    self.num_bodies = len(tree.body_names)
    self.register_buffer("parent", tree.parent.clone())
    self.register_buffer("body_pos", tree.body_pos.clone())
    self.register_buffer("body_quat", tree.body_quat.clone())
    self.register_buffer("joint_index", tree.joint_index.clone())
    self.register_buffer("joint_pos", tree.joint_pos.clone())
    self.register_buffer("joint_axis", tree.joint_axis.clone())
    self.register_buffer("joint_type", tree.joint_type.clone())

    n = self.num_bodies
    depth = torch.zeros(n, dtype=torch.long)
    for i in range(1, n):
      depth[i] = depth[int(tree.parent[i])] + 1
    max_depth = int(depth.max().item()) if n > 1 else 0
    order: list[torch.Tensor] = []
    offsets = [0]
    bodies_by_level: list[list[int]] = []
    parents_by_level: list[list[int]] = []
    for d in range(1, max_depth + 1):
      ids = (depth == d).nonzero(as_tuple=False).squeeze(-1)
      order.append(ids)
      bodies_by_level.append(ids.tolist())
      parents_by_level.append([int(tree.parent[i]) for i in ids.tolist()])
      offsets.append(offsets[-1] + int(ids.numel()))
    level_ids = torch.cat(order) if order else torch.empty(0, dtype=torch.long)
    self.register_buffer("level_ids", level_ids)
    self._level_offsets = offsets
    self._level_bodies = bodies_by_level
    self._level_parents = parents_by_level

  @classmethod
  def from_entity(
    cls,
    entity: Entity,
    body_names: Sequence[str],
    joint_names: Sequence[str],
  ) -> "DifferentiableKinematics":
    model = entity.spec.compile()
    return cls(build_kinematic_tree(model, body_names, joint_names))

  def forward(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat, _ = self._forward_impl(root_pos, root_quat, joint_pos)
    return pos, quat

  def forward_with_velocity(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos, quat, vel = self._forward_impl(
      root_pos,
      root_quat,
      joint_pos,
      root_lin_vel=root_lin_vel,
      root_ang_vel=root_ang_vel,
      joint_vel=joint_vel,
    )
    assert vel is not None
    return pos, quat, vel[..., :3], vel[..., 3:]

  def _forward_impl(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    *,
    root_lin_vel: torch.Tensor | None = None,
    root_ang_vel: torch.Tensor | None = None,
    joint_vel: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Vectorized body-by-body FK, one batched pass per tree level."""
    batch = root_pos.shape[0]
    dtype = root_pos.dtype
    compute_vel = root_lin_vel is not None
    if compute_vel:
      assert root_ang_vel is not None and joint_vel is not None
    jv = joint_vel if joint_vel is not None else torch.zeros_like(joint_pos)

    pos_list: list[torch.Tensor | None] = [None] * self.num_bodies
    quat_list: list[torch.Tensor | None] = [None] * self.num_bodies
    vel_list: list[torch.Tensor | None] = [None] * self.num_bodies
    pos_list[0] = root_pos
    quat_list[0] = root_quat
    if compute_vel:
      vel_list[0] = torch.cat([root_lin_vel, root_ang_vel], dim=-1)

    for level, (bodies, parents) in enumerate(
      zip(self._level_bodies, self._level_parents, strict=False)
    ):
      start = self._level_offsets[level]
      end = self._level_offsets[level + 1]
      idx = self.level_ids[start:end]
      body_pos = self.body_pos[idx].to(dtype).unsqueeze(0)
      body_quat = self.body_quat[idx].to(dtype).unsqueeze(0)
      joint_pos_local = self.joint_pos[idx].to(dtype).unsqueeze(0)
      joint_axis = self.joint_axis[idx].to(dtype).unsqueeze(0)
      j_type = self.joint_type[idx]
      j_idx = self.joint_index[idx].clamp(min=0)
      has_joint = (j_type > 0).to(dtype).unsqueeze(0).unsqueeze(-1)
      is_hinge = (j_type == 1).unsqueeze(0).unsqueeze(-1)
      is_slide = (j_type == 2).unsqueeze(0).unsqueeze(-1)

      parent_pos = torch.stack(
        [cast(torch.Tensor, pos_list[p]) for p in parents], dim=1
      )
      parent_quat = torch.stack(
        [cast(torch.Tensor, quat_list[p]) for p in parents], dim=1
      )
      pre_pos = parent_pos + quat_apply(parent_quat, body_pos.expand(batch, -1, 3))
      pre_quat = quat_mul(parent_quat, body_quat.expand(batch, -1, 4))

      q = joint_pos[:, j_idx] * has_joint.squeeze(-1)
      qd = jv[:, j_idx] * has_joint.squeeze(-1)
      axis_world = quat_apply(pre_quat, joint_axis.expand(batch, -1, 3))
      anchor = pre_pos + quat_apply(pre_quat, joint_pos_local.expand(batch, -1, 3))
      rot = quat_from_angle_axis(q, joint_axis.expand(batch, -1, 3))
      hinge_quat = quat_mul(pre_quat, rot)
      hinge_pos = anchor - quat_apply(hinge_quat, joint_pos_local.expand(batch, -1, 3))
      slide_pos = pre_pos + axis_world * q.unsqueeze(-1)
      new_quat = torch.where(is_hinge, hinge_quat, pre_quat)
      new_pos = torch.where(
        is_hinge, hinge_pos, torch.where(is_slide, slide_pos, pre_pos)
      )

      if compute_vel:
        parent_vel = torch.stack(
          [cast(torch.Tensor, vel_list[p]) for p in parents], dim=1
        )
        parent_lin = parent_vel[..., :3]
        parent_ang = parent_vel[..., 3:]
        pre_lin = parent_lin + torch.cross(parent_ang, pre_pos - parent_pos, dim=-1)
        joint_ang = axis_world * qd.unsqueeze(-1) * is_hinge.to(dtype)
        hinge_lin = (
          parent_lin
          + torch.cross(parent_ang, new_pos - parent_pos, dim=-1)
          + torch.cross(joint_ang, new_pos - anchor, dim=-1)
        )
        hinge_ang = parent_ang + joint_ang
        slide_lin = (
          parent_lin
          + torch.cross(parent_ang, new_pos - parent_pos, dim=-1)
          + axis_world * qd.unsqueeze(-1) * is_slide.to(dtype)
        )
        new_lin = torch.where(
          is_hinge, hinge_lin, torch.where(is_slide, slide_lin, pre_lin)
        )
        new_ang = torch.where(is_hinge, hinge_ang, parent_ang)
        new_vel = torch.cat([new_lin, new_ang], dim=-1)
      else:
        new_vel = None

      for j, body in enumerate(bodies):
        pos_list[body] = new_pos[:, j]
        quat_list[body] = new_quat[:, j]
        if compute_vel:
          assert new_vel is not None
          vel_list[body] = new_vel[:, j]

    body_pos_out = torch.stack([cast(torch.Tensor, t) for t in pos_list], dim=1)
    body_quat_out = torch.stack([cast(torch.Tensor, t) for t in quat_list], dim=1)
    body_vel_out = (
      torch.stack([cast(torch.Tensor, t) for t in vel_list], dim=1)
      if compute_vel
      else None
    )
    return body_pos_out, body_quat_out, body_vel_out


def project_gravity(root_quat: torch.Tensor) -> torch.Tensor:
  """Project the world gravity direction into the root frame."""
  gravity_w = torch.tensor([0.0, 0.0, -1.0], device=root_quat.device)
  gravity_w = gravity_w.expand(root_quat.shape[0], 3)
  return quat_apply(quat_inv(root_quat), gravity_w)
