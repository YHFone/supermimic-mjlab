"""State and context extraction for SuperMimic.

The world model operates on a compact state

``s = [root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel]``

with root quantities in the world frame and positions relative to the
environment origin (so environments on different terrain patches share a
distribution).  The per-episode simulator context ``xi`` collects the
domain-randomization parameters that the world model conditions on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import mujoco
import torch

from .kinematics import DifferentiableKinematics, build_kinematic_tree

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _model_name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
  obj_id = mujoco.mj_name2id(model, obj_type, name)
  if obj_id >= 0:
    return obj_id
  count = model.nbody if obj_type == mujoco.mjtObj.mjOBJ_BODY else model.ngeom
  for i in range(count):
    candidate = (
      model.body(i).name if obj_type == mujoco.mjtObj.mjOBJ_BODY else model.geom(i).name
    )
    if candidate.rsplit("/", 1)[-1] == name:
      return i
  raise ValueError(f"Unknown model object {name!r}")


@dataclass
class SuperMimicStateSpec:
  """Static tensors shared by the state extractor and the learned models."""

  num_envs: int
  num_joints: int
  default_joint_pos: torch.Tensor  # (J,)
  soft_joint_pos_limits: torch.Tensor  # (J, 2)
  tracked_body_indices: torch.Tensor  # (B,) indices into the tracked body list
  tracked_body_names: tuple[str, ...]
  robot_body_names: tuple[str, ...]
  joint_names: tuple[str, ...]
  device: torch.device

  @property
  def state_dim(self) -> int:
    return 13 + 2 * self.num_joints

  @property
  def action_dim(self) -> int:
    return self.num_joints

  @property
  def context_dim(self) -> int:
    return 1 + self.num_joints + 3

  @classmethod
  def from_env(
    cls, env: ManagerBasedRlEnv, tracked_body_names: Sequence[str]
  ) -> "SuperMimicStateSpec":
    robot = env.scene["robot"]
    kinematic_names, tracked_indices = _ancestor_body_names(
      env.sim.mj_model, tracked_body_names
    )
    return cls(
      num_envs=env.num_envs,
      num_joints=len(robot.joint_names),
      default_joint_pos=robot.data.default_joint_pos[0].clone(),
      soft_joint_pos_limits=robot.data.soft_joint_pos_limits[0].clone(),
      tracked_body_indices=torch.tensor(
        tracked_indices, dtype=torch.long, device=env.device
      ),
      tracked_body_names=kinematic_names,
      robot_body_names=tuple(robot.body_names),
      joint_names=tuple(robot.joint_names),
      device=torch.device(env.device),
    )


def _ancestor_body_names(
  model: mujoco.MjModel, tracked_body_names: Sequence[str]
) -> tuple[tuple[str, ...], list[int]]:
  """Tracked bodies plus every ancestor needed to reach the root.

  Tracked lists such as the G1's skip intermediate links (hip pitch/yaw,
  ankle pitch).  Their joints still affect the tracked poses, so the FK tree
  must include them even though the reward only scores the tracked subset.
  """
  tracked_ids = [
    _model_name_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in tracked_body_names
  ]
  selected = set(tracked_ids)
  for body_id in tracked_ids:
    parent = int(model.body_parentid[body_id])
    while parent > 0:
      selected.add(parent)
      parent = int(model.body_parentid[parent])
  ordered = sorted(selected)
  position = {body_id: i for i, body_id in enumerate(ordered)}
  names = tuple(model.body(body_id).name.rsplit("/", 1)[-1] for body_id in ordered)
  return names, [position[body_id] for body_id in tracked_ids]


class StateExtractor:
  """Reads the world-model state and simulator context from an mjlab env."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    spec: SuperMimicStateSpec,
    *,
    torso_body_name: str = "torso_link",
    left_foot_geom_pattern: str = r"^left_foot[1-7]_collision$",
    right_foot_geom_pattern: str = r"^right_foot[1-7]_collision$",
  ) -> None:
    self.env = env
    self.spec = spec
    self.robot = env.scene["robot"]
    model = env.sim.mj_model
    self.torso_body_id = _model_name_id(
      model, mujoco.mjtObj.mjOBJ_BODY, torso_body_name
    )
    left_ids = [
      i
      for i in range(model.ngeom)
      if re.search(left_foot_geom_pattern, _short_name(model.geom(i).name))
    ]
    right_ids = [
      i
      for i in range(model.ngeom)
      if re.search(right_foot_geom_pattern, _short_name(model.geom(i).name))
    ]
    self.left_foot_geom_ids = torch.tensor(
      left_ids, dtype=torch.long, device=env.device
    )
    self.right_foot_geom_ids = torch.tensor(
      right_ids, dtype=torch.long, device=env.device
    )
    self.default_body_ipos = env.sim.get_default_field("body_ipos")

  @property
  def origins(self) -> torch.Tensor:
    return self.env.scene.env_origins

  def state(self) -> torch.Tensor:
    data = self.robot.data
    return torch.cat(
      [
        data.root_link_pos_w - self.origins,
        data.root_link_quat_w,
        data.root_link_lin_vel_w,
        data.root_link_ang_vel_w,
        data.joint_pos,
        data.joint_vel,
      ],
      dim=-1,
    )

  def encoder_bias(self) -> torch.Tensor:
    return self.robot.data.encoder_bias

  def context(self) -> torch.Tensor:
    model = self.env.sim.model
    left = model.geom_friction[:, self.left_foot_geom_ids, 0].mean(dim=-1)
    right = model.geom_friction[:, self.right_foot_geom_ids, 0].mean(dim=-1)
    friction = 0.5 * (left + right)
    com_offset = (
      model.body_ipos[:, self.torso_body_id]
      - self.default_body_ipos[self.torso_body_id][None, :]
    )
    return torch.cat([friction[:, None], self.encoder_bias(), com_offset], dim=-1)


def _short_name(name: str) -> str:
  return name.rsplit("/", 1)[-1]


def build_kinematics(
  env: ManagerBasedRlEnv,
  spec: SuperMimicStateSpec,
) -> DifferentiableKinematics:
  """Build a differentiable kinematic tree for the simulated robot."""
  model = env.sim.mj_model
  return DifferentiableKinematics(
    build_kinematic_tree(
      model,
      body_names=spec.tracked_body_names,
      joint_names=spec.joint_names,
    )
  ).to(spec.device)
