"""State and simulator-context extraction for SuperMimic.

The world model operates on the full state

    s = [root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel]

with root quantities in the world frame and positions relative to the
environment origin.  The per-episode simulator context ``xi`` follows the
official implementation (``supermimic/sim_context.py``):

* column order ``[foot_friction(1), torso_com_offset(3), encoder_bias(J)]``;
* each column normalized by its domain-randomization range:
  ``(raw - mean) / scale`` with ``mean = (lo + hi) / 2`` and
  ``scale = (hi - lo) / 2``; a half-width below 1e-3 means the parameter is not
  actually randomized, so ``scale = 1`` (avoids amplifying float noise);
* values are read at the **mjModel global entity ids** the DR events write to.

The previous local layout was ``[friction, bias, com]`` with raw (unnormalized)
values, which puts the joint biases and the CoM offset in swapped columns and
shifts every context column by a per-parameter constant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import mujoco
import torch
from torch import nn

from supermimic.mdp.features import make_state

from .kinematics import DifferentiableKinematics, build_kinematic_tree

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEGENERATE_HALF_WIDTH = 1e-3
"""DR ranges narrower than this are treated as constant (scale stays 1)."""


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


def _mean_scale(lo: float, hi: float) -> tuple[float, float]:
  half = (hi - lo) / 2.0
  return (lo + hi) / 2.0, half if half > _DEGENERATE_HALF_WIDTH else 1.0


@dataclass
class SuperMimicStateSpec:
  """Static tensors shared by the state extractor and the learned models."""

  num_envs: int
  num_joints: int
  default_joint_pos: torch.Tensor
  soft_joint_pos_limits: torch.Tensor
  tracked_body_indices: torch.Tensor
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
    return 1 + 3 + self.num_joints

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
  """Tracked bodies plus every ancestor needed to reach the root."""
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


class SimContext(nn.Module):
  """Per-episode domain-randomization context ``ξ``, normalized per column."""

  def __init__(
    self,
    *,
    num_joints: int,
    torso_body_id: int,
    foot_geom_id: int,
    friction_range: tuple[float, float],
    com_ranges: dict[int, tuple[float, float]],
    bias_range: tuple[float, float],
  ) -> None:
    super().__init__()
    self.num_joints = num_joints
    self.torso_body_id = int(torso_body_id)
    self.foot_geom_id = int(foot_geom_id)
    stats = [
      _mean_scale(*friction_range),
      *(_mean_scale(*com_ranges[i]) for i in range(3)),
      *(_mean_scale(*bias_range) for _ in range(num_joints)),
    ]
    self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
    self.register_buffer("scale", torch.tensor([s for _, s in stats]))
    self.dim = 1 + 3 + num_joints

  def raw(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    model = env.sim.model
    friction = model.geom_friction[:, self.foot_geom_id, 0]
    com = model.body_ipos[:, self.torso_body_id, :]
    bias = env.scene["robot"].data.encoder_bias
    return torch.cat([friction[:, None], com, bias], dim=-1)

  def forward(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    raw = self.raw(env).to(self.mean.dtype)
    return (raw - self.mean) / self.scale

  @classmethod
  def from_env(
    cls, env: ManagerBasedRlEnv, *, torso_body_name: str = "torso_link"
  ) -> "SimContext":
    events = env.cfg.events
    # Evaluation and play configs drop the domain-randomization events.  The
    # context must still be normalized with the *training* ranges, otherwise the
    # world model sees a differently scaled xi than it was trained on.  Fall back
    # to the official defaults instead of failing.
    missing = [n for n in ("foot_friction", "base_com", "encoder_bias") if n not in events]
    if missing:
      print(
        f"[WARN] SimContext: missing DR events {missing}; using the training "
        "ranges and reading the (constant) model values.",
        flush=True,
      )
    model = env.sim.mj_model
    robot = env.scene["robot"]
    torso_id = _model_name_id(model, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)

    if "foot_friction" in events:
      friction_ranges = events["foot_friction"].params["ranges"]
      if isinstance(friction_ranges, dict):
        friction_range = tuple(friction_ranges[0])
      else:
        friction_range = tuple(friction_ranges)
      pattern = events["foot_friction"].params["asset_cfg"].geom_names
    else:
      friction_range = (0.3, 1.2)
      pattern = r"^(left|right)_foot[1-7]_collision$"

    if "base_com" in events:
      com_params = events["base_com"].params
      com_ranges = {i: tuple(com_params["ranges"][i]) for i in range(3)}
    else:
      com_ranges = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
    bias_range = (
      tuple(events["encoder_bias"].params["bias_range"])
      if "encoder_bias" in events
      else (-0.01, 0.01)
    )
    regex = re.compile(pattern) if isinstance(pattern, str) else None
    foot_geom = None
    for i in range(model.ngeom):
      short = model.geom(i).name.rsplit("/", 1)[-1]
      if (regex and regex.match(short)) or (
        not regex and short in set(pattern or ())
      ):
        foot_geom = i
        break
    if foot_geom is None:
      raise RuntimeError(f"SimContext: no foot geom matches {pattern!r}")

    return cls(
      num_joints=len(robot.joint_names),
      torso_body_id=torso_id,
      foot_geom_id=foot_geom,
      friction_range=friction_range,
      com_ranges=com_ranges,
      bias_range=bias_range,
    ).to(env.device)


def context_encoder_bias(context: torch.Tensor, num_joints: int) -> torch.Tensor:
  """Encoder calibration offsets inside ``ξ = [friction, com(3), bias(J)]``."""
  return context[..., 4 : 4 + num_joints]


class StateExtractor:
  """Reads the full world-model state and the simulator context from an mjlab env."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    spec: SuperMimicStateSpec,
    *,
    torso_body_name: str = "torso_link",
  ) -> None:
    self.env = env
    self.spec = spec
    self.robot = env.scene["robot"]
    self.sim_context = SimContext.from_env(env, torso_body_name=torso_body_name)

  @property
  def origins(self) -> torch.Tensor:
    return self.env.scene.env_origins

  def state(self) -> torch.Tensor:
    """Full state in the official column order.

    Delegates to :func:`make_state` so the layout has exactly one definition; a
    mismatch here silently scrambles every consumer (representation, FK, reward).
    """
    data = self.robot.data
    return make_state(
      data.root_link_pos_w - self.origins,
      data.root_link_quat_w,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_pos,
      data.joint_vel,
    )

  def context(self) -> torch.Tensor:
    """Normalized simulator context ``ξ``, in the official column order."""
    return self.sim_context(self.env)


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
