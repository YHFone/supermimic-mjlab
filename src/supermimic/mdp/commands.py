"""Multi-motion reference command for SuperMimic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from ..algo.sampler import ReturnBasedSampler
from .features import GoalFeatures, make_state
from .motion_library import MotionLibrary, MotionLibraryCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionLibraryCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  motion_files: tuple[str, ...] = ()
  body_names: tuple[str, ...] = ()
  anchor_body_name: str = ""
  goal_frames: int = 10
  sampling_mode: Literal["start", "uniform", "return"] = "return"
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  bin_size_s: float = 1.0
  max_return: float = 10.0
  device: str | None = None
  max_frames: int | None = None

  def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
    return MotionLibraryCommand(self, env)


class MotionLibraryCommand(CommandTerm):
  cfg: MotionLibraryCommandCfg

  def __init__(self, cfg: MotionLibraryCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    if not cfg.motion_files:
      raise ValueError("MotionLibraryCommandCfg.motion_files must not be empty.")
    if not cfg.body_names:
      raise ValueError("MotionLibraryCommandCfg.body_names must not be empty.")
    self.robot: Entity = env.scene[cfg.entity_name]
    if cfg.anchor_body_name not in cfg.body_names:
      raise ValueError(f"Anchor body {cfg.anchor_body_name!r} is not in body_names.")
    self.anchor_index = cfg.body_names.index(cfg.anchor_body_name)
    self.robot_body_indexes, _ = self.robot.find_bodies(
      cfg.body_names, preserve_order=True
    )
    self.robot_anchor_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.joint_names = self.robot.joint_names
    self.num_joints = len(self.joint_names)

    self.library = MotionLibrary(
      MotionLibraryCfg(
        motion_files=cfg.motion_files,
        body_names=cfg.body_names,
        device=cfg.device or self.device,
        max_frames=cfg.max_frames,
      )
    )
    self.default_joint_pos = self.robot.data.default_joint_pos[0].clone()
    self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits
    self.goal_features = GoalFeatures(
      num_joints=self.num_joints,
      default_joint_pos=self.default_joint_pos,
      anchor_index=self.anchor_index,
      horizon=cfg.goal_frames,
    )

    self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.sampler = ReturnBasedSampler(
      self.library.clip_lengths_t,
      self.library.dt,
      self.device,
      bin_size_s=cfg.bin_size_s,
      max_return=cfg.max_return,
    )

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["clip_index"] = torch.zeros(self.num_envs, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

  @property
  def command(self) -> torch.Tensor:
    return self.goal()

  def goal(self) -> torch.Tensor:
    reference = self.library.gather(
      self.clip_ids, self.time_steps, self.cfg.goal_frames
    )
    root_pos = self.robot.data.root_link_pos_w - self._env.scene.env_origins
    root_quat = self.robot.data.root_link_quat_w
    return self.goal_features.from_reference(reference, root_pos, root_quat)

  def state(self) -> torch.Tensor:
    data = self.robot.data
    return make_state(
      data.root_link_pos_w - self._env.scene.env_origins,
      data.root_link_quat_w,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_pos,
      data.joint_vel,
    )

  def is_motion_end(self) -> torch.Tensor:
    return self.time_steps >= (self.library.clip_lengths_t[self.clip_ids] - 1)

  def _reference(self, horizon: int = 1) -> dict[str, torch.Tensor]:
    return self.library.gather(self.clip_ids, self.time_steps, horizon)

  @property
  def body_pos_w(self) -> torch.Tensor:
    ref = self._reference()
    return ref["body_pos"][:, 0] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._reference()["body_quat"][:, 0]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._reference()["body_lin_vel"][:, 0]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._reference()["body_ang_vel"][:, 0]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._reference()["body_pos"][:, 0, self.anchor_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._reference()["body_quat"][:, 0, self.anchor_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._reference()["body_lin_vel"][:, 0, self.anchor_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._reference()["body_ang_vel"][:, 0, self.anchor_index]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._reference()["joint_pos"][:, 0]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._reference()["joint_vel"][:, 0]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_index]

  def update_relative_body_poses(self) -> None:
    reference = self._reference()
    anchor_pos = reference["body_pos"][:, 0, self.anchor_index] + (
      self._env.scene.env_origins
    )
    anchor_quat = reference["body_quat"][:, 0, self.anchor_index]
    robot_anchor_pos = self.robot.data.body_link_pos_w[:, self.robot_anchor_index]
    robot_anchor_quat = self.robot.data.body_link_quat_w[:, self.robot_anchor_index]
    num_bodies = len(self.cfg.body_names)
    delta_pos_w = robot_anchor_pos[:, None, :].expand(-1, num_bodies, -1).clone()
    delta_pos_w[..., 2] = anchor_pos[:, None, 2].expand(-1, num_bodies)
    delta_ori_w = yaw_quat(
      quat_mul(
        robot_anchor_quat[:, None, :].expand(-1, num_bodies, -1),
        quat_inv(anchor_quat[:, None, :].expand(-1, num_bodies, -1)),
      )
    )
    self.body_quat_relative_w = quat_mul(delta_ori_w, reference["body_quat"][:, 0])
    ref_body_pos_w = (
      reference["body_pos"][:, 0] + self._env.scene.env_origins[:, None, :]
    )
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w,
      ref_body_pos_w - anchor_pos[:, None, :],
    )

  def _update_metrics(self) -> None:
    ref = self._reference()
    robot_pos = self.robot.data.body_link_pos_w[:, self.robot_anchor_index]
    robot_quat = self.robot.data.body_link_quat_w[:, self.robot_anchor_index]
    error = ref["body_pos"][:, 0, self.anchor_index] + self._env.scene.env_origins
    self.metrics["error_anchor_pos"] = torch.norm(error - robot_pos, dim=-1)
    q_err = quat_mul(ref["body_quat"][:, 0, self.anchor_index], quat_inv(robot_quat))
    self.metrics["error_anchor_rot"] = 2.0 * torch.acos(
      q_err[:, 0].abs().clamp(max=1.0)
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      ref["body_lin_vel"][:, 0, self.anchor_index]
      - self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_index],
      dim=-1,
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      ref["body_ang_vel"][:, 0, self.anchor_index]
      - self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_index],
      dim=-1,
    )
    self.metrics["clip_index"] = self.clip_ids.float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    if self.cfg.sampling_mode == "start":
      clip_ids = torch.randint(0, self.library.num_clips, (n,), device=self.device)
      times = torch.zeros(n, dtype=torch.long, device=self.device)
    elif self.cfg.sampling_mode == "uniform":
      clip_ids = torch.randint(0, self.library.num_clips, (n,), device=self.device)
      lengths = self.library.clip_lengths_t[clip_ids]
      times = (torch.rand(n, device=self.device) * lengths.float()).long()
    else:
      clip_ids, times = self.sampler.sample(n)

    reference = self.library.gather(clip_ids, times, 1)
    root_pos = reference["body_pos"][:, 0, 0] + self._env.scene.env_origins[env_ids]
    root_quat = reference["body_quat"][:, 0, 0]
    root_lin_vel = reference["body_lin_vel"][:, 0, 0]
    root_ang_vel = reference["body_ang_vel"][:, 0, 0]
    joint_pos = reference["joint_pos"][:, 0]
    joint_vel = reference["joint_vel"][:, 0]

    pose_ranges = torch.tensor(
      [
        self.cfg.pose_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    pose = sample_uniform(
      pose_ranges[:, 0], pose_ranges[:, 1], (n, 6), device=self.device
    )
    root_pos = root_pos + pose[:, :3]
    root_quat = quat_mul(
      quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), root_quat
    )
    vel_ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    vel = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=self.device)
    root_lin_vel = root_lin_vel + vel[:, :3]
    root_ang_vel = root_ang_vel + vel[:, 3:]
    joint_pos = joint_pos + sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      device=self.device,
    )
    limits = self.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_quat,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self.clip_ids[env_ids] = clip_ids
    self.time_steps[env_ids] = times

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    lengths = self.library.clip_lengths_t[self.clip_ids] - 1
    self.time_steps = torch.minimum(self.time_steps, lengths)
    self.update_relative_body_poses()

  def update_returns(
    self, clip_ids: torch.Tensor, times: torch.Tensor, returns: torch.Tensor
  ) -> None:
    if self.cfg.sampling_mode == "return":
      self.sampler.update_returns(clip_ids, times, returns)

  def reset_to_frame(self, env_ids: torch.Tensor, clip_id: int, frame: int) -> None:
    n = len(env_ids)
    clip_ids = torch.full((n,), clip_id, dtype=torch.long, device=self.device)
    times = torch.full((n,), frame, dtype=torch.long, device=self.device)
    reference = self.library.gather(clip_ids, times, 1)
    self._write_reference_state_to_sim(
      env_ids,
      reference["body_pos"][:, 0, 0] + self._env.scene.env_origins[env_ids],
      reference["body_quat"][:, 0, 0],
      reference["body_lin_vel"][:, 0, 0],
      reference["body_ang_vel"][:, 0, 0],
      reference["joint_pos"][:, 0],
      reference["joint_vel"][:, 0],
    )
    self.clip_ids[env_ids] = clip_ids
    self.time_steps[env_ids] = times
