"""Unitree G1 SuperMimic environment configuration."""

from __future__ import annotations

from typing import Sequence

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import body_com_offset, encoder_bias, geom_friction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from supermimic.mdp import MotionLibraryCommandCfg, motion_end
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

TRACKED_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def unitree_g1_supermimic_env_cfg(
  motion_files: Sequence[str],
  num_envs: int = 4096,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 SuperMimic tracking environment."""
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      entities={"robot": get_g1_robot_cfg()},
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    episode_length_s=10.0,
    auto_reset=False,
  )

  joint_pos_action = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=G1_ACTION_SCALE,
    use_default_offset=True,
  )

  motion_cmd = MotionLibraryCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    motion_files=tuple(motion_files),
    body_names=TRACKED_BODY_NAMES,
    anchor_body_name="pelvis",
    goal_frames=10,
    sampling_mode="start" if play else "return",
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range=VELOCITY_RANGE,
    joint_position_range=(-0.1, 0.1),
  )

  cfg.actions = {"joint_pos": joint_pos_action}
  cfg.commands = {"motion": motion_cmd}

  actuator_terms = {
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(func=env_mdp.joint_vel_rel),
    "base_ang_vel": ObservationTermCfg(func=env_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=env_mdp.projected_gravity),
    "last_action": ObservationTermCfg(func=env_mdp.last_action),
  }
  goal_terms = {
    "motion_goal": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "motion"},
    ),
  }
  cfg.observations = {
    "proprio": ObservationGroupCfg(
      terms=actuator_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
      flatten_history_dim=True,
    ),
    "goal": ObservationGroupCfg(
      terms=goal_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.events = {
    "base_com": EventTermCfg(
      mode="reset",
      func=body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="reset",
      func=encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="reset",
      func=geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
  }

  cfg.rewards = {}

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "motion_end": TerminationTermCfg(
      func=motion_end,
      time_out=True,
      params={"command_name": "motion"},
    ),
  }
  return cfg
