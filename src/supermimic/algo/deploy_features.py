"""Deployment-side assembly of the SuperMimic policy input.

The training observation and goal are built from simulator state
(:mod:`supermimic.mdp.features`).  At deployment time the same numbers have to be
produced from the IMU, the joint encoders, and a reference motion clip, in plain
``float32``.  This module is the reference implementation of that contract: the
``g1_deploy`` C++ adapter mirrors it operation for operation, and
``tests/test_deploy_features.py`` pins it against the training features.

Layouts (``J`` joints, ``H`` history frames, ``K`` goal frames)::

    obs  = H * (3 J + 6)      q - q0 | dq | omega_b | g_b | a_{t-1}
    goal = K * (2 J + 15)     q_ref - q0 | dq_ref | anchor_pos | anchor_ori |
                              anchor_lin_vel | anchor_ang_vel

Clip frame ``t`` anchors the goal: the position terms of goal frame ``k`` are
relative to the anchor pose at ``t`` and expressed in its frame, and goal frame
``k`` refers to clip frame ``t + 1 + k``.  Anchor orientations are relative to the
robot's current root orientation, as in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANCHOR_POS_DIM = 3
ANCHOR_ORI_DIM = 6
ANCHOR_VEL_DIM = 6


def goal_frame_dim(num_joints: int) -> int:
  """Numbers per goal frame: ``2 J + 3 + 6 + 3 + 3``."""
  return 2 * num_joints + ANCHOR_POS_DIM + ANCHOR_ORI_DIM + ANCHOR_VEL_DIM


def obs_frame_dim(num_joints: int) -> int:
  """Numbers per observation frame: ``3 J + 6``."""
  return 3 * num_joints + 6


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
  """Rotation matrix of a ``(..., 4)`` ``wxyz`` quaternion."""
  quat = np.asarray(quat, dtype=np.float64)
  quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
  w, x, y, z = np.moveaxis(quat, -1, 0)
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ],
    -2,
  )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product of ``(..., 4)`` ``wxyz`` quaternions."""
  aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
  bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    -1,
  )


def quat_inv(quat: np.ndarray) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float64)
  return quat * np.array([1.0, -1.0, -1.0, -1.0])


@dataclass(frozen=True)
class DeployMotion:
  """A reference clip in the ``g1_deploy`` playback layout."""

  joint_pos: np.ndarray  # (F, J)
  joint_vel: np.ndarray  # (F, J)
  anchor_pos: np.ndarray  # (F, 3)
  anchor_quat: np.ndarray  # (F, 4) wxyz
  anchor_lin_vel: np.ndarray  # (F, 3)
  anchor_ang_vel: np.ndarray  # (F, 3)
  fps: float

  @property
  def num_frames(self) -> int:
    return int(self.joint_pos.shape[0])

  @property
  def num_joints(self) -> int:
    return int(self.joint_pos.shape[1])

  @property
  def frame_stride(self) -> int:
    return 2 * self.num_joints + 13

  def to_frames(self) -> np.ndarray:
    """``(F, 2 J + 13)`` flat float32 array in the ``.bin`` layout."""
    return np.concatenate(
      [
        self.joint_pos,
        self.joint_vel,
        self.anchor_pos,
        self.anchor_quat,
        self.anchor_lin_vel,
        self.anchor_ang_vel,
      ],
      axis=1,
    ).astype("<f4")

  @classmethod
  def from_arrays(
    cls,
    joint_pos,
    joint_vel,
    anchor_pos,
    anchor_quat,
    anchor_lin_vel,
    anchor_ang_vel,
    fps: float,
  ) -> "DeployMotion":
    return cls(
      joint_pos=np.asarray(joint_pos, dtype=np.float32),
      joint_vel=np.asarray(joint_vel, dtype=np.float32),
      anchor_pos=np.asarray(anchor_pos, dtype=np.float32),
      anchor_quat=np.asarray(anchor_quat, dtype=np.float32),
      anchor_lin_vel=np.asarray(anchor_lin_vel, dtype=np.float32),
      anchor_ang_vel=np.asarray(anchor_ang_vel, dtype=np.float32),
      fps=float(fps),
    )

  @classmethod
  def from_frames(cls, frames: np.ndarray, fps: float) -> "DeployMotion":
    """Rebuild from a ``(F, 2 J + 13)`` playback array."""
    frames = np.asarray(frames, dtype=np.float32)
    num_joints = (frames.shape[1] - 13) // 2
    cuts = [num_joints, 2 * num_joints, 2 * num_joints + 3, 2 * num_joints + 7]
    cuts.append(2 * num_joints + 10)
    parts = np.split(frames, cuts, axis=1)
    return cls.from_arrays(*parts, fps=fps)

  @classmethod
  def from_npz(cls, path: str | Path, anchor: str = "pelvis") -> "DeployMotion":
    with np.load(path, allow_pickle=False) as data:
      body_names = [str(name) for name in data["body_names"]]
      index = body_names.index(anchor)
      return cls.from_arrays(
        joint_pos=data["joint_pos"],
        joint_vel=data["joint_vel"],
        anchor_pos=data["body_pos_w"][:, index],
        anchor_quat=data["body_quat_w"][:, index],
        anchor_lin_vel=data["body_lin_vel_w"][:, index],
        anchor_ang_vel=data["body_ang_vel_w"][:, index],
        fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
      )


def build_observation(
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  default_joint_pos: np.ndarray,
  omega_body: np.ndarray,
  gravity_body: np.ndarray,
  last_action: np.ndarray,
) -> np.ndarray:
  """One 93-dimensional proprioceptive frame (``3 J + 6``)."""
  joint_pos = np.asarray(joint_pos, dtype=np.float32)
  return np.concatenate(
    [
      joint_pos - np.asarray(default_joint_pos, dtype=np.float32),
      np.asarray(joint_vel, dtype=np.float32),
      np.asarray(omega_body, dtype=np.float32),
      np.asarray(gravity_body, dtype=np.float32),
      np.asarray(last_action, dtype=np.float32),
    ]
  ).astype(np.float32)


def build_goal(
  motion: DeployMotion,
  index: int,
  root_quat: np.ndarray,
  default_joint_pos: np.ndarray,
  horizon: int,
  yaw_offset: np.ndarray | None = None,
) -> np.ndarray:
  """``horizon * (2 J + 15)`` goal for clip frame ``index``."""
  num_joints = motion.num_joints
  default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
  window = np.clip(
    np.arange(index + 1, index + 1 + horizon), 0, motion.num_frames - 1
  )
  anchor_quat = motion.anchor_quat.astype(np.float64)
  origin_quat = anchor_quat[min(max(index, 0), motion.num_frames - 1)]

  origin_rot = quat_to_matrix(origin_quat)
  delta = motion.anchor_pos[window] - motion.anchor_pos[index]
  pos_b = delta @ origin_rot  # R(origin)^T (p_k - p_t)

  ref_quat = anchor_quat[window]
  if yaw_offset is not None:
    ref_quat = quat_mul(np.asarray(yaw_offset, dtype=np.float64), ref_quat)
  rel_quat = quat_mul(quat_inv(np.asarray(root_quat, dtype=np.float64)), ref_quat)
  # First two columns of the rotation matrix, as ``matrix_from_quat(...)[..., :2]``.
  ori = quat_to_matrix(rel_quat)[:, :, :2].reshape(len(window), 6)

  frame_rot = quat_to_matrix(anchor_quat[window])
  lin_b = np.einsum("kji,kj->ki", frame_rot, motion.anchor_lin_vel[window])
  ang_b = np.einsum("kji,kj->ki", frame_rot, motion.anchor_ang_vel[window])

  goal = np.concatenate(
    [
      motion.joint_pos[window] - default_joint_pos,
      motion.joint_vel[window],
      pos_b,
      ori,
      lin_b,
      ang_b,
    ],
    axis=1,
  )
  return goal.reshape(1, horizon * goal_frame_dim(num_joints)).astype(np.float32)[0]


def assemble_policy_input(history: np.ndarray, goal: np.ndarray) -> np.ndarray:
  """Flatten ``(H, 3 J + 6)`` history and the goal into the ONNX input."""
  return np.concatenate(
    [np.asarray(history, dtype=np.float32).reshape(-1), np.asarray(goal, np.float32)]
  )
