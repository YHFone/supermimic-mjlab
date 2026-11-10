"""The deployment-side input assembly must match the training features.

``g1_deploy`` rebuilds the observation and the goal in C++ from the IMU, the
encoders, and a motion clip.  These tests pin the reference implementation of
that contract against the training-side feature builders, so a layout drift
breaks here instead of on the robot.
"""

import numpy as np
import torch

from supermimic.algo.deploy_features import (
  DeployMotion,
  build_goal,
  build_observation,
  assemble_policy_input,
  goal_frame_dim,
)
from supermimic.mdp.features import GoalFeatures, ProprioFeatures


def _rotation(seed: int, count: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  axis = rng.normal(size=(count, 3))
  axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
  angle = rng.uniform(-np.pi, np.pi, size=(count, 1))
  return np.concatenate(
    [np.cos(0.5 * angle), np.sin(0.5 * angle) * axis], axis=1
  ).astype(np.float32)


def _motion(num_frames: int = 40, num_joints: int = 3, seed: int = 0) -> DeployMotion:
  rng = np.random.default_rng(seed)
  return DeployMotion.from_arrays(
    joint_pos=rng.normal(size=(num_frames, num_joints)),
    joint_vel=rng.normal(size=(num_frames, num_joints)) * 0.3,
    anchor_pos=rng.normal(size=(num_frames, 3)),
    anchor_quat=_rotation(seed + 1, num_frames),
    anchor_lin_vel=rng.normal(size=(num_frames, 3)),
    anchor_ang_vel=rng.normal(size=(num_frames, 3)),
    fps=50.0,
  )


def test_goal_matches_training_features():
  motion = _motion()
  horizon = 5
  index = 7
  default_joint_pos = np.linspace(-0.2, 0.2, motion.num_joints).astype(np.float32)
  root_quat = _rotation(99, 1)

  # Training side: (B, K + 1, bodies, ...) with frame t anchoring the window.
  window = np.arange(index, index + horizon + 1)
  reference = {
    "joint_pos": torch.as_tensor(motion.joint_pos[window])[None],
    "joint_vel": torch.as_tensor(motion.joint_vel[window])[None],
    "body_pos": torch.as_tensor(motion.anchor_pos[window])[None, :, None, :],
    "body_quat": torch.as_tensor(motion.anchor_quat[window])[None, :, None, :],
    "body_lin_vel": torch.as_tensor(motion.anchor_lin_vel[window])[None, :, None, :],
    "body_ang_vel": torch.as_tensor(motion.anchor_ang_vel[window])[None, :, None, :],
  }
  features = GoalFeatures(
    motion.num_joints,
    torch.as_tensor(default_joint_pos),
    anchor_index=0,
    horizon=horizon,
  )
  expected = features.from_reference(
    reference, torch.as_tensor(root_quat).reshape(1, 4)
  ).numpy().reshape(horizon, features.frame_dim)

  got = build_goal(
    motion, index, root_quat, default_joint_pos, horizon
  ).reshape(horizon, goal_frame_dim(motion.num_joints))

  assert got.shape == expected.shape == (horizon, 2 * motion.num_joints + 15)
  np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-5)


def test_goal_yaw_offset_only_moves_orientations():
  motion = _motion(seed=3)
  default_joint_pos = np.zeros(motion.num_joints, dtype=np.float32)
  root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
  yaw = np.array([np.cos(0.4), 0.0, 0.0, np.sin(0.4)], dtype=np.float32)

  plain = build_goal(motion, 4, root_quat, default_joint_pos, 3)
  rotated = build_goal(motion, 4, root_quat, default_joint_pos, 3, yaw_offset=yaw)
  frame = goal_frame_dim(motion.num_joints)
  pos_slice = slice(2 * motion.num_joints, 2 * motion.num_joints + 3)
  ori_slice = slice(2 * motion.num_joints + 3, 2 * motion.num_joints + 9)
  for k in range(3):
    base = k * frame
    np.testing.assert_allclose(plain[base + pos_slice.start : base + pos_slice.stop],
                               rotated[base + pos_slice.start : base + pos_slice.stop],
                               atol=1e-6)
    assert not np.allclose(plain[base + ori_slice.start : base + ori_slice.stop],
                           rotated[base + ori_slice.start : base + ori_slice.stop])


def test_goal_window_is_clamped_to_the_clip():
  motion = _motion(num_frames=10)
  default_joint_pos = np.zeros(motion.num_joints, dtype=np.float32)
  root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
  goal = build_goal(motion, motion.num_frames - 1, root_quat, default_joint_pos, 4)
  frame = goal_frame_dim(motion.num_joints)
  joint = goal.reshape(4, frame)[:, : motion.num_joints]
  # Every goal frame past the end repeats the last clip frame.
  np.testing.assert_allclose(joint, np.repeat(joint[:1], 4, axis=0), atol=1e-6)


def test_observation_matches_training_features():
  num_joints = 4
  rng = np.random.default_rng(5)
  joint_pos = rng.normal(size=num_joints).astype(np.float32)
  joint_vel = rng.normal(size=num_joints).astype(np.float32)
  default_joint_pos = np.linspace(-0.1, 0.1, num_joints).astype(np.float32)
  omega_body = np.array([0.1, -0.2, 0.3], dtype=np.float32)
  last_action = rng.normal(size=num_joints).astype(np.float32)

  # Identity root orientation: body-frame quantities are world-frame ones and
  # gravity points straight down.
  field = ProprioFeatures(num_joints, torch.as_tensor(default_joint_pos))
  state = torch.zeros(1, 13 + 2 * num_joints)
  state[0, 3] = 1.0
  state[0, 10:13] = torch.as_tensor(omega_body)
  state[0, 13 : 13 + num_joints] = torch.as_tensor(joint_pos)
  state[0, 13 + num_joints :] = torch.as_tensor(joint_vel)
  expected = field(state, torch.as_tensor(last_action)[None], None, add_noise=False)

  got = build_observation(
    joint_pos,
    joint_vel,
    default_joint_pos,
    omega_body,
    np.array([0.0, 0.0, -1.0], dtype=np.float32),
    last_action,
  )
  np.testing.assert_allclose(got, expected.numpy()[0], atol=1e-6)


def test_frames_roundtrip_and_input_order():
  motion = _motion(num_frames=12, seed=11)
  frames = motion.to_frames()
  assert frames.shape == (12, motion.frame_stride)
  again = DeployMotion.from_frames(frames, fps=motion.fps)
  np.testing.assert_allclose(again.joint_pos, motion.joint_pos)
  np.testing.assert_allclose(again.anchor_ang_vel, motion.anchor_ang_vel)

  history = np.arange(3 * motion.num_joints * 0 + 3 * motion.num_joints + 6)
  history = np.tile(history, (4, 1)).astype(np.float32)
  goal = build_goal(
    motion, 0, np.array([1.0, 0.0, 0.0, 0.0], np.float32),
    np.zeros(motion.num_joints, np.float32), 2,
  )
  policy_input = assemble_policy_input(history, goal)
  assert policy_input.shape == (history.size + goal.size,)
  np.testing.assert_allclose(policy_input[: history.size], history.reshape(-1))
  np.testing.assert_allclose(policy_input[history.size :], goal)
