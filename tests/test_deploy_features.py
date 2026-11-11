"""Deployment-side feature assembly (official layouts)."""

import numpy as np

from supermimic.algo import deploy_features as d

J = 29


def _motion(frames: int = 40) -> d.DeployMotion:
  return d.DeployMotion.from_arrays(
    joint_pos=np.zeros((frames, J)),
    joint_vel=np.zeros((frames, J)),
    anchor_pos=np.zeros((frames, 3)),
    anchor_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
    anchor_lin_vel=np.zeros((frames, 3)),
    anchor_ang_vel=np.zeros((frames, 3)),
    fps=50.0,
  )


def test_observation_order_is_official():
  obs = d.build_observation(
    np.zeros(J),
    np.arange(J, dtype=np.float32),
    np.zeros(J),
    np.array([1.0, 2.0, 3.0]),
    np.array([0.0, 0.0, -1.0]),
    np.full(J, 0.5),
  )
  assert obs.shape == (3 * J + 6,)
  assert np.allclose(obs[:3], [0.0, 0.0, -1.0])  # gravity first
  assert np.allclose(obs[3:6], [1.0, 2.0, 3.0])  # then base angular velocity
  assert np.allclose(obs[6 : 6 + J], 0.0)  # then q - q0
  assert np.allclose(obs[6 + J : 6 + 2 * J], np.arange(J))  # then dq
  assert np.allclose(obs[-J:], 0.5)  # last action


def test_goal_is_block_major_640():
  goal = d.build_goal(_motion(), index=5, root_quat=np.array([1.0, 0.0, 0.0, 0.0]))
  assert goal.shape == (1, 640)
  goal = goal[0]
  assert d.goal_dim(J) == 640
  # The orientation block sits at the end and starts as identity's first two
  # columns repeated for every frame.
  ori = goal[580:].reshape(10, 6)
  for frame in ori:
    # Training flattens ``matrix_from_quat(...)[..., :2]``: the first two ROWS
    # of R, i.e. [R00, R01, R02, R10, R11, R12] = [1, 0, 0, 0, 1, 0] for I.
    assert np.allclose(frame, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-6)
