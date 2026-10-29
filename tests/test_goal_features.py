"""Goal-encoding checks, including the reference velocity terms."""

import torch

from supermimic.mdp.features import GoalFeatures


def _reference(num_frames: int, num_joints: int = 3) -> dict[str, torch.Tensor]:
  batch, bodies = 1, 1
  quat = torch.zeros(batch, num_frames, bodies, 4)
  quat[..., 0] = 1.0
  return {
    "body_pos": torch.zeros(batch, num_frames, bodies, 3),
    "body_quat": quat,
    "body_lin_vel": torch.zeros(batch, num_frames, bodies, 3),
    "body_ang_vel": torch.zeros(batch, num_frames, bodies, 3),
    "joint_pos": torch.zeros(batch, num_frames, num_joints),
    "joint_vel": torch.zeros(batch, num_frames, num_joints),
  }


def test_goal_layout_includes_reference_velocities():
  horizon = 2
  features = GoalFeatures(3, torch.zeros(3), anchor_index=0, horizon=horizon)
  assert features.frame_dim == 2 * 3 + 3 + 6 + 3 + 3

  reference = _reference(horizon + 1)
  # Frame 0 only anchors the position; frames 1..K are goal frames.
  reference["body_lin_vel"][:, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
  reference["body_ang_vel"][:, 1, 0] = torch.tensor([4.0, 5.0, 6.0])
  reference["body_lin_vel"][:, 2, 0] = torch.tensor([7.0, 8.0, 9.0])
  reference["body_ang_vel"][:, 2, 0] = torch.tensor([10.0, 11.0, 12.0])
  root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

  goal = features.from_reference(reference, root_quat)
  goal = goal.reshape(1, horizon, features.frame_dim)
  assert torch.allclose(goal[0, 0, -6:-3], torch.tensor([1.0, 2.0, 3.0]))
  assert torch.allclose(goal[0, 0, -3:], torch.tensor([4.0, 5.0, 6.0]))
  assert torch.allclose(goal[0, 1, -6:-3], torch.tensor([7.0, 8.0, 9.0]))
  assert torch.allclose(goal[0, 1, -3:], torch.tensor([10.0, 11.0, 12.0]))


def test_reference_velocities_are_local_to_the_reference_frame():
  horizon = 1
  features = GoalFeatures(3, torch.zeros(3), anchor_index=0, horizon=horizon)

  reference = _reference(horizon + 1)
  # 180 deg yaw: the local x/y axes flip, z is unchanged.
  reference["body_quat"][:, 1, 0] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
  reference["body_lin_vel"][:, 1, 0] = torch.tensor([1.0, 2.0, 3.0])
  reference["body_ang_vel"][:, 1, 0] = torch.tensor([4.0, 5.0, 6.0])
  root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

  goal = features.from_reference(reference, root_quat)
  goal = goal.reshape(1, horizon, features.frame_dim)
  assert torch.allclose(goal[0, 0, -6:-3], torch.tensor([-1.0, -2.0, 3.0]))
  assert torch.allclose(goal[0, 0, -3:], torch.tensor([-4.0, -5.0, 6.0]))
