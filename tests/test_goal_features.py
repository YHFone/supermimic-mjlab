"""Official goal window: block-major, root-relative, stride 5."""

import torch

from supermimic.mdp.features import GoalFeatures


def _reference(frames: int, num_joints: int = 3) -> dict[str, torch.Tensor]:
  quat = torch.zeros(1, frames, 1, 4)
  quat[..., 0] = 1.0
  pos = torch.zeros(1, frames, 1, 3)
  pos[0, :, 0, 0] = torch.arange(frames, dtype=torch.float32)
  return {
    "body_pos": pos,
    "body_quat": quat,
    "body_lin_vel": torch.zeros(1, frames, 1, 3),
    "body_ang_vel": torch.zeros(1, frames, 1, 3),
    "joint_pos": torch.arange(frames, dtype=torch.float32)[None, :, None].expand(
      1, frames, num_joints
    ),
    "joint_vel": torch.full((1, frames, num_joints), 0.5),
  }


def test_goal_is_block_major_with_absolute_joints():
  feats = GoalFeatures(num_joints=3, anchor_index=0, horizon=2)
  ref = _reference(2)
  root_pos = torch.zeros(1, 3)
  root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  wm, actor = feats.from_reference(ref, root_pos, root_quat)
  assert wm.shape == (1, 2 * 2 * 3 + 2 * 3 + 2 * 6)
  assert actor.shape == (1, 2 * 2 * 3 + 2 * 6)
  # Content is [qj, vj] per frame, frame-major: qj_f0, vj_f0, qj_f1, vj_f1.
  assert torch.allclose(actor[0, :3], torch.zeros(3))
  assert torch.allclose(actor[0, 3:6], torch.full((3,), 0.5))
  assert torch.allclose(actor[0, 6:9], torch.ones(3))
  # rel_pos follows content only in the world-model goal: x = 0 then 1.
  rel_pos = wm[0, 12:18].reshape(2, 3)
  assert torch.allclose(rel_pos[:, 0], torch.tensor([0.0, 1.0]))


def test_goal_is_root_relative():
  feats = GoalFeatures(num_joints=1, anchor_index=0, horizon=1)
  ref = _reference(1, num_joints=1)
  ref["body_pos"][0, 0, 0] = torch.tensor([5.0, 0.0, 0.0])
  wm_near, _ = feats.from_reference(
    ref, torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]])
  )
  wm_far, _ = feats.from_reference(
    ref, torch.tensor([[5.0, 0.0, 0.0]]), torch.tensor([[1.0, 0, 0, 0]])
  )
  # Same world anchor, different robot root: the relative position changes.
  assert torch.allclose(wm_near[0, 2:5], torch.tensor([5.0, 0.0, 0.0]))
  assert torch.allclose(wm_far[0, 2:5], torch.tensor([-0.0, 0.0, 0.0]))
