"""Replay-window sampling must respect episode boundaries."""

import torch

from supermimic.algo.buffer import ReplayBuffer


def test_windows_do_not_cross_episode_boundaries():
  buffer = ReplayBuffer(2, 8, 3, 2, 1, "cpu")
  for t in range(6):
    buffer.add(
      state=torch.full((2, 3), float(t)),
      action=torch.full((2, 2), float(t)),
      reward=torch.zeros(2),
      done=torch.tensor([t == 2, False]),
      context=torch.zeros(2, 1),
      clip_id=torch.zeros(2, dtype=torch.long),
      time_step=torch.full((2,), t, dtype=torch.long),
    )
  env_ids, positions = buffer.sample_windows(512, length=3)
  states, actions, dones = buffer.gather_states(env_ids, positions, before=2)
  assert states.shape == (512, 3, 3)
  assert actions.shape == (512, 3, 2)
  assert dones.shape == (512, 3)
  assert not dones[:, :2].any()


def test_ring_buffer_wraps():
  buffer = ReplayBuffer(1, 4, 2, 1, 1, "cpu")
  for t in range(10):
    buffer.add(
      state=torch.full((1, 2), float(t)),
      action=torch.full((1, 1), float(t)),
      reward=torch.zeros(1),
      done=torch.zeros(1, dtype=torch.bool),
      context=torch.zeros(1, 1),
      clip_id=torch.zeros(1, dtype=torch.long),
      time_step=torch.full((1,), t, dtype=torch.long),
    )
  env_ids, positions = buffer.sample_windows(64, length=4)
  states, _, _ = buffer.gather_states(env_ids, positions, before=3)
  assert states.shape == (64, 4, 2)
  assert int(buffer.count.item()) == 4
