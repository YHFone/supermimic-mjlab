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


def test_terminal_states_are_stored_for_the_termination_head():
  buffer = ReplayBuffer(2, 8, 3, 2, 1, "cpu")
  next_state = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
  buffer.add(
    state=torch.zeros(2, 3),
    action=torch.zeros(2, 2),
    reward=torch.zeros(2),
    done=torch.tensor([True, False]),
    context=torch.zeros(2, 1),
    clip_id=torch.tensor([4, 5]),
    time_step=torch.tensor([7, 7]),
    terminated=torch.tensor([True, False]),
    next_state=next_state,
  )
  assert buffer.terminal_count == 1
  assert torch.allclose(buffer.terminal_states[0], next_state[0])
  assert int(buffer.terminal_clip_ids[0]) == 4
  assert int(buffer.terminal_time_steps[0]) == 8
  sample = buffer.sample_terminal(8)
  assert sample["state"].shape == (8, 3)
  assert torch.allclose(sample["state"], next_state[0].expand(8, -1))


def test_windows_are_dropped_instead_of_crossing_a_reset():
  cap = 8
  buffer = ReplayBuffer(1, cap, 3, 2, 1, "cpu")
  for t in range(cap):
    buffer.add(
      state=torch.full((1, 3), float(t)),
      action=torch.zeros(1, 2),
      reward=torch.zeros(1),
      done=torch.tensor([t % 2 == 1]),
      context=torch.zeros(1, 1),
      clip_id=torch.zeros(1, dtype=torch.long),
      time_step=torch.full((1,), t, dtype=torch.long),
    )
  env_ids, positions = buffer.sample_windows(32, length=3)
  assert env_ids.numel() == 0
  assert positions.numel() == 0
