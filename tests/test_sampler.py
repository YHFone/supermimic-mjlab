"""Return-based sampler: one EMA step per bin, at the bin start."""

from types import SimpleNamespace

import torch

from supermimic.algo.sampler import ReturnBasedSampler
from supermimic.algo.trainer import SuperMimicTrainer


def test_update_returns_aggregates_duplicate_bins():
  sampler = ReturnBasedSampler(torch.tensor([200]), 0.02, "cpu", bin_size_s=1.0)
  # Two frames of the same 1 s bin, and the second bin for good measure.
  clip_ids = torch.zeros(3, dtype=torch.long)
  times = torch.tensor([0, 40, 50])
  values = torch.tensor([1.0, 3.0, 5.0])
  sampler.update_returns(clip_ids, times, values)
  assert torch.allclose(sampler.bin_returns[0], torch.tensor(0.2))  # 0.1 * mean(1, 3)
  assert torch.allclose(sampler.bin_returns[1], torch.tensor(0.5))  # 0.1 * 5
  assert torch.allclose(sampler.visits[:2], torch.tensor([2.0, 1.0]))


def test_finish_episodes_uses_bin_frames_not_seconds():
  length = 120
  gamma = 0.99
  sampler = ReturnBasedSampler(torch.tensor([length]), 0.02, "cpu", bin_size_s=1.0)
  command = SimpleNamespace(sampler=sampler, update_returns=sampler.update_returns)
  stub = SimpleNamespace(
    cfg=SimpleNamespace(gamma=gamma),
    command=command,
    library=SimpleNamespace(dt=0.02),
    device=torch.device("cpu"),
    episode_rewards=torch.zeros(1, length + 2),
    episode_len=torch.tensor([length]),
    episode_start_clip=torch.tensor([0]),
    episode_start_time=torch.tensor([0]),
  )
  rewards = torch.full((length,), 0.08)
  stub.episode_rewards[0, :length] = rewards
  SuperMimicTrainer._finish_episodes(stub, torch.tensor([0]))

  values = torch.zeros(length)
  disc = 0.0
  for t in range(length - 1, -1, -1):
    disc = float(rewards[t]) + gamma * disc
    values[t] = disc
  expected = torch.stack([values[0], values[50], values[100]])
  assert torch.allclose(sampler.visits, torch.tensor([1.0, 1.0, 1.0]))
  assert torch.allclose(sampler.bin_returns[:3], 0.1 * expected, atol=1e-6)
