"""Official return-based reset sampler."""

import torch

from supermimic.algo.sampler import ReturnSampler, ReturnSamplerCfg


def _sampler(num_envs: int = 4) -> ReturnSampler:
  return ReturnSampler(
    torch.tensor([200, 120]),
    0.02,
    "cpu",
    num_envs=num_envs,
    cfg=ReturnSamplerCfg(horizon=5),
  )


def test_sample_stays_inside_clips():
  sampler = _sampler()
  clip, frame = sampler.sample(256)
  assert clip.shape == frame.shape == (256,)
  lengths = torch.tensor([200, 120])[clip]
  assert (frame >= 0).all() and (frame < lengths).all()


def test_distribution_sums_to_one():
  sampler = _sampler()
  dist = sampler._distribution()
  assert dist.shape == (sampler.bin_count,)
  assert torch.allclose(dist.sum(), torch.ones(()), atol=1e-5)
  assert (dist >= 0).all()


def test_record_updates_value_and_returns_distribution():
  sampler = _sampler()
  refreshed = False
  for _ in range(sampler.cfg.horizon):
    refreshed = sampler.record(
      frames=torch.zeros(4, dtype=torch.long),
      rewards=torch.ones(4),
      terminated=torch.zeros(4, dtype=torch.bool),
      clip_end=torch.zeros(4, dtype=torch.bool),
      timed_out=torch.zeros(4, dtype=torch.bool),
    )
  assert refreshed
  # The visited bin's value must move away from the v_max initialization.
  assert (sampler.value != sampler.v_max).any()
