"""Return-based reset sampling, ported from the official SuperMimic runner.

The motion timeline is cut into bins.  ``V[b]`` measures the discounted return
the policy earns from bin ``b`` on its own rollouts, and turns it into the reset
distribution

    p(b) = explore/N + (1 - explore - keep) * softmax(-steepness * V[b] / v_max)
                  + keep * effort[b] / Σ effort

where ``effort`` is the running sum of the *squared* focus each bin received, so
a bin that stays hard keeps getting practice while one that stops being hard
fades out.  Statistics update every ``horizon`` recorded steps; a bin's earlier
samples count at most ``memory`` times against each update, so ``V`` follows the
current policy instead of averaging over the whole run.

This replaces the previous ``uniform 0.5 / softmax 0.4 / past 0.1`` mixture with
an EMA of the return, which is *not* the official rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReturnSamplerCfg:
  gamma: float = 0.99
  horizon: int = 20
  """Steps per update; each env's last ``horizon`` steps become return samples."""
  memory: int = 256
  """Most samples a bin's ``V`` remembers."""
  steepness: float = 25.0
  explore: float = 0.5
  keep: float = 0.1
  bin_size_s: float = 1.0
  max_step_reward: float = 0.1


class ReturnSampler:
  """Samples ``(clip, frame)`` resets from the official return-weighted mixture."""

  def __init__(
    self,
    clip_lengths: torch.Tensor,
    frame_dt: float,
    device: str | torch.device,
    *,
    num_envs: int,
    cfg: ReturnSamplerCfg | None = None,
  ) -> None:
    self.cfg = cfg or ReturnSamplerCfg()
    self.device = torch.device(device)
    self.dt = frame_dt
    lengths = clip_lengths.cpu().tolist()
    self.clip_lengths = list(lengths)
    self.frame_count = int(sum(lengths))
    self.bin_count = max(
      1, int(round(self.frame_count * frame_dt / self.cfg.bin_size_s))
    )
    self.bin_frames = max(1, int(round(self.cfg.bin_size_s / frame_dt)))
    self.v_max = self.cfg.max_step_reward / (1.0 - self.cfg.gamma)

    # Global frame index -> clip id, for turning a sampled frame into a reset.
    self.frame_clip = torch.cat(
      [
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(lengths)
      ]
    ).to(self.device)
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.long), 0)
    self.clip_offsets = offsets.to(self.device)

    self.value = torch.full((self.bin_count,), self.v_max, device=self.device)
    self.count = torch.zeros(self.bin_count, device=self.device)
    self.effort = torch.zeros(self.bin_count, dtype=torch.float64, device=self.device)
    shape = (num_envs, self.cfg.horizon)
    self._frames = torch.zeros(shape, dtype=torch.long, device=self.device)
    self._rewards = torch.zeros(shape, device=self.device)
    self._terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._clip_end = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._timed_out = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self._step = 0
    self._distribution_t = self._uniform()

  # ------------------------------------------------------------------
  # sampling
  # ------------------------------------------------------------------
  def _uniform(self) -> torch.Tensor:
    return torch.full(
      (self.bin_count,), 1.0 / self.bin_count, device=self.device
    )

  def _bins(self, frames: torch.Tensor) -> torch.Tensor:
    return (frames * self.bin_count // self.frame_count).clamp_max(self.bin_count - 1)

  def _distribution(self) -> torch.Tensor:
    cfg = self.cfg
    focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=0)
    self.effort += focus.double().square()
    keep = (self.effort / self.effort.sum().clamp_min(1e-12)).float()
    explore = cfg.explore / self.bin_count
    return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep

  def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.nan_to_num(self._distribution_t, nan=0.0)
    if float(probs.sum()) <= 0:
      probs = self._uniform()
    probs = probs / probs.sum()
    bins = torch.multinomial(probs, n, replacement=True)
    # Official mapping: a continuous position inside the bin, rescaled to the
    # global timeline (mjlab_research/.../commands.py:231-240).
    u = torch.rand(n, device=self.device)
    frames = (
      (bins.float() + u) / self.bin_count * (self.frame_count - 1)
    ).long().clamp(max=self.frame_count - 1)
    clip_ids = self.frame_clip[frames]
    local = frames - self.clip_offsets[clip_ids]
    return clip_ids, local

  # ------------------------------------------------------------------
  # statistics
  # ------------------------------------------------------------------
  def record(
    self,
    frames: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    clip_end: torch.Tensor,
    timed_out: torch.Tensor,
  ) -> bool:
    """Buffer one env step; returns True when the distribution was refreshed."""
    col = self._step
    self._frames[:, col] = frames
    self._rewards[:, col] = rewards
    self._terminated[:, col] = terminated
    self._clip_end[:, col] = clip_end
    self._timed_out[:, col] = timed_out
    self._step += 1
    if self._step < self.cfg.horizon:
      return False
    self._step = 0
    self._update()
    self._distribution_t = self._distribution()
    return True

  def _dones(self) -> torch.Tensor:
    return self._terminated | self._clip_end | self._timed_out

  def _returns(self) -> torch.Tensor:
    gamma = self.cfg.gamma
    rewards = self._rewards
    next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
    next_value = torch.where(
      self._terminated,
      torch.zeros_like(self.value[self._bins(next_frames)]),
      torch.where(
        self._clip_end,
        self.value.mean().expand_as(self.value[self._bins(next_frames)]),
        self.value[self._bins(next_frames)],
      ),
    )
    dones = self._dones()
    returns = torch.empty_like(rewards)
    next_return = next_value[:, -1]
    for t in reversed(range(self.cfg.horizon)):
      next_return = torch.where(dones[:, t], next_value[:, t], next_return)
      returns[:, t] = rewards[:, t] + gamma * next_return
      next_return = returns[:, t]
    return returns

  def _update(self) -> None:
    returns = self._returns()
    dones = self._dones()
    num_envs, horizon = dones.shape
    episode = torch.cumsum(dones.long(), dim=1) - dones.long()
    envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
    key = (envs * horizon + episode) * self.bin_count + self._bins(self._frames)
    keys, inverse, visits = torch.unique(
      key.reshape(-1), return_inverse=True, return_counts=True
    )
    samples = torch.zeros(keys.numel(), device=returns.device)
    samples.scatter_add_(0, inverse, returns.reshape(-1))
    samples /= visits
    bins = keys % self.bin_count
    new = torch.zeros_like(self.count).scatter_add_(
      0, bins, torch.ones_like(samples)
    )
    total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
    self.count = self.count.clamp(max=self.cfg.memory) + new
    self.value = torch.where(
      new > 0, self.value + (total - new * self.value) / self.count, self.value
    )
