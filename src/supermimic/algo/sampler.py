"""Return-based reset sampling (Appendix D.1 of the SuperMimic paper)."""

from __future__ import annotations

import torch


class ReturnBasedSampler:
  """Samples ``(clip, frame)`` start pairs from a return-weighted mixture."""

  def __init__(
    self,
    clip_lengths: torch.Tensor,
    frame_dt: float,
    device: str | torch.device,
    *,
    bin_size_s: float = 1.0,
    max_return: float = 10.0,
    uniform_ratio: float = 0.5,
    softmax_ratio: float = 0.4,
    past_ratio: float = 0.1,
    temperature: float = 25.0,
    ema: float = 0.1,
  ) -> None:
    self.device = torch.device(device)
    self.dt = frame_dt
    self.bin_size = max(1, round(bin_size_s / frame_dt))
    self.max_return = max_return
    self.uniform_ratio = uniform_ratio
    self.softmax_ratio = softmax_ratio
    self.past_ratio = past_ratio
    self.temperature = temperature
    self.ema = ema
    self.num_clips = int(clip_lengths.shape[0])
    self.max_bins = int(torch.ceil(clip_lengths.max() * frame_dt / bin_size_s).item())

    clip_ids, bin_starts, bin_ends = [], [], []
    lengths = clip_lengths.cpu().tolist()
    for clip, length in enumerate(lengths):
      for b in range((length + self.bin_size - 1) // self.bin_size):
        start = b * self.bin_size
        end = min(length, (b + 1) * self.bin_size)
        if end > start:
          clip_ids.append(clip)
          bin_starts.append(start)
          bin_ends.append(end)
    self.bin_clip = torch.tensor(clip_ids, dtype=torch.long, device=self.device)
    self.bin_start = torch.tensor(bin_starts, dtype=torch.long, device=self.device)
    self.bin_end = torch.tensor(bin_ends, dtype=torch.long, device=self.device)
    self.num_bins = self.bin_clip.shape[0]
    bins_per_clip = torch.tensor(
      [(length + self.bin_size - 1) // self.bin_size for length in lengths],
      dtype=torch.long,
      device=self.device,
    )
    self.bins_per_clip = bins_per_clip
    self.clip_bin_offset = torch.zeros_like(bins_per_clip)
    self.clip_bin_offset[1:] = torch.cumsum(bins_per_clip[:-1], dim=0)

    self.bin_returns = torch.zeros(self.num_bins, device=self.device)
    self.past_weights = torch.full(
      (self.num_bins,), 1.0 / self.num_bins, device=self.device
    )
    self.visits = torch.zeros(self.num_bins, device=self.device)
    self._probabilities = self._compute_probabilities()

  def _softmax_weights(self) -> torch.Tensor:
    scaled = -self.temperature * self.bin_returns / self.max_return
    return torch.softmax(scaled, dim=0)

  def _compute_probabilities(self) -> torch.Tensor:
    uniform = torch.full_like(self.bin_returns, 1.0 / self.num_bins)
    softmax = self._softmax_weights()
    past = self.past_weights / self.past_weights.sum().clamp_min(1e-12)
    probs = (
      self.uniform_ratio * uniform
      + self.softmax_ratio * softmax
      + self.past_ratio * past
    )
    return probs / probs.sum()

  def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = self._probabilities
    probs = torch.nan_to_num(probs, nan=0.0)
    if probs.sum() <= 0:
      probs = torch.full_like(probs, 1.0 / self.num_bins)
    bins = torch.multinomial(probs, n, replacement=True)
    span = (self.bin_end[bins] - self.bin_start[bins]).clamp_min(1)
    offsets = (torch.rand(n, device=self.device) * span.float()).long()
    frames = self.bin_start[bins] + offsets
    return self.bin_clip[bins], frames

  def update_returns(
    self,
    clip_ids: torch.Tensor,
    times: torch.Tensor,
    returns: torch.Tensor,
  ) -> None:
    bin_ids = self._bin_of(clip_ids, times)
    valid = bin_ids >= 0
    if not bool(valid.any()):
      return
    bin_ids = bin_ids[valid]
    values = returns[valid]
    self.visits.index_add_(0, bin_ids, torch.ones_like(values))
    self.bin_returns.index_copy_(
      0,
      bin_ids,
      self.bin_returns[bin_ids] * (1.0 - self.ema) + values * self.ema,
    )
    self.past_weights = 0.9 * self.past_weights + 0.1 * self._softmax_weights()
    self._probabilities = self._compute_probabilities()

  def _bin_of(self, clip_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    local_bin = times // self.bin_size
    bin_id = self.clip_bin_offset[clip_ids] + local_bin
    valid = local_bin < self.bins_per_clip[clip_ids]
    return torch.where(valid, bin_id, torch.full_like(bin_id, -1))
