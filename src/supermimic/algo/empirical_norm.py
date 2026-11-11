"""Empirical normalization matching the official (rsl_rl) semantics.

The official SuperMimic implementation normalizes the world-model input with
``rsl_rl.modules.EmpiricalNormalization``:

* ``x_norm = (x - mean) / (std + eps)`` with ``eps = 1e-2`` **outside** the
  square root,
* statistics update only while the module is in ``train()`` mode — at eval
  time the normalizer is a fixed affine map, so exported policies see exactly
  the statistics they were trained with,
* ``inverse`` restores physical units.

The previous local ``RunningNorm`` used ``eps = 1e-5`` **inside** the square
root and normalized in both modes, which changes both the scale (by up to
~1e-2 relative) and the train/eval behaviour of the exported policy.
"""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNorm(nn.Module):
  """Running mean/std normalizer with the official train-only update rule."""

  def __init__(self, shape: int, *, eps: float = 1e-2) -> None:
    super().__init__()
    self.eps = eps
    self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
    self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
    self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  @property
  def mean(self) -> torch.Tensor:
    return self._mean.squeeze(0).clone()

  @property
  def std(self) -> torch.Tensor:
    return self._std.squeeze(0).clone()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self._mean) / (self._std + self.eps)

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    """Update the running statistics; a no-op in eval mode."""
    if not self.training:
      return
    x = x.detach().float()
    if x.numel() == 0:
      return
    count_x = x.shape[0]
    self.count += count_x
    rate = count_x / self.count
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    delta_mean = mean_x - self._mean
    self._mean += rate * delta_mean
    self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
    self._std.copy_(torch.sqrt(self._var))

  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    return y * (self._std + self.eps) + self._mean
