"""Reference-motion storage for SuperMimic.

A :class:`MotionLibrary` loads a set of mjlab motion ``.npz`` files (the format
produced by :mod:`supermimic.scripts.preprocess`) and concatenates
them so that reference frames can be gathered for many environments at once.
Only the bodies that are actually tracked are kept in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionLibraryCfg:
  motion_files: tuple[str, ...]
  body_names: tuple[str, ...]
  device: str = "cuda:0"
  max_frames: int | None = None


class MotionLibrary:
  """Concatenated, GPU-resident reference-motion library."""

  def __init__(self, cfg: MotionLibraryCfg) -> None:
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.dt = 1.0 / 50.0
    self.clip_names: list[str] = []
    self.clip_lengths: list[int] = []
    self.fps: float = 50.0

    joint_pos, joint_vel = [], []
    body_pos, body_quat, body_lin_vel, body_ang_vel = [], [], [], []

    for path_str in cfg.motion_files:
      path = Path(path_str)
      with np.load(path, allow_pickle=False) as data:
        if "body_names" in data:
          file_bodies = tuple(str(x) for x in data["body_names"].tolist())
          indices = [file_bodies.index(name) for name in cfg.body_names]
        else:
          indices = list(range(len(cfg.body_names)))
        n = int(data["joint_pos"].shape[0])
        if cfg.max_frames is not None:
          n = min(n, cfg.max_frames)
        self.clip_names.append(path.stem)
        self.clip_lengths.append(n)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        joint_pos.append(torch.as_tensor(data["joint_pos"][:n], dtype=torch.float32))
        joint_vel.append(torch.as_tensor(data["joint_vel"][:n], dtype=torch.float32))
        body_pos.append(
          torch.as_tensor(data["body_pos_w"][:n][:, indices], dtype=torch.float32)
        )
        body_quat.append(
          torch.as_tensor(data["body_quat_w"][:n][:, indices], dtype=torch.float32)
        )
        body_lin_vel.append(
          torch.as_tensor(data["body_lin_vel_w"][:n][:, indices], dtype=torch.float32)
        )
        body_ang_vel.append(
          torch.as_tensor(data["body_ang_vel_w"][:n][:, indices], dtype=torch.float32)
        )

    if not joint_pos:
      raise ValueError("MotionLibrary received no motion files.")
    self.dt = 1.0 / self.fps
    self.num_clips = len(self.clip_names)
    self.clip_lengths_t = torch.tensor(
      self.clip_lengths, dtype=torch.long, device=self.device
    )
    self.offsets = torch.zeros(self.num_clips + 1, dtype=torch.long)
    self.offsets[1:] = torch.cumsum(self.clip_lengths_t.cpu(), dim=0)
    self.offsets = self.offsets.to(self.device)
    self.num_frames = int(self.offsets[-1].item())

    def _cat(xs: list[torch.Tensor]) -> torch.Tensor:
      return torch.cat(xs, dim=0).to(self.device, non_blocking=False)

    self.joint_pos = _cat(joint_pos)
    self.joint_vel = _cat(joint_vel)
    self.body_pos = _cat(body_pos)
    self.body_quat = _cat(body_quat)
    self.body_lin_vel = _cat(body_lin_vel)
    self.body_ang_vel = _cat(body_ang_vel)

  @property
  def num_bodies(self) -> int:
    return self.body_pos.shape[1]

  def global_index(self, clip_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    times = torch.clamp(times, min=0)
    lengths = self.clip_lengths_t[clip_ids] - 1
    times = torch.minimum(times, lengths)
    return self.offsets[clip_ids] + times

  def gather(
    self,
    clip_ids: torch.Tensor,
    times: torch.Tensor,
    horizon: int,
    offset: int = 0,
  ) -> dict[str, torch.Tensor]:
    batch = clip_ids.shape[0]
    frames = (
      times[:, None] + offset + torch.arange(horizon, device=clip_ids.device)[None, :]
    )
    idx = self.global_index(clip_ids[:, None].expand(batch, horizon), frames)
    return {
      "joint_pos": self.joint_pos[idx],
      "joint_vel": self.joint_vel[idx],
      "body_pos": self.body_pos[idx],
      "body_quat": self.body_quat[idx],
      "body_lin_vel": self.body_lin_vel[idx],
      "body_ang_vel": self.body_ang_vel[idx],
    }

  def gather_index(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
      "joint_pos": self.joint_pos[index],
      "joint_vel": self.joint_vel[index],
      "body_pos": self.body_pos[index],
      "body_quat": self.body_quat[index],
      "body_lin_vel": self.body_lin_vel[index],
      "body_ang_vel": self.body_ang_vel[index],
    }

  def clip_and_bin(
    self, clip_ids: torch.Tensor, times: torch.Tensor, bin_size_s: float = 1.0
  ) -> tuple[torch.Tensor, torch.Tensor]:
    bin_size = max(1, round(bin_size_s / self.dt))
    return clip_ids, times // bin_size

  @property
  def max_bins(self) -> int:
    return int(np.ceil(max(self.clip_lengths) * self.dt / 1.0))
