"""Per-environment ring buffer of simulator transitions."""

from __future__ import annotations

import torch


class ReplayBuffer:
  def __init__(
    self,
    num_envs: int,
    capacity: int,
    state_dim: int,
    action_dim: int,
    context_dim: int,
    device: str | torch.device,
    *,
    terminal_capacity: int = 65_536,
  ) -> None:
    self.num_envs = num_envs
    self.capacity = capacity
    self.device = torch.device(device)
    self.states = torch.zeros(num_envs, capacity, state_dim, device=self.device)
    self.actions = torch.zeros(num_envs, capacity, action_dim, device=self.device)
    self.contexts = torch.zeros(num_envs, capacity, context_dim, device=self.device)
    self.rewards = torch.zeros(num_envs, capacity, device=self.device)
    self.dones = torch.zeros(num_envs, capacity, dtype=torch.bool, device=self.device)
    self.terminations = torch.zeros(
      num_envs, capacity, dtype=torch.bool, device=self.device
    )
    self.clip_ids = torch.zeros(
      num_envs, capacity, dtype=torch.int32, device=self.device
    )
    self.time_steps = torch.zeros(
      num_envs, capacity, dtype=torch.long, device=self.device
    )
    self.ptr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    # Hard-terminal states s_{t+1}.  They are never stored in ``states`` (the
    # env is reset before the next add), but the termination head needs positive
    # examples of "this state is terminal" for Eq. 1/2/3.
    self.terminal_capacity = int(terminal_capacity)
    self.terminal_states = torch.zeros(
      self.terminal_capacity, state_dim, device=self.device
    )
    self.terminal_clip_ids = torch.zeros(
      self.terminal_capacity, dtype=torch.long, device=self.device
    )
    self.terminal_time_steps = torch.zeros(
      self.terminal_capacity, dtype=torch.long, device=self.device
    )
    self.terminal_ptr = 0
    self.terminal_count = 0

  @property
  def numel(self) -> int:
    return int(self.count.sum().item())

  def add(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    context: torch.Tensor,
    clip_id: torch.Tensor,
    time_step: torch.Tensor,
    terminated: torch.Tensor | None = None,
    next_state: torch.Tensor | None = None,
  ) -> None:
    idx = self.ptr
    env_ids = torch.arange(self.num_envs, device=self.device)
    self.states[env_ids, idx] = state
    self.actions[env_ids, idx] = action
    self.contexts[env_ids, idx] = context
    self.rewards[env_ids, idx] = reward
    self.dones[env_ids, idx] = done
    # Hard simulator termination (height/tilt), as opposed to a time-out.
    self.terminations[env_ids, idx] = done if terminated is None else terminated
    self.clip_ids[env_ids, idx] = clip_id.to(torch.int32)
    self.time_steps[env_ids, idx] = time_step
    self.ptr = (self.ptr + 1) % self.capacity
    self.count = torch.minimum(
      self.count + 1, torch.full_like(self.count, self.capacity)
    )
    if next_state is not None and self.terminal_capacity > 0:
      hard = done if terminated is None else terminated
      ids = hard.nonzero(as_tuple=False).squeeze(-1)
      if ids.numel() > 0:
        n = ids.numel()
        slots = (
          self.terminal_ptr + torch.arange(n, device=self.device)
        ) % self.terminal_capacity
        self.terminal_states[slots] = next_state[ids]
        self.terminal_clip_ids[slots] = clip_id[ids].long()
        self.terminal_time_steps[slots] = time_step[ids].long() + 1
        self.terminal_ptr = int((self.terminal_ptr + n) % self.terminal_capacity)
        self.terminal_count = min(self.terminal_count + n, self.terminal_capacity)

  def _valid_len(self) -> torch.Tensor:
    return self.count

  def sample_windows(
    self, batch_size: int, length: int, generator: torch.Generator | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(env_id, end_position)`` pairs for windows inside one episode.

    The returned position is the last state of the window, so
    ``gather_states(env_ids, positions, before=length - 1)`` yields the
    window in chronological order.  Rows whose environment has no clean
    window are dropped, so the returned batch can be shorter than
    ``batch_size`` (or empty).
    """
    env_ids = torch.randint(
      0, self.num_envs, (batch_size,), device=self.device, generator=generator
    )
    valid = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
    positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    for _ in range(32):
      remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
      if remaining.numel() == 0:
        break
      n = remaining.numel()
      e = env_ids[remaining]
      valid_len = self._valid_len()[e]
      ok = valid_len >= length
      if not bool(ok.any()):
        break
      max_start = (valid_len - length).clamp_min(0)
      offset = (
        torch.rand(n, device=self.device, generator=generator) * (max_start + 1).float()
      ).long()
      pos = (self.ptr[e] - valid_len + offset) % self.capacity
      offsets = torch.arange(length - 1, device=self.device)
      check = (pos[:, None] + offsets[None, :]) % self.capacity
      bad = self.dones[e[:, None], check].any(dim=-1) | (~ok)
      end = (pos + (length - 1)) % self.capacity
      positions[remaining] = torch.where(bad, positions[remaining], end)
      valid[remaining] = ~bad
    if not bool(valid.all()):
      # Never return a window that crosses a reset: drop the rows instead
      # (callers handle a shorter/empty batch).
      keep = valid.nonzero(as_tuple=False).squeeze(-1)
      env_ids = env_ids[keep]
      positions = positions[keep]
    return env_ids, positions

  def sample_starts(
    self, batch_size: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    env_ids = torch.randint(
      0, self.num_envs, (batch_size,), device=self.device, generator=generator
    )
    valid_len = self._valid_len()[env_ids].clamp_min(1)
    offset = (
      torch.rand(batch_size, device=self.device, generator=generator)
      * valid_len.float()
    ).long()
    pos = (self.ptr[env_ids] - valid_len + offset) % self.capacity
    return {
      "env_ids": env_ids,
      "positions": pos,
      "state": self.states[env_ids, pos],
      "context": self.contexts[env_ids, pos],
      "clip_ids": self.clip_ids[env_ids, pos].long(),
      "time_steps": self.time_steps[env_ids, pos],
    }

  def gather_states(
    self, env_ids: torch.Tensor, positions: torch.Tensor, before: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.arange(-before, 1, device=self.device)
    idx = (positions[:, None] + offsets[None, :]) % self.capacity
    return (
      self.states[env_ids[:, None], idx],
      self.actions[env_ids[:, None], idx],
      self.dones[env_ids[:, None], idx],
    )

  def sample_terminal(
    self, n: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    """Hard-terminal states (label 1 for the termination head)."""
    if self.terminal_count == 0 or n <= 0:
      return {
        "state": torch.zeros(0, self.terminal_states.shape[1], device=self.device),
        "clip_ids": torch.zeros(0, dtype=torch.long, device=self.device),
        "time_steps": torch.zeros(0, dtype=torch.long, device=self.device),
      }
    idx = torch.randint(
      0, self.terminal_count, (n,), device=self.device, generator=generator
    )
    return {
      "state": self.terminal_states[idx],
      "clip_ids": self.terminal_clip_ids[idx],
      "time_steps": self.terminal_time_steps[idx],
    }

  def sample_random(
    self, n: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    flat = torch.randint(
      0, self.num_envs * self.capacity, (n,), device=self.device, generator=generator
    )
    env_ids = flat // self.capacity
    pos = flat % self.capacity
    live = pos < self.count[env_ids]
    return {
      "state": self.states[env_ids, pos],
      "action": self.actions[env_ids, pos],
      "context": self.contexts[env_ids, pos],
      "clip_ids": self.clip_ids[env_ids, pos].long(),
      "time_steps": self.time_steps[env_ids, pos],
      "terminated": self.terminations[env_ids, pos],
      "live": live,
    }
