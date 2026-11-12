"""Command-term plumbing: RSI clip-end bookkeeping and reference write-back."""

from types import SimpleNamespace

import torch

from supermimic.mdp.commands import MotionLibraryCommand


def _command(cached, fallback):
  manager = SimpleNamespace(get_term=lambda name: cached)
  return SimpleNamespace(
    _env=SimpleNamespace(termination_manager=manager),
    is_motion_end=fallback,
  )


def test_clip_end_uses_cached_termination_term_not_resampled_time_steps():
  """After ``env.step`` auto-resets, ``is_motion_end()`` reads the new episode."""
  fallback = lambda: torch.tensor([False, False])  # noqa: E731 - stand-in
  cmd = _command(torch.tensor([True, False]), fallback)
  out = MotionLibraryCommand._clip_end_flags(cmd)
  assert out.tolist() == [True, False]


def test_clip_end_falls_back_when_term_is_not_active():
  def raise_key_error(name):
    raise KeyError(name)

  manager = SimpleNamespace(get_term=raise_key_error)
  cmd = SimpleNamespace(
    _env=SimpleNamespace(termination_manager=manager),
    is_motion_end=lambda: torch.tensor([False, True]),
  )
  out = MotionLibraryCommand._clip_end_flags(cmd)
  assert out.tolist() == [False, True]


def test_record_step_splits_clip_end_from_time_out():
  recorded = {}

  class Sampler:
    def record(self, frames, rewards, terminated, clip_end, timed_out):
      recorded.update(
        terminated=terminated.clone(),
        clip_end=clip_end.clone(),
        timed_out=timed_out.clone(),
      )
      return False

  cmd = SimpleNamespace(
    cfg=SimpleNamespace(sampling_mode="return"),
    sampler=Sampler(),
    _env=SimpleNamespace(
      termination_manager=SimpleNamespace(
        get_term=lambda name: torch.tensor([True, False, False])
      )
    ),
    is_motion_end=lambda: torch.tensor([False, False, False]),
  )
  cmd._clip_end_flags = lambda: MotionLibraryCommand._clip_end_flags(cmd)
  terminated = torch.tensor([False, False, True])
  truncated = torch.tensor([True, False, True])
  MotionLibraryCommand.record_step(
    cmd, torch.zeros(3, dtype=torch.long), torch.zeros(3), terminated, truncated
  )
  # A clip that ends is its own bucket, and a step that both fails and times
  # out counts as a time-out -- exactly the official split.
  assert recorded["clip_end"].tolist() == [True, False, False]
  assert recorded["terminated"].tolist() == [False, False, False]
  assert recorded["timed_out"].tolist() == [False, False, True]


def test_reference_write_clips_joint_positions_to_soft_limits():
  written = {}

  class Robot:
    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
      written["joint_pos"] = joint_pos.clone()

    def write_root_state_to_sim(self, root_state, env_ids=None):
      pass

    def reset(self, env_ids=None):
      pass

  cmd = SimpleNamespace(
    robot=Robot(),
    soft_joint_pos_limits=torch.tensor([[[-0.5, 0.5], [-0.5, 0.5]]]),
  )
  MotionLibraryCommand._write_reference_state_to_sim(
    cmd,
    torch.tensor([0]),
    torch.zeros(1, 3),
    torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    torch.zeros(1, 3),
    torch.zeros(1, 3),
    torch.tensor([[2.0, -2.0]]),
    torch.zeros(1, 2),
  )
  assert written["joint_pos"].tolist() == [[0.5, -0.5]]
