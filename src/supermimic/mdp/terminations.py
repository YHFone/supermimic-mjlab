"""SuperMimic-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from supermimic.mdp.commands import MotionLibraryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_end(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Terminate (as a time-out) when the reference clip reaches its last frame."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, MotionLibraryCommand)
  return command.is_motion_end()
