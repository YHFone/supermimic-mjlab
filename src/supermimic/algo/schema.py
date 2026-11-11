"""Versioning for checkpoints, exported policies, and deployment sidecars.

The official SuperMimic implementation (``HybridRobotics/supermimic``) defines
the numeric contract this package now follows: a 68-dimensional GenCoord
representation, 35-dimensional dynamics output, 67-dimensional goal frames with
stride 5, and a 640-dimensional actor goal with a 670-dimensional world-model
goal.  Any artifact produced before that alignment is numerically incompatible —
loading one silently would mis-index every feature — so every reader validates
the version first and fails loudly.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 2
"""Version of the observation/state/goal contract.

* ``1`` — pre-alignment layout: 71-dim state, 70-dim prediction, 73-dim goal
  frames (730 total), learned termination head, 10-frame proprio history.
* ``2`` — official GenCoord layout: 68/7/35, 67-dim goal frames (actor 640,
  world model 670), analytic termination, optional 1-frame rep_state history.
"""


class SchemaMismatchError(ValueError):
  """Raised when an artifact was written by an incompatible version."""


def check_version(actual: Any, *, what: str) -> None:
  """Validate ``actual`` against :data:`SCHEMA_VERSION`."""
  if actual is None:
    raise SchemaMismatchError(
      f"{what} has no schema version; it predates the GenCoord alignment and "
      f"cannot be loaded.  Retrain or re-export with schema v{SCHEMA_VERSION}."
    )
  try:
    version = int(actual)
  except (TypeError, ValueError) as exc:
    raise SchemaMismatchError(f"{what} has a non-numeric schema version {actual!r}.") from exc
  if version != SCHEMA_VERSION:
    raise SchemaMismatchError(
      f"{what} was written with schema v{version}, this build expects "
      f"v{SCHEMA_VERSION} (68/7/35 GenCoord, 640/670 goal).  Retrain or "
      f"re-export the artifact."
    )


def stamp(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Build an artifact metadata block carrying the schema version."""
  meta: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
  if extra:
    meta.update(extra)
  return meta
