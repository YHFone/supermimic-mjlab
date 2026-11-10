"""Convert a SuperMimic motion clip into the ``g1_deploy`` playback format.

The deployment controller replays a motion clip from a flat little-endian
``float32`` ``.bin`` while assembling the policy goal on the fly.  Each frame
holds the tracked joints and the anchor body pose *and velocity*::

    frame = [joint_pos (J) | joint_vel (J) | anchor_pos_w (3) |
             anchor_quat_w (4) | anchor_lin_vel_w (3) | anchor_ang_vel_w (3)]

``anchor_quat_w`` is ``(w, x, y, z)``, matching MuJoCo, mjlab, and the Unitree
``LowState`` IMU.  The anchor velocities are what the goal encoder sees in
``anchor_lin_vel_ref`` / ``anchor_ang_vel_ref``, so they must come from the same
motion source that trained the policy.

A ``.json`` description is written next to the ``.bin``; the controller reads
``frame_stride`` and ``fps`` from it.  ``--layout v1`` drops the velocity terms
(65 floats per frame) for controllers that predate them.

Example::

    supermimic-export-motion \
        --motion ~/motion_data/supermimic/g1_lafan1_40/walk1_subject1.npz \
        --out tf_s1_infer/motions/walk1_subject1.bin
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class MotionArgs:
  motion: str
  """Input ``.npz`` written by ``supermimic-preprocess``."""

  out: str = "motion.bin"
  """Output binary path; the JSON description is written next to it."""

  anchor: str = "pelvis"
  """Body whose world pose anchors the goal."""

  layout: str = "v2"
  """"v2" adds the anchor velocities; "v1" keeps the 2J + 7 legacy frame."""

  start_frame: int = 0
  """First frame of the export window."""

  num_frames: int = 0
  """Number of frames to export (0 = to the end of the clip)."""


def convert(args: MotionArgs) -> tuple[Path, Path]:
  source = Path(args.motion).expanduser()
  data = np.load(source, allow_pickle=False)
  body_names = [str(name) for name in data["body_names"]]
  if args.anchor not in body_names:
    raise SystemExit(f"{args.anchor!r} not in body_names: {body_names}")
  anchor = body_names.index(args.anchor)
  joint_names = [str(name) for name in data["joint_names"]]

  end = None if args.num_frames <= 0 else args.start_frame + args.num_frames
  window = slice(args.start_frame, end)

  joint_pos = np.asarray(data["joint_pos"][window], dtype=np.float32)
  joint_vel = np.asarray(data["joint_vel"][window], dtype=np.float32)
  anchor_pos = np.asarray(data["body_pos_w"][window, anchor], dtype=np.float32)
  anchor_quat = np.asarray(data["body_quat_w"][window, anchor], dtype=np.float32)
  anchor_lin_vel = np.asarray(data["body_lin_vel_w"][window, anchor], dtype=np.float32)
  anchor_ang_vel = np.asarray(data["body_ang_vel_w"][window, anchor], dtype=np.float32)
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])

  num_frames, num_joints = joint_pos.shape
  for name, array, shape in (
    ("anchor_pos", anchor_pos, (num_frames, 3)),
    ("anchor_quat", anchor_quat, (num_frames, 4)),
    ("anchor_lin_vel", anchor_lin_vel, (num_frames, 3)),
    ("anchor_ang_vel", anchor_ang_vel, (num_frames, 3)),
  ):
    if array.shape != shape:
      raise SystemExit(f"unexpected {name} shape {array.shape}, expected {shape}")
  # mjlab stores quaternions as (w, x, y, z).  Clips that fall over sweep the
  # whole rotation group, so only the first frame is a reliable convention
  # check (it is always close to identity); a transposed layout would put ~0
  # there.
  if abs(float(anchor_quat[0, 0])) < 0.7:
    raise SystemExit(
      "anchor_quat_w does not look like (w, x, y, z) in the first frame; "
      "refusing to convert"
    )

  terms: list[tuple[str, np.ndarray]] = [
    ("joint_pos", joint_pos),
    ("joint_vel", joint_vel),
    ("anchor_pos_w", anchor_pos),
    ("anchor_quat_w", anchor_quat),
  ]
  if args.layout == "v2":
    terms += [("anchor_lin_vel_w", anchor_lin_vel), ("anchor_ang_vel_w", anchor_ang_vel)]
  elif args.layout != "v1":
    raise SystemExit(f"unknown layout {args.layout!r} (expected v1 or v2)")

  frames = np.concatenate([array for _, array in terms], axis=1).astype("<f4")

  out = Path(args.out).expanduser()
  out.parent.mkdir(parents=True, exist_ok=True)
  frames.tofile(out)

  meta = {
    "format": f"supermimic-motion/{args.layout}",
    "source": str(source),
    "fps": fps,
    "num_frames": int(num_frames),
    "num_joints": int(num_joints),
    "joint_names": joint_names,
    "anchor": args.anchor,
    "anchor_index": int(anchor),
    "layout": [name for name, _ in terms],
    "quat_convention": "wxyz",
    "dtype": "float32",
    "frame_stride": int(frames.shape[1]),
    "start_frame": int(args.start_frame),
    "duration_s": float(num_frames / fps),
  }
  meta_path = out.with_suffix(".json")
  meta_path.write_text(json.dumps(meta, indent=2) + "\n")
  return out, meta_path


def main() -> None:
  args = tyro.cli(MotionArgs)
  out, meta_path = convert(args)
  meta = json.loads(meta_path.read_text())
  print(
    f"[INFO] wrote {out} ({meta['num_frames']} frames x {meta['frame_stride']} floats,"
    f" {meta['duration_s']:.1f} s @ {meta['fps']:.0f} Hz, layout {meta['layout']})"
  )
  print(f"[INFO] wrote {meta_path}")


if __name__ == "__main__":
  main()
