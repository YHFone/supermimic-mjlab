"""Convert the LAFAN1 G1 retargeting CSVs to SuperMimic motion ``.npz`` files.

The LAFAN1 retargeting release stores each frame as 36 comma-separated values:

    root_pos(3), root_quat_xyzw(4), joint_pos(29)

at 30 Hz, with the joints in the order listed in ``CSV_JOINT_NAMES``.  This
script resamples to 50 Hz, converts the quaternion to mjlab's wxyz convention,
reorders the joints to the mjlab G1 order, differentiates positions for
velocities, and runs differentiable FK to obtain all body poses and spatial
velocities in the format consumed by ``MotionLibrary``.

Usage:

    python -m supermimic.scripts.preprocess_lafan1 \
        --dataset-root ~/motion_data/LAFAN1_Retargeting_Dataset/g1 \
        --output-dir ~/motion_data/supermimic/g1_lafan1_40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from supermimic.algo.kinematics import (
  DifferentiableKinematics,
  build_kinematic_tree,
)
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)

CSV_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dataset-root",
    type=Path,
    default=Path("~/motion_data/LAFAN1_Retargeting_Dataset/g1").expanduser(),
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("~/motion_data/supermimic/g1_lafan1_40").expanduser(),
  )
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--max-clips", type=int, default=None)
  parser.add_argument(
    "--clip",
    action="append",
    default=None,
    help="Only convert these CSV stems (can be repeated).",
  )
  parser.add_argument("--skip-existing", action="store_true")
  return parser.parse_args()


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Spherical interpolation with the shortest arc and a linear fallback."""
  dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1 = torch.where(dot < 0, -q1, q1)
  dot = dot.abs().clamp(max=1.0)
  theta = torch.acos(dot)
  sin_theta = torch.sin(theta).clamp_min(1e-6)
  w0 = torch.sin((1.0 - t) * theta) / sin_theta
  w1 = torch.sin(t * theta) / sin_theta
  out = w0 * q0 + w1 * q1
  linear = (1.0 - t) * q0 + t * q1
  out = torch.where(theta < 1e-4, linear, out)
  return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(1e-6)


def so3_derivative(q: torch.Tensor, dt: float) -> torch.Tensor:
  """World-frame angular velocity from a wxyz quaternion sequence."""
  if q.shape[0] < 3:
    return torch.zeros(q.shape[0], 3)
  q_prev, q_next = q[:-2], q[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def make_quaternion_continuous(q: torch.Tensor) -> torch.Tensor:
  """Flip quaternion signs so consecutive frames lie in the same hemisphere."""
  out = [q[0]]
  for i in range(1, q.shape[0]):
    prev = out[-1]
    current = q[i]
    current = torch.where(torch.dot(prev, current) < 0, -current, current)
    out.append(current)
  return torch.stack(out, dim=0)


class Converter:
  def __init__(self, input_fps: float, output_fps: float) -> None:
    self.input_fps = input_fps
    self.output_fps = output_fps
    entity = Entity(get_g1_robot_cfg())
    model = entity.spec.compile()
    self.body_names = tuple(entity.body_names)
    self.joint_names = tuple(entity.joint_names)
    if set(self.joint_names) != set(CSV_JOINT_NAMES):
      missing = set(self.joint_names) - set(CSV_JOINT_NAMES)
      extra = set(CSV_JOINT_NAMES) - set(self.joint_names)
      raise ValueError(f"Joint mismatch: missing={missing}, extra={extra}")
    self.csv_joint_indices = [CSV_JOINT_NAMES.index(n) for n in self.joint_names]
    self.kinematics = DifferentiableKinematics(
      build_kinematic_tree(model, self.body_names, self.joint_names)
    )

  def convert(self, csv_path: Path, output_path: Path) -> dict:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36:
      raise ValueError(f"Unexpected CSV shape {raw.shape} for {csv_path}")
    if raw.shape[0] < 4:
      raise ValueError(f"Clip too short: {csv_path}")

    root_pos = torch.tensor(raw[:, 0:3], dtype=torch.float32)
    # LAFAN1 stores QX QY QZ QW; mjlab uses QW QX QY QZ.
    root_quat = torch.tensor(raw[:, [6, 3, 4, 5]], dtype=torch.float32)
    root_quat = root_quat / torch.linalg.vector_norm(
      root_quat, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    root_quat = make_quaternion_continuous(root_quat)
    joint_pos = torch.tensor(raw[:, 7:][:, self.csv_joint_indices], dtype=torch.float32)

    num_frames = raw.shape[0]
    duration = (num_frames - 1) / self.input_fps
    times = torch.arange(0.0, duration, 1.0 / self.output_fps, dtype=torch.float32)
    src = times * self.input_fps
    idx0 = torch.floor(src).long().clamp(0, num_frames - 1)
    idx1 = (idx0 + 1).clamp(max=num_frames - 1)
    blend = (src - idx0).unsqueeze(-1)

    root_pos_i = root_pos[idx0] * (1.0 - blend) + root_pos[idx1] * blend
    joint_pos_i = joint_pos[idx0] * (1.0 - blend) + joint_pos[idx1] * blend
    root_quat_i = slerp(root_quat[idx0], root_quat[idx1], blend)

    dt = 1.0 / self.output_fps
    root_lin_vel = torch.gradient(root_pos_i, spacing=dt, dim=0)[0]
    joint_vel = torch.gradient(joint_pos_i, spacing=dt, dim=0)[0]
    root_ang_vel = so3_derivative(root_quat_i, dt)
    body_pos, body_quat, body_lin_vel, body_ang_vel = (
      self.kinematics.forward_with_velocity(
        root_pos_i,
        root_quat_i,
        root_lin_vel,
        root_ang_vel,
        joint_pos_i,
        joint_vel,
      )
    )
    zmin = body_pos[..., 2].min(dim=1).values
    stats = {
      "frames": int(joint_pos_i.shape[0]),
      "duration_s": float(duration),
      "zmin_min": float(zmin.min().item()),
      "zmin_max": float(zmin.max().item()),
    }
    np.savez(
      output_path,
      fps=np.array([self.output_fps], dtype=np.float32),
      body_names=np.array(self.body_names),
      joint_names=np.array(self.joint_names),
      joint_pos=joint_pos_i.numpy(),
      joint_vel=joint_vel.numpy(),
      body_pos_w=body_pos.numpy(),
      body_quat_w=body_quat.numpy(),
      body_lin_vel_w=body_lin_vel.numpy(),
      body_ang_vel_w=body_ang_vel.numpy(),
    )
    return stats


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  csv_paths = sorted(args.dataset_root.glob("*.csv"))
  if args.clip is not None:
    wanted = set(args.clip)
    csv_paths = [p for p in csv_paths if p.stem in wanted]
  if not csv_paths:
    raise ValueError(f"No CSV files found in {args.dataset_root}.")
  if args.max_clips is not None:
    csv_paths = csv_paths[: args.max_clips]

  converter = Converter(args.input_fps, args.output_fps)
  manifest = []
  for csv_path in tqdm(csv_paths, desc="Converting LAFAN1"):
    output_path = args.output_dir / f"{csv_path.stem}.npz"
    if args.skip_existing and output_path.exists():
      try:
        with np.load(output_path) as data:
          frames = int(data["joint_pos"].shape[0])
          fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      except Exception:
        frames, fps = 0, args.output_fps
      manifest.append(
        {
          "name": csv_path.stem,
          "path": str(output_path),
          "frames": frames,
          "duration_s": frames / fps if fps else 0.0,
        }
      )
      continue
    try:
      stats = converter.convert(csv_path, output_path)
    except Exception as exc:
      print(f"[WARN] failed {csv_path.name}: {exc}")
      continue
    manifest.append(
      {
        "name": csv_path.stem,
        "path": str(output_path),
        "frames": stats["frames"],
        "duration_s": stats["duration_s"],
        "zmin_min": stats["zmin_min"],
        "zmin_max": stats["zmin_max"],
      }
    )

  index_path = args.output_dir / "index.json"
  with open(index_path, "w") as f:
    json.dump(
      {
        "fps": args.output_fps,
        "num_clips": len(manifest),
        "clips": manifest,
        "source": str(args.dataset_root),
        "format": "LAFAN1 retargeting CSV (root xyz + quaternion xyzw + 29 joints)",
      },
      f,
      indent=2,
    )
  total_hours = sum(float(m["duration_s"]) for m in manifest) / 3600.0
  print(
    f"Wrote {len(manifest)} motions ({total_hours:.2f} h, "
    f"{sum(int(m['frames']) for m in manifest):,} frames) to {args.output_dir}"
  )


if __name__ == "__main__":
  main()
