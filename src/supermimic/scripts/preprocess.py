"""Preprocess BONES-SEED G1 CSV clips into mjlab motion ``.npz`` files."""

from __future__ import annotations

import argparse
import csv
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
  quat_from_euler_xyz,
  quat_mul,
)

EXCLUDE_KEYWORDS = (
  "stair",
  "ladder",
  "box",
  "chair",
  "sit",
  "climb",
  "jump off 50cm",
  "look over fence",
  "ledge",
  "bench",
  "table",
  "sofa",
  "bed",
  "wall",
  "door",
  "fence",
  "platform",
  "rail",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dataset-root",
    type=Path,
    default=Path("~/motion_data/bones-seed").expanduser(),
  )
  parser.add_argument("--metadata", type=Path, default=None)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--input-fps", type=float, default=120.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--max-clips", type=int, default=None)
  parser.add_argument("--min-duration-s", type=float, default=1.0)
  parser.add_argument("--max-duration-s", type=float, default=None)
  parser.add_argument("--include-category", action="append", default=None)
  parser.add_argument(
    "--exclude-category",
    action="append",
    default=["Object Manipulation", "Object Interaction", "Environments"],
  )
  parser.add_argument("--no-name-filter", action="store_true")
  parser.add_argument("--feasibility-filter", action="store_true")
  parser.add_argument("--skip-existing", action="store_true")
  return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
  with open(path, newline="") as f:
    return list(csv.DictReader(f))


def select_clips(rows, args):
  selected = []
  include = set(args.include_category or [])
  exclude = set(args.exclude_category or [])
  for row in rows:
    if include and row["category"] not in include:
      continue
    if row["category"] in exclude:
      continue
    duration_s = float(row["move_duration_frames"]) / args.input_fps
    if duration_s < args.min_duration_s:
      continue
    if args.max_duration_s is not None and duration_s > args.max_duration_s:
      continue
    if not args.no_name_filter:
      name = row["move_name"].lower()
      if any(keyword in name for keyword in EXCLUDE_KEYWORDS):
        continue
    selected.append(row)
  if args.max_clips is not None and len(selected) > args.max_clips:
    stride = len(selected) / args.max_clips
    selected = [selected[int(i * stride)] for i in range(args.max_clips)]
  return selected


def slerp(q0, q1, t):
  dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1 = torch.where(dot < 0, -q1, q1)
  dot = dot.abs().clamp(max=1.0)
  theta = torch.acos(dot)
  sin_theta = torch.sin(theta).clamp_min(1e-6)
  w0 = torch.sin((1 - t) * theta) / sin_theta
  w1 = torch.sin(t * theta) / sin_theta
  out = w0 * q0 + w1 * q1
  linear = (1 - t) * q0 + t * q1
  out = torch.where(theta < 1e-4, linear, out)
  return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True)


def so3_derivative(q, dt):
  if q.shape[0] < 3:
    return torch.zeros_like(q[:, :3])
  q_prev, q_next = q[:-2], q[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


class Converter:
  def __init__(self, input_fps, output_fps):
    self.input_fps = input_fps
    self.output_fps = output_fps
    entity = Entity(get_g1_robot_cfg())
    model = entity.spec.compile()
    self.body_names = tuple(entity.body_names)
    self.joint_names = tuple(entity.joint_names)
    self.kinematics = DifferentiableKinematics(
      build_kinematic_tree(model, self.body_names, self.joint_names)
    )

  def convert(self, csv_path: Path, output_path: Path) -> dict:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 36:
      raise ValueError(f"Unexpected CSV shape {raw.shape} for {csv_path}")
    root_pos = torch.tensor(raw[:, 1:4] / 100.0, dtype=torch.float32)
    euler = torch.tensor(np.deg2rad(raw[:, 4:7]), dtype=torch.float32)
    root_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    joint_pos = torch.tensor(np.deg2rad(raw[:, 7:36]), dtype=torch.float32)

    n = raw.shape[0]
    duration = (n - 1) / self.input_fps
    times = torch.arange(0, duration, 1.0 / self.output_fps)
    phase = times / max(duration, 1e-9)
    idx0 = (phase * (n - 1)).floor().long().clamp(0, n - 1)
    idx1 = (idx0 + 1).clamp(max=n - 1)
    blend = (phase * (n - 1) - idx0).unsqueeze(-1)
    root_pos_i = root_pos[idx0] * (1 - blend) + root_pos[idx1] * blend
    joint_pos_i = joint_pos[idx0] * (1 - blend) + joint_pos[idx1] * blend
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


def feasibility_ok(stats: dict) -> bool:
  return not (stats["zmin_min"] < -0.12 or stats["zmin_max"] > 0.20)


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  metadata_path = args.metadata or (
    args.dataset_root / "metadata" / "seed_metadata_v004.csv"
  )
  rows = load_metadata(metadata_path)
  selected = select_clips(rows, args)
  if not selected:
    raise ValueError("No clips selected; relax the filters.")
  print(f"Selected {len(selected)} clips from {len(rows)} metadata rows.")
  converter = Converter(args.input_fps, args.output_fps)
  manifest = []
  for row in tqdm(selected, desc="Converting"):
    csv_path = args.dataset_root / row["move_g1_path"]
    output_path = args.output_dir / f"{row['move_name']}.npz"
    if args.skip_existing and output_path.exists():
      manifest.append({"name": row["move_name"], "path": str(output_path)})
      continue
    if not csv_path.exists():
      print(f"[WARN] missing {csv_path}")
      continue
    try:
      stats = converter.convert(csv_path, output_path)
    except Exception as exc:
      print(f"[WARN] failed {row['move_name']}: {exc}")
      continue
    if args.feasibility_filter and not feasibility_ok(stats):
      output_path.unlink(missing_ok=True)
      continue
    manifest.append(
      {
        "name": row["move_name"],
        "path": str(output_path),
        "category": row["category"],
        "frames": stats["frames"],
        "duration_s": stats["duration_s"],
      }
    )
  index_path = args.output_dir / "index.json"
  with open(index_path, "w") as f:
    json.dump(
      {
        "fps": args.output_fps,
        "num_clips": len(manifest),
        "clips": manifest,
        "source": str(metadata_path),
        "filters": {
          "max_clips": args.max_clips,
          "include_category": args.include_category,
          "exclude_category": args.exclude_category,
          "name_filter": not args.no_name_filter,
          "feasibility_filter": args.feasibility_filter,
        },
      },
      f,
      indent=2,
    )
  total_hours = sum((float(m["duration_s"]) for m in manifest), 0.0) / 3600.0
  print(f"Wrote {len(manifest)} motions ({total_hours:.2f} h) to {args.output_dir}")


if __name__ == "__main__":
  main()
