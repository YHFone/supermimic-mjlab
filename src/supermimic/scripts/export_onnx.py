"""Export a trained SuperMimic checkpoint to ONNX for ``g1_deploy``.

The exported graph takes one ``[obs | goal]`` vector and returns the raw action
of the actor (``q_target = q0 + action_scale * action``); observation and goal
normalization are baked in.  A ``.json`` sidecar records the exact layout, the
shapes, and (when ``mjlab`` is importable) the deployment robot parameters, so
the C++ controller can be configured without guessing.

Example::

    supermimic-export-onnx \
        --checkpoint logs/2026-10-29_20-19-56/checkpoint_0036000.pt \
        --output /path/to/g1_deploy/tf_s1_infer/policy/g1_supermimic_36k.onnx
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from supermimic.algo.export import (
  GOAL_FRAMES,
  GOAL_FRAME_STRIDE,
  BASE_OBS_DIM,
  ActorSpec,
  export_actor_onnx,
  load_actor,
  onnx_io_shapes,
  verify_onnx,
)


@dataclass
class ExportArgs:
  checkpoint: str
  """Training checkpoint (``logs/<run>/checkpoint_XXXXXXX.pt``)."""

  output: str | None = None
  """Output ``.onnx`` path (default: ``<checkpoint dir>/policy_<iter>.onnx``)."""

  meta: str | None = None
  """Output ``.json`` path (default: next to the ``.onnx`` file)."""

  opset: int = 17
  """ONNX opset of the exported graph."""

  input_name: str = "policy_input"
  """Name of the single graph input."""

  output_name: str = "actions"
  """Name of the single graph output."""

  device: str = "cpu"
  """Torch device used while tracing the actor."""

  verify: bool = True
  """Check the graph against the actor with onnxruntime."""

  batch: int = 8
  """Batch size used for the verification samples."""

  robot: str = "g1"
  """Robot whose deployment parameters are recorded in the sidecar."""

  extra: dict[str, str] = field(default_factory=dict)
  """Free-form ``key=value`` pairs copied into the sidecar."""


def _default_output(checkpoint: Path, iteration: int | None) -> Path:
  suffix = f"_{iteration // 1000}k" if iteration else ""
  return checkpoint.parent / f"policy{suffix}.onnx"


def _g1_deploy_params() -> dict[str, object]:
  """Robot constants of the mjlab G1 task, for the deployment sidecar."""
  try:
    import re

    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
  except Exception as exc:  # pragma: no cover - mjlab is a hard dependency
    return {"error": f"mjlab G1 constants unavailable: {exc}"}

  model = mujoco.MjModel.from_xml_path(str(g1_constants.G1_XML))
  free = int(mujoco.mjtJoint.mjJNT_FREE)
  joint_names = [
    model.joint(i).name for i in range(model.njnt) if int(model.jnt_type[i]) != free
  ]

  kp: list[float] = []
  kd: list[float] = []
  scale: list[float] = []
  effort: list[float] = []
  for name in joint_names:
    actuator = next(
      a
      for a in g1_constants.G1_ARTICULATION.actuators
      if any(re.fullmatch(pattern, name) for pattern in a.target_names_expr)
    )
    kp.append(actuator.stiffness)
    kd.append(actuator.damping)
    effort.append(actuator.effort_limit)
    # ``JointPositionActionCfg`` scale used during training.
    scale.append(0.25 * actuator.effort_limit / actuator.stiffness)

  keyframe = g1_constants.KNEES_BENT_KEYFRAME
  default_joint_pos = [
    next(
      (value for pattern, value in keyframe.joint_pos.items() if re.fullmatch(pattern, name)),
      0.0,
    )
    for name in joint_names
  ]

  joint_ids = [model.joint(name).id for name in joint_names]
  return {
    "joint_names": joint_names,
    "default_joint_pos": default_joint_pos,
    "action_scale": scale,
    "kp": kp,
    "kd": kd,
    "effort_limit": effort,
    "joint_limit_lower": [float(model.jnt_range[i, 0]) for i in joint_ids],
    "joint_limit_upper": [float(model.jnt_range[i, 1]) for i in joint_ids],
    "mjcf": str(g1_constants.G1_XML),
  }


def _sidecar(
  spec: ActorSpec,
  payload: dict,
  *,
  checkpoint: Path,
  onnx_path: Path,
  io_shapes: tuple[list, list] | None,
  tensor_names: tuple[str, str],
  robot_params: dict[str, object],
  extra: dict[str, str],
) -> dict[str, object]:
  num_joints = spec.num_joints
  meta: dict[str, object] = {
    "format": "supermimic-onnx/v1",
    "checkpoint": str(checkpoint),
    "checkpoint_name": checkpoint.name,
    "iteration": int(payload.get("iteration", -1)),
    "onnx_file": onnx_path.name,
    "input_name": tensor_names[0],
    "output_name": tensor_names[1],
    "shapes": {
      "input": io_shapes[0] if io_shapes else None,
      "output": io_shapes[1] if io_shapes else None,
      "obs_dim": spec.obs_dim,
      "goal_dim": spec.goal_dim,
      "input_dim": spec.input_dim,
      "action_dim": spec.action_dim,
      "history": spec.history,
      "obs_frame_dim": spec.obs_frame_dim,
      "goal_frames": GOAL_FRAMES,
      "goal_stride": GOAL_FRAME_STRIDE,
      "num_joints": num_joints,
    },
    "obs_layout": {
      "terms": ["projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel", "last_action"],
      "base_terms_dim": BASE_OBS_DIM,
      "history_order": "oldest -> newest",
      "note": "joint_pos_rel includes the encoder bias during training",
    },
    "goal_layout": {
      "terms": ["content[K x (joint_pos, joint_vel)]", "anchor_ori_rot6d[K x 6]"],
      "frames": GOAL_FRAMES,
      "stride": GOAL_FRAME_STRIDE,
      "block_major": True,
      "note": (
        "content is absolute joint angles; the window starts at the current "
        "clip frame and advances by stride; anchor orientation is relative to "
        "the robot root"
      ),
    },
    "quat_convention": "wxyz",
    "robot": robot_params,
    "extra": dict(extra),
  }
  return meta


def main() -> None:
  args = tyro.cli(ExportArgs)
  checkpoint = Path(args.checkpoint).expanduser()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

  actor, spec, payload = load_actor(checkpoint, device=args.device)
  iteration = int(payload.get("iteration", -1))
  onnx_path = (
    Path(args.output).expanduser()
    if args.output
    else _default_output(checkpoint, iteration if iteration > 0 else None)
  )
  meta_path = Path(args.meta).expanduser() if args.meta else onnx_path.with_suffix(".json")

  print(f"[INFO] checkpoint {checkpoint} (iteration {iteration})")
  print(f"[INFO] actor: {spec.describe()}")
  export_actor_onnx(
    actor,
    onnx_path,
    opset=args.opset,
    input_name=args.input_name,
    output_name=args.output_name,
  )
  shapes = onnx_io_shapes(onnx_path)
  print(f"[INFO] wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KiB)")
  print(f"[INFO] graph input {shapes[0]} -> output {shapes[1]}")

  if args.verify:
    worst = verify_onnx(actor, onnx_path, batch=args.batch)
    print(f"[INFO] onnxruntime vs torch max abs diff {worst:.3e}")
    if worst > 1e-4:
      raise SystemExit(f"ONNX export deviates from the actor ({worst:.3e})")

  robot_params = _g1_deploy_params() if args.robot == "g1" else {}
  if "error" in robot_params:
    print(f"[WARN] {robot_params['error']}")
  meta = _sidecar(
    spec,
    payload,
    checkpoint=checkpoint,
    onnx_path=onnx_path,
    io_shapes=shapes,
    tensor_names=(args.input_name, args.output_name),
    robot_params=robot_params,
    extra=args.extra,
  )
  meta_path.parent.mkdir(parents=True, exist_ok=True)
  meta_path.write_text(json.dumps(meta, indent=2) + "\n")
  print(f"[INFO] wrote {meta_path}")
  if "joint_names" in robot_params:
    print(
      "[INFO] deploy: policy_file should point at this .onnx, "
      f"num_actions={spec.action_dim}, history={spec.history}, "
      f"goal_frames={spec.goal_frames}"
    )


if __name__ == "__main__":
  main()
