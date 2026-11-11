"""Train a SuperMimic motion-tracking policy in mjlab."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tyro

from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.g1.env_cfgs import (
  unitree_g1_supermimic_env_cfg,
)
from supermimic.config.g1.rl_cfg import (
  unitree_g1_supermimic_cfg,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainArgs:
  motion_dir: str | None = None
  motion_files: tuple[str, ...] = ()
  num_envs: int = 4096
  episode_length_s: float = 10.0
  device: str = "cuda:0"
  seed: int = 0
  log_dir: str = "logs/supermimic"
  experiment_name: str = "g1_supermimic"
  resume: str | None = None
  sm: SuperMimicCfg = field(default_factory=unitree_g1_supermimic_cfg)


def find_motion_files(
  motion_dir: str | None, motion_files: tuple[str, ...]
) -> tuple[str, ...]:
  if motion_files:
    return motion_files
  if motion_dir is None:
    raise ValueError("Provide --motion-dir or --motion-files.")
  root = Path(motion_dir)
  index = root / "index.json"
  if index.exists():
    import json

    with open(index) as f:
      manifest = json.load(f)
    paths = tuple(str(root / f"{clip['name']}.npz") for clip in manifest["clips"])
    return tuple(path for path in paths if Path(path).exists())
  paths = sorted(str(p) for p in root.glob("*.npz"))
  if not paths:
    raise ValueError(f"No .npz files found in {root}.")
  return tuple(paths)


def resolve_motion_files(args: TrainArgs) -> tuple[str, ...]:
  return find_motion_files(args.motion_dir, args.motion_files)


def timestamped_name(name: str, when: datetime | None = None) -> str:
  """Append a run timestamp so each training run gets its own log directory."""
  stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
  return f"{name}/{stamp}"


def main() -> None:
  args = tyro.cli(TrainArgs)
  configure_torch_backends()
  files = resolve_motion_files(args)
  if len(files) == 0:
    raise ValueError("No motion files resolved.")
  print(f"[INFO] Using {len(files)} motion clips.", flush=True)

  env_cfg = unitree_g1_supermimic_env_cfg(files, num_envs=args.num_envs, play=False)
  env_cfg.episode_length_s = args.episode_length_s
  env_cfg.seed = args.seed

  cfg = args.sm
  cfg.device = args.device
  cfg.seed = args.seed
  cfg.log_dir = args.log_dir
  cfg.experiment_name = timestamped_name(args.experiment_name)
  print(f"[INFO] Logging to {Path(cfg.log_dir) / cfg.experiment_name}", flush=True)
  if cfg.lr_decay_iters > cfg.num_iterations:
    print(
      f"[INFO] lr_decay_iters={cfg.lr_decay_iters} > num_iterations="
      f"{cfg.num_iterations}; annealing over the run length instead.",
      flush=True,
    )
    cfg.lr_decay_iters = cfg.num_iterations

  env = ManagerBasedRlEnv(env_cfg, device=args.device)
  trainer = SuperMimicTrainer(env, cfg, log_dir=cfg.log_dir)
  if args.resume is not None:
    trainer.load(args.resume)
  try:
    trainer.train()
    trainer.save("checkpoint_final.pt")
  except KeyboardInterrupt:
    print("[INFO] Interrupted; saving checkpoint_interrupt.pt ...", flush=True)
    trainer.save("checkpoint_interrupt.pt")
  finally:
    trainer.close()
    env.close()


if __name__ == "__main__":
  main()
