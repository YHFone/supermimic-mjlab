"""Evaluate a SuperMimic checkpoint on reference motions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr import geom_friction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.g1.env_cfgs import (
  unitree_g1_supermimic_env_cfg,
)
from supermimic.mdp.commands import MotionLibraryCommandCfg
from supermimic.scripts.train import find_motion_files
from mjlab.utils.torch import configure_torch_backends


@dataclass
class EvalArgs:
  checkpoint: str
  motion_dir: str | None = None
  motion_files: tuple[str, ...] = ()
  num_envs: int = 64
  episodes_per_env: int = 4
  seed: int = 0
  device: str = "cuda:0"
  output: str | None = None
  video: str | None = None
  video_fps: int = 50
  max_steps: int = 5000


def main() -> None:
  args = tyro.cli(EvalArgs)
  configure_torch_backends()
  checkpoint = torch.load(args.checkpoint, map_location=args.device)
  cfg = SuperMimicCfg(**checkpoint["cfg"])
  cfg.device = args.device
  files = find_motion_files(args.motion_dir, args.motion_files)
  num_clips = len(files)
  num_envs = max(1, min(args.num_envs, num_clips))
  torch.manual_seed(args.seed)
  env_cfg = unitree_g1_supermimic_env_cfg(files, num_envs=num_envs, play=True)
  env_cfg.seed = args.seed
  # Paper F.1: no domain randomization, foot friction 1.0, clean observations.
  env_cfg.events = {
    "eval_friction": EventTermCfg(
      mode="reset",
      func=geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (1.0, 1.0),
        "shared_random": True,
      },
    )
  }

  command_cfg = env_cfg.commands["motion"]
  assert isinstance(command_cfg, MotionLibraryCommandCfg)
  command_cfg.sampling_mode = "cycle"
  command_cfg.pose_range = {}
  command_cfg.velocity_range = {}
  command_cfg.joint_position_range = (0.0, 0.0)
  render_mode = "rgb_array" if args.video else None
  env = ManagerBasedRlEnv(env_cfg, device=args.device, render_mode=render_mode)
  trainer = SuperMimicTrainer(env, cfg, log_dir="/tmp/supermimic_eval")
  trainer.load(args.checkpoint)
  trainer.initialize()
  command = trainer.command
  actor = trainer.actor_critic
  actor.eval()

  n = env.num_envs
  ep_return = torch.zeros(n, device=trainer.device)
  ep_steps = torch.zeros(n, dtype=torch.long, device=trainer.device)
  ep_failed = torch.zeros(n, dtype=torch.bool, device=trainer.device)
  completed_returns: list[float] = []
  completed_steps: list[int] = []
  completed_failed: list[bool] = []
  completed_error_pos: list[float] = []
  completed_error_rot: list[float] = []
  ep_error_pos = torch.zeros(n, device=trainer.device)
  ep_error_rot = torch.zeros(n, device=trainer.device)
  # One deterministic pass over the library: cycle mode assigns clips in
  # order, and only the first completed rollout of each clip is recorded.
  seen_clips: set[int] = set()
  print(
    f"[INFO] {num_clips} clips, {n} envs; stopping after one rollout per clip.",
    flush=True,
  )
  frames = []

  for _ in range(args.max_steps):
    state = trainer.extractor.state()
    context = trainer.extractor.context()
    obs = trainer.observation(state, sensor=trainer._env_sensor)
    
    goal = command.goal()[1]
    with torch.no_grad():
      action = actor.act(obs, goal)
    _, _, terminated, truncated, _ = env.step(action)
    next_state = trainer.extractor.state()
    reference = trainer.library.gather(command.clip_ids, command.time_steps, 1)
    reward, _ = trainer.reward(
      next_state,
      action,
      trainer.last_action,
      {k: v[:, 0] for k, v in reference.items()},
    )
    ep_return += reward
    ep_steps += 1
    ep_failed |= terminated
    ep_error_pos += command.metrics["error_anchor_pos"]
    ep_error_rot += command.metrics["error_anchor_rot"]
    trainer.last_action = action
    done = terminated | truncated
    if args.video:
      frame = env.render()
      if frame is not None:
        frames.append(frame)
    if bool(done.any()):
      done_ids = done.nonzero(as_tuple=False).squeeze(-1)
      for env_id in done_ids.cpu().tolist():
        clip_id = int(command.clip_ids[env_id].item())
        if clip_id in seen_clips:
          continue
        seen_clips.add(clip_id)
        completed_returns.append(float(ep_return[env_id].item()))
        completed_steps.append(int(ep_steps[env_id].item()))
        completed_failed.append(bool(ep_failed[env_id].item()))
        steps = max(1, int(ep_steps[env_id].item()))
        completed_error_pos.append(float(ep_error_pos[env_id].item()) / steps)
        completed_error_rot.append(float(ep_error_rot[env_id].item()) / steps)
      env.reset(env_ids=done_ids)
      new_state = trainer.extractor.state()[done_ids]
      new_context = trainer.extractor.context()[done_ids]
      new_obs = trainer.observation(new_state, sensor=trainer._env_sensor)
      
      trainer.last_action[done_ids] = 0.0
      ep_return[done_ids] = 0.0
      ep_steps[done_ids] = 0
      ep_failed[done_ids] = False
      ep_error_pos[done_ids] = 0.0
      ep_error_rot[done_ids] = 0.0
      if len(seen_clips) >= num_clips:
        break

  summaries = summarize(
    completed_returns,
    completed_steps,
    completed_failed,
    completed_error_pos,
    completed_error_rot,
  )
  print(json.dumps(summaries, indent=2))
  if args.output is not None:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summaries, indent=2))
  if args.video and frames:
    import mediapy as media

    media.write_video(args.video, frames, fps=args.video_fps)
    print(f"[INFO] Wrote {args.video}")


def summarize(returns, steps, failed, error_pos, error_rot) -> dict[str, float]:
  if not returns:
    return {"episodes": 0}
  max_per_step = 0.1
  normalized = [
    r / (max_per_step * max(s, 1)) for r, s in zip(returns, steps, strict=False)
  ]
  n = len(returns)
  failures = sum(failed)
  return {
    "episodes": n,
    "success_rate": (n - failures) / n,
    "failed_episodes": failures,
    "mean_return": sum(returns) / n,
    "mean_normalized_return": sum(normalized) / n,
    "mean_episode_steps": sum(steps) / n,
    "mean_error_anchor_pos": sum(error_pos) / n,
    "mean_error_anchor_rot": sum(error_rot) / n,
  }


if __name__ == "__main__":
  main()
