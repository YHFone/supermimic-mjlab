"""Play a trained SuperMimic policy, in a viewer or as an MP4."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.g1.env_cfgs import (
  unitree_g1_supermimic_env_cfg,
)
from supermimic.mdp.commands import (
  MotionLibraryCommand,
  MotionLibraryCommandCfg,
)
from supermimic.scripts.train import find_motion_files
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

ViewerChoice = Literal["auto", "native", "viser", "none"]


@dataclass
class PlayArgs:
  checkpoint: str | None = None
  motion_dir: str | None = None
  motion_files: tuple[str, ...] = ()
  clip: str = "0"
  start_frame: int = 0
  agent: Literal["trained", "zero", "random"] = "trained"
  num_envs: int = 1
  device: str = "cuda:0"
  viewer: ViewerChoice = "auto"
  viser_port: int | None = None
  video: str | None = None
  video_length: int = 500
  fps: int = 50
  width: int = 640
  height: int = 480
  distance: float = 2.2
  azimuth: float = 120.0
  elevation: float = -5.0
  compare_reference: bool = False
  preview: str | None = None
  stop_on_done: bool = True
  no_terminations: bool = False
  camera: str | None = None


def _load_font(size: int) -> ImageFont.ImageFont:
  for path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  ):
    if Path(path).exists():
      return ImageFont.truetype(path, size=size)
  return ImageFont.load_default()


def _label_frame(frame: np.ndarray, label: str, info: str | None = None) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  font = _load_font(max(14, image.width // 30))
  draw.rectangle([0, 0, image.width, font.size + 8], fill=(20, 20, 20))
  draw.text((6, 3), label, font=font, fill=(240, 240, 240))
  if info is not None:
    small = _load_font(max(11, image.width // 45))
    draw.rectangle(
      [0, image.height - small.size - 8, image.width, image.height],
      fill=(20, 20, 20),
    )
    draw.text(
      (6, image.height - small.size - 6), info, font=small, fill=(220, 220, 220)
    )
  return np.asarray(image)


def _select_clip(name: str, clip_names: list[str]) -> int:
  try:
    index = int(name)
  except ValueError:
    index = -1
  if index >= 0:
    if index >= len(clip_names):
      raise IndexError(f"clip index {index} out of range [0, {len(clip_names)})")
    return index
  if name not in clip_names:
    raise ValueError(f"Unknown clip {name!r}.")
  return clip_names.index(name)


class SuperMimicPolicy:
  def __init__(self, trainer: SuperMimicTrainer) -> None:
    self.trainer = trainer
    self.actor = trainer.actor_critic
    self.actor.eval()
    self.n = trainer.env.num_envs
    self._first = True

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    del obs
    trainer = self.trainer
    env = trainer.env
    state = trainer.extractor.state()
    context = trainer.extractor.context()
    bias = context[:, 2 : 2 + trainer.spec.num_joints]
    obs_t = trainer.proprio(state, trainer.last_action, bias, add_noise=False)
    if self._first:
      reset = torch.ones(self.n, dtype=torch.bool, device=trainer.device)
      self._first = False
    else:
      reset = env.reset_buf
    history = torch.where(
      reset[:, None, None],
      obs_t[:, None, :].expand_as(trainer.history),
      torch.cat([trainer.history[:, 1:], obs_t[:, None]], dim=1),
    )
    last_action = torch.where(
      reset[:, None], torch.zeros_like(trainer.last_action), trainer.last_action
    )
    trainer.history = history
    trainer.last_action = last_action
    goal = trainer.command.goal()
    action = self.actor.act(history.reshape(self.n, -1), goal)
    trainer.last_action = action
    return action


class ZeroPolicy:
  def __init__(self, num_envs: int, action_dim: int, device: str) -> None:
    self.num_envs = num_envs
    self.action_dim = action_dim
    self.device = device

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    del obs
    return torch.zeros(self.num_envs, self.action_dim, device=self.device)


class RandomPolicy:
  def __init__(self, num_envs: int, action_dim: int, device: str) -> None:
    self.num_envs = num_envs
    self.action_dim = action_dim
    self.device = device

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    del obs
    return torch.rand(self.num_envs, self.action_dim, device=self.device) * 2 - 1


def _build_env_cfg(args: PlayArgs, files, *, interactive: bool):
  env_cfg = unitree_g1_supermimic_env_cfg(files, num_envs=args.num_envs, play=True)
  env_cfg.events = {}
  env_cfg.episode_length_s = 60.0
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  env_cfg.viewer.distance = args.distance
  env_cfg.viewer.azimuth = args.azimuth
  env_cfg.viewer.elevation = args.elevation
  if args.camera is not None:
    env_cfg.viewer.camera = args.camera
  command_cfg = env_cfg.commands["motion"]
  assert isinstance(command_cfg, MotionLibraryCommandCfg)
  command_cfg.sampling_mode = "start"
  command_cfg.pose_range = {}
  command_cfg.velocity_range = {}
  command_cfg.joint_position_range = (0.0, 0.0)
  if args.no_terminations:
    env_cfg.terminations = {}
  env_cfg.auto_reset = interactive
  return env_cfg


def _run_video(args: PlayArgs, files) -> None:
  if args.checkpoint is None:
    raise ValueError("--checkpoint is required for --agent trained.")
  device = torch.device(args.device)
  checkpoint = torch.load(args.checkpoint, map_location=device)
  cfg = SuperMimicCfg(**checkpoint["cfg"])
  cfg.device = args.device
  env_cfg = _build_env_cfg(args, files, interactive=False)
  env = ManagerBasedRlEnv(env_cfg, device=args.device, render_mode="rgb_array")
  trainer = SuperMimicTrainer(env, cfg, log_dir="/tmp/supermimic_play")
  trainer.load(args.checkpoint)
  trainer.initialize()
  actor = trainer.actor_critic
  actor.eval()
  command = trainer.command
  clip_index = _select_clip(args.clip, command.library.clip_names)
  clip_name = command.library.clip_names[clip_index]
  print(f"[INFO] Rendering clip {clip_index}: {clip_name}", flush=True)

  env_ids = torch.tensor([0], device=trainer.device)
  command.reset_to_frame(env_ids, clip_index, args.start_frame)
  env.sim.forward()
  state = trainer.extractor.state()
  context = trainer.extractor.context()
  bias = context[:, 2 : 2 + trainer.spec.num_joints]
  obs = trainer.proprio(state, trainer.last_action, bias, add_noise=False)
  trainer.history[:] = obs[:, None, :].expand_as(trainer.history)
  trainer.last_action.zero_()

  reference_env = None
  if args.compare_reference:
    ref_cfg = _build_env_cfg(args, files, interactive=False)
    reference_env = ManagerBasedRlEnv(
      ref_cfg, device=args.device, render_mode="rgb_array"
    )
    reference_env.reset()

  frames: list[np.ndarray] = []
  n = env.num_envs
  try:
    for step in range(args.video_length):
      frame_index = args.start_frame + step
      with torch.no_grad():
        goal = command.goal()
        action = actor.act(trainer.history.reshape(n, -1), goal)
      _, _, terminated, truncated, _ = env.step(action)
      next_state = trainer.extractor.state()
      reference = trainer.library.gather(command.clip_ids, command.time_steps, 1)
      reward, terms = trainer.reward(
        next_state,
        action,
        trainer.last_action,
        {k: v[:, 0] for k, v in reference.items()},
      )
      trainer.last_action = action
      next_obs = trainer.proprio(
        next_state,
        trainer.last_action,
        trainer.extractor.context()[:, 2 : 2 + trainer.spec.num_joints],
        add_noise=False,
      )
      trainer.history = torch.cat([trainer.history[:, 1:], next_obs[:, None]], dim=1)
      policy_frame = env.render()
      if policy_frame is None:
        raise RuntimeError("Environment did not return an RGB frame.")
      info = (
        f"clip={clip_name}  frame={frame_index}  "
        f"r={float(reward[0]):.4f}  "
        f"pos_err={float(1.0 - terms['root_pos'][0]):.3f}"
      )
      panels = [_label_frame(policy_frame, "SuperMimic policy", info)]
      if reference_env is not None:
        ref = trainer.library.gather(
          torch.tensor([clip_index], device=trainer.device),
          torch.tensor([frame_index], device=trainer.device),
          1,
        )
        robot = reference_env.scene["robot"]
        root_state = torch.cat(
          [
            ref["body_pos"][0, 0, 0] + reference_env.scene.env_origins[0],
            ref["body_quat"][0, 0, 0],
            ref["body_lin_vel"][0, 0, 0],
            ref["body_ang_vel"][0, 0, 0],
          ],
          dim=-1,
        )[None]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
          ref["joint_pos"][0, 0][None], ref["joint_vel"][0, 0][None]
        )
        reference_env.sim.forward()
        ref_frame = reference_env.render()
        if ref_frame is not None:
          panels.append(_label_frame(ref_frame, "Reference motion"))
      frames.append(np.concatenate(panels, axis=1))
      if bool((terminated | truncated)[0]) and args.stop_on_done:
        print(f"[INFO] Episode ended at step {step + 1}.", flush=True)
        break
  finally:
    if reference_env is not None:
      reference_env.close()
    trainer.close()
    env.close()
  if not frames:
    raise RuntimeError("No frames were rendered.")
  import mediapy as media

  media.write_video(args.video, frames, fps=args.fps)
  print(f"[INFO] Wrote {args.video} ({len(frames)} frames).", flush=True)
  if args.preview is not None:
    index = min(len(frames) - 1, max(0, len(frames) // 2))
    Image.fromarray(frames[index]).save(args.preview)
    print(f"[INFO] Wrote preview {args.preview}.", flush=True)


class SuperMimicViserPlayViewer(ViserPlayViewer):
  """Viser viewer with a translucent ghost of the reference robot.

  The environment only contains the controlled robot; the reference motion is
  a command tensor.  This subclass feeds the reference qpos to the viewer's
  existing ghost-mesh path, so the reference appears as the same robot with
  its normal colors at reduced opacity instead of a separate marker overlay.
  """

  def __init__(self, env, policy, **kwargs) -> None:  # noqa: ANN001
    super().__init__(env, policy, **kwargs)
    self._reference_command: MotionLibraryCommand | None = None
    self._ghost_model = None
    self._reference_toggle = None

  def setup(self) -> None:
    super().setup()
    command = self.env.unwrapped.command_manager.get_term("motion")
    assert isinstance(command, MotionLibraryCommand)
    self._reference_command = command
    self._build_ghost_model()

    with self._server.gui.add_folder("SuperMimic reference"):
      self._reference_toggle = self._server.gui.add_checkbox(
        "Show reference ghost", True
      )

  GHOST_RGB = (0.46, 0.75, 0.95)

  def _build_ghost_model(self) -> None:
    """Copy the scene model and hide collision geoms from the ghost."""
    model = copy.deepcopy(self.env.unwrapped.sim.mj_model)
    for geom_id in range(model.ngeom):
      if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0:
        model.geom_rgba[geom_id, 3] = 0.0
      else:
        model.geom_rgba[geom_id, :3] = self.GHOST_RGB
    self._ghost_model = model

  def _reference_qpos(self) -> np.ndarray:
    """Assemble the reference robot qpos at the current command time."""
    command = self._reference_command
    assert command is not None
    env = self.env.unwrapped
    env_idx = int(self._scene.env_idx)
    clip = torch.tensor([int(command.clip_ids[env_idx].item())], device=command.device)
    time_step = torch.tensor(
      [int(command.time_steps[env_idx].item())], device=command.device
    )
    reference = command.library.gather(clip, time_step, 1)
    origin = env.scene.env_origins[env_idx].detach().cpu().numpy()

    indexing = command.robot.indexing
    qpos = np.zeros(env.sim.mj_model.nq, dtype=np.float64)
    free = indexing.free_joint_q_adr.cpu().numpy()
    joints = indexing.joint_q_adr.cpu().numpy()
    qpos[free[0:3]] = (
      reference["body_pos"][0, 0, command.anchor_index].detach().cpu().numpy() + origin
    )
    qpos[free[3:7]] = (
      reference["body_quat"][0, 0, command.anchor_index].detach().cpu().numpy()
    )
    qpos[joints] = reference["joint_pos"][0, 0].detach().cpu().numpy()
    return qpos

  def _queue_reference_ghost(self) -> None:
    if self._reference_toggle is None or not self._reference_toggle.value:
      return
    if self._ghost_model is None:
      return
    self._scene.add_ghost_mesh(
      self._reference_qpos(),
      model=self._ghost_model,
      alpha=0.45,
      label=f"reference_{self._scene.env_idx}",
    )

  def _queue_debug_visualizers(self) -> None:
    super()._queue_debug_visualizers()
    with self._sim_lock:
      self._queue_reference_ghost()


def _run_viewer(args: PlayArgs, files) -> None:
  env_cfg = _build_env_cfg(args, files, interactive=True)
  env = ManagerBasedRlEnv(env_cfg, device=args.device)
  trainer: SuperMimicTrainer | None = None
  if args.agent == "trained":
    if args.checkpoint is None:
      raise ValueError("--checkpoint is required for --agent trained.")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = SuperMimicCfg(**checkpoint["cfg"])
    cfg.device = args.device
    trainer = SuperMimicTrainer(env, cfg, log_dir="/tmp/supermimic_play")
    trainer.load(args.checkpoint)
    trainer.initialize()
    policy = SuperMimicPolicy(trainer)
  else:
    action_dim = env.action_manager.total_action_dim
    if args.agent == "zero":
      policy = ZeroPolicy(env.num_envs, action_dim, env.device)
    else:
      policy = RandomPolicy(env.num_envs, action_dim, env.device)
  if args.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = args.viewer
  try:
    if resolved == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved == "viser":
      if args.viser_port is not None:
        import viser

        server = viser.ViserServer(port=args.viser_port, label="mjlab")
        SuperMimicViserPlayViewer(env, policy, viser_server=server).run()
      else:
        SuperMimicViserPlayViewer(env, policy).run()
    else:
      raise ValueError("Use --video for headless rendering or --viewer native/viser.")
  finally:
    if trainer is not None:
      trainer.close()
    env.close()


def main() -> None:
  args = tyro.cli(PlayArgs)
  configure_torch_backends()
  files = find_motion_files(args.motion_dir, args.motion_files)
  if args.video is not None:
    _run_video(args, files)
  else:
    _run_viewer(args, files)


if __name__ == "__main__":
  main()
