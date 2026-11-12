"""SuperMimic training loop (official GenCoord contract).

Three phases alternate every round (paper Algorithm A.1):

1. **collect** transitions with the current policy in the simulator,
2. **fit** the world model on short replay-buffer windows,
3. **improve** the actor and critic by back-propagating an imagined return
   through the frozen world model.

The learned interfaces follow ``HybridRobotics/supermimic``: the world model
works on the 68-dim GenCoord state and predicts 35 acceleration-level deltas,
termination is analytic, the actor goal is 640-dim and the world-model/critic
goal 670-dim.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from supermimic.algo.buffer import ReplayBuffer
from supermimic.algo.capture import CapturedUpdate, graph_state
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.networks import (
  ActorCritic,
  ActorCriticCfg,
  WorldModel,
  WorldModelCfg,
)
from supermimic.algo.representation import (
  GenCoordCfg,
  GenCoordRepresentation,
  decompose_full_state,
)
from supermimic.algo.reward import (
  TrackingReward,
  TrackingRewardCfg,
  soft_termination_prob,
)
from supermimic.algo.schema import SCHEMA_VERSION, check_version, stamp
from supermimic.algo.state import (
  StateExtractor,
  SuperMimicStateSpec,
  build_kinematics,
)
from supermimic.mdp.commands import MotionLibraryCommand, max_step_reward_ceiling

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def build_optimizers(
  modules: list[nn.Module],
  lr: float,
  betas: tuple[float, float] = (0.9, 0.999),
) -> list[torch.optim.Optimizer]:
  """Muon on the hidden ``Linear`` weights, Adam on everything else.

  Mirrors the official ``learner_utils.build_optimizers``: only the weights
  between the first and the last ``Linear`` of each ``nn.Sequential`` go to
  Muon; the rest (embeddings, LayerNorms, output layers) use Adam.
  """
  hidden: list[torch.Tensor] = [
    linear.weight
    for module in modules
    for seq in module.modules()
    if isinstance(seq, nn.Sequential)
    for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
  ]
  hidden_ids = {id(p) for p in hidden}
  rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
  cuda = any(p.is_cuda for p in rest)
  # A device-tensor lr can be updated in place, which keeps optimizer.step()
  # capture-safe (official learner_utils.py does the same).
  lr_arg = torch.tensor(lr, device=rest[0].device) if cuda and rest else lr
  optimizers: list[torch.optim.Optimizer] = [
    torch.optim.Adam(
      rest, lr=lr_arg, betas=betas, fused=cuda, capturable=cuda
    )
  ]
  if hidden and hasattr(torch.optim, "Muon"):
    optimizers.append(
      torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")
    )
  return optimizers


class SuperMimicTrainer:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: SuperMimicCfg,
    *,
    log_dir: str | Path | None = None,
  ) -> None:
    self.env = env
    # Official: the reward ceiling and the soft-termination thresholds come from
    # the live env config, not from hard-coded constants
    # (``envs/world_model_env.py:59-76``).
    self.max_step_reward = max_step_reward_ceiling(env)
    terms = env.cfg.terminations
    self.cfg = replace(
      cfg,
      termination_pos_threshold=float(
        terms["anchor_pos"].params["threshold"]
        if "anchor_pos" in terms
        else cfg.termination_pos_threshold
      ),
      termination_ori_threshold=float(
        terms["anchor_ori"].params["threshold"]
        if "anchor_ori" in terms
        else cfg.termination_ori_threshold
      ),
    )
    self.device = torch.device(cfg.device)
    self.log_dir = Path(log_dir or cfg.log_dir) / cfg.experiment_name
    self.iteration = 0
    self.env_steps = 0
    self._rng = torch.Generator(device=self.device)
    # Official: no seeding unless the environment is given one.
    if cfg.seed is not None:
      self._rng.manual_seed(int(cfg.seed))
      torch.manual_seed(int(cfg.seed))
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    command = env.command_manager.get_term("motion")
    assert isinstance(command, MotionLibraryCommand)
    self.command: MotionLibraryCommand = command
    self.library = self.command.library
    body_names = self.command.cfg.body_names
    self.spec = SuperMimicStateSpec.from_env(env, body_names)
    self.extractor = StateExtractor(env, self.spec)
    self.kinematics = build_kinematics(env, self.spec)
    self.forward_kinematics = (
      torch.compile(self.kinematics.forward_with_velocity, mode="default")
      if cfg.compile
      else self.kinematics.forward_with_velocity
    )
    self.anchor_index = body_names.index(self.command.cfg.anchor_body_name)

    self.representation = GenCoordRepresentation(
      GenCoordCfg(
        num_joints=self.spec.num_joints,
        dt=env.step_dt,
        decimation=env.cfg.decimation,
      ),
      default_joint_pos=self.spec.default_joint_pos,
    ).to(self.device)

    if cfg.compile and torch.cuda.is_available():
      # Official compiles ``WorldModelEnv.step``; keep the eager method as
      # ``_imagine_step_eager`` for the CUDA-graph-free path.
      self._imagine_step_eager = self._imagine_step
      self._imagine_step = torch.compile(self._imagine_step, mode="default")
    self.reward = TrackingReward(
      self.kinematics,
      self.spec.tracked_body_indices,
      self.anchor_index,
      self.spec.soft_joint_pos_limits,
      TrackingRewardCfg(dt=env.step_dt),
    ).to(self.device)

    self.policy_mode = cfg.policy_mode
    self.gamma = cfg.gamma if cfg.policy_mode == "actor-critic" else cfg.bptt_gamma
    self.actor_obs_mode = cfg.actor_obs_mode
    self.obs_dim = (
      self.representation.proprio_obs_dim(self.spec.action_dim)
      if self.actor_obs_mode == "proprio"
      else self.representation.rep_state_dim
    )
    self.actor_goal_dim = self.command.goal_features.actor_dim
    self.wm_goal_dim = self.command.goal_features.wm_dim

    self.buffer = ReplayBuffer(
      num_envs=env.num_envs,
      capacity=cfg.buffer_steps,
      state_dim=self.spec.state_dim,
      action_dim=self.spec.action_dim,
      context_dim=self.spec.context_dim,
      device=self.device,
    )

    # ``cfg.scale`` may be a float or a ``{joint_name: value}`` dict; the action
    # term resolves it into a ``(num_envs, action_dim)`` tensor in action order.
    raw_scale = env.action_manager.get_term("joint_pos").scale
    action_scale = torch.as_tensor(raw_scale, dtype=torch.float32, device=self.device)
    if action_scale.dim() == 2:
      action_scale = action_scale[0]
    action_scale = action_scale.flatten()
    if action_scale.numel() == 1:
      action_scale = action_scale.expand(self.spec.num_joints)
    if action_scale.numel() != self.spec.num_joints:
      raise ValueError(
        f"action scale has {action_scale.numel()} entries, expected "
        f"{self.spec.num_joints}"
      )
    self.world_model = WorldModel(
      self.representation,
      WorldModelCfg(
        context_dim=self.spec.context_dim,
        hidden_dims=cfg.wm_hidden_dims,
        nominal_error_root_pos=cfg.wm_nominal_error_root_pos,
        nominal_error_root_ori=cfg.wm_nominal_error_root_ori,
        nominal_error_joint_pos=cfg.wm_nominal_error_joint_pos,
        nominal_error_joint_vel=cfg.wm_nominal_error_joint_vel,
      ),
      action_scale=action_scale,
      action_offset=self.spec.default_joint_pos,
    ).to(self.device)
    self.actor_critic = ActorCritic(
      ActorCriticCfg(
        obs_dim=self.obs_dim,
        goal_dim=self.actor_goal_dim,
        wm_goal_dim=self.wm_goal_dim,
        rep_state_dim=self.representation.rep_state_dim,
        context_dim=self.spec.context_dim,
        action_dim=self.spec.action_dim,
        latent_dim=cfg.latent_dim,
        encoder_hidden=cfg.encoder_hidden_dims,
        decoder_hidden=cfg.decoder_hidden_dims,
        critic_hidden=cfg.critic_hidden_dims,
        max_step_reward=self.max_step_reward,
        gamma=cfg.gamma,
      )
    ).to(self.device)

    # Official compiles the actor used inside the imagined rollout
    # (``learners/policy_learner.py:78-80``).  The eager rollout keeps every
    # Linear/LayerNorm/GELU intermediate of the 32-step, 4096-sample BPTT graph
    # alive until backward, which costs ~3.7 GiB more than the compiled graph
    # and leaves no room for the simulator on a 24 GiB card.
    self._rollout_act = (
      torch.compile(self.actor_critic.act, mode="default")
      if cfg.compile and torch.cuda.is_available()
      else self.actor_critic.act
    )

    self.wm_optimizers = build_optimizers([self.world_model], cfg.wm_lr)
    self.actor_params = list(self.actor_critic.encoder.parameters()) + list(
      self.actor_critic.decoder.parameters()
    )
    self.actor_optimizers = build_optimizers(
      [self.actor_critic.encoder, self.actor_critic.decoder],
      cfg.actor_lr,
      betas=(0.9, 0.95),
    )
    self.critic_optimizers = build_optimizers(
      [self.actor_critic.critic], cfg.critic_lr
    )

    # Official: the world-model rollout and loss are compiled, and the whole
    # optimizer step is captured as one CUDA graph (``learners/wm_learner.py:70-76``).
    self._wm_loss_fn = (
      torch.compile(self._wm_rollout_loss, mode="default")
      if cfg.compile_world_model
      else self._wm_rollout_loss
    )
    if cfg.use_cuda_graph and torch.cuda.is_available():
      self._wm_step = CapturedUpdate(
        self._wm_update_from,
        lambda: graph_state(
          [self.world_model, self.representation], self.wm_optimizers
        ),
        warmup=3,
      )
    else:
      self._wm_step = self._wm_update_from

    self._graph_state_fn = lambda: graph_state(
      [self.actor_critic, self.world_model, self.representation],
      [*self.actor_optimizers, *self.critic_optimizers],
    )
    if cfg.use_cuda_graph and torch.cuda.is_available():
      self._policy_step = CapturedUpdate(
        self._policy_update_from, self._graph_state_fn, warmup=3
      )
    else:
      self._policy_step = self._policy_update_from
    self.last_action = torch.zeros(
      env.num_envs, self.spec.action_dim, device=self.device
    )
    self._env_sensor: torch.Tensor | None = None
    """Latest unnoised proprio sensor from the env (proprio mode only)."""

    self.episode_len = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
    self.episode_return = torch.zeros(env.num_envs, device=self.device)
    self.episode_length = torch.zeros(
      env.num_envs, dtype=torch.long, device=self.device
    )
    # Official episode-length metric: flush on a hard failure or when the cap is
    # reached, then average the last 100 flushes (supermimic/utils/logger.py:56-66).
    self._fail_len = torch.zeros(env.num_envs, device=self.device)
    self._fail_rew = torch.zeros(env.num_envs, device=self.device)
    self._fail_len_buf: deque[float] = deque(maxlen=100)
    self._fail_rew_buf: deque[float] = deque(maxlen=100)
    self._max_episode_length = int(env.max_episode_length)
    self._ep_extras: list[dict[str, float]] = []
    self._action_std_sum = 0.0
    self._action_std_steps = 0
    self._step_reward_sum = 0.0
    self._step_reward_count = 0
    self._completed_returns: list[float] = []
    self._completed_lengths: list[int] = []
    self._completed_success: list[bool] = []
    self._train_start_time = 0.0
    self._start_iteration = 0
    self._start_env_steps = 0
    self._step_accumulator = 0.0
    self._writer = None

  # ------------------------------------------------------------------
  # observations
  # ------------------------------------------------------------------
  def _sensor_from_rep(
    self, rep: torch.Tensor, last_action: torch.Tensor
  ) -> torch.Tensor:
    """Synthesize the proprio sensor from a GenCoord state.

    Layout ``[projected_gravity(3), base_ang_vel(3), joint_pos_rel(J),
    joint_vel(J), last_action(J)]`` (official ``compute_wm_obs``): projected
    gravity and the body twist are already columns of the rep state.
    """
    joints = self.spec.num_joints
    qj = rep[..., 10 : 10 + joints]
    vj = rep[..., 10 + joints :]
    joint_rel = qj - self.spec.default_joint_pos
    return torch.cat([rep[..., 1:4], rep[..., 7:10], joint_rel, vj, last_action], dim=-1)

  def observation(
    self,
    full_state: torch.Tensor,
    *,
    sensor: torch.Tensor | None = None,
    noisy: bool = False,
  ) -> torch.Tensor:
    """The actor input: 68-dim rep state, or the 93-dim (noised) proprio sensor."""
    rep = self.representation.represent(full_state)
    if self.actor_obs_mode == "rep_state":
      return rep
    obs = sensor if sensor is not None else self._sensor_from_rep(rep, self.last_action)
    if noisy:
      obs = self.representation.add_proprio_noise(obs, self.spec.action_dim)
    return obs

  def initialize(self) -> None:
    obs = self.env.reset()
    self._env_sensor = (
      obs["proprio"] if self.actor_obs_mode == "proprio" and isinstance(obs, dict) else None
    )
    self.episode_len.zero_()
    self.episode_return.zero_()
    self.episode_length.zero_()
    self._completed_returns.clear()
    self._completed_lengths.clear()
    self._completed_success.clear()
    self._action_std_sum = 0.0
    self._action_std_steps = 0
    self._step_reward_sum = 0.0
    self._step_reward_count = 0
    self.log_dir.mkdir(parents=True, exist_ok=True)
    try:
      from torch.utils.tensorboard import SummaryWriter

      self._writer = SummaryWriter(str(self.log_dir))
    except Exception:  # pragma: no cover - tensorboard is optional.
      self._writer = None
    with open(self.log_dir / "config.json", "w") as f:
      json.dump(asdict(self.cfg), f, indent=2)

  # ------------------------------------------------------------------
  # collect
  # ------------------------------------------------------------------
  @torch.no_grad()
  def _collect(self, steps: int) -> dict[str, float]:
    if steps <= 0:
      return {"collect/reward": 0.0, "collect/done_rate": 0.0}
    rollout_rewards = []
    done_count = 0
    for _ in range(steps):
      state = self.extractor.state()
      context = self.extractor.context()
      obs = (
        self.observation(state, sensor=self._env_sensor, noisy=True)
        if self.actor_obs_mode == "proprio"
        else self.observation(state)
      )
      wm_goal, actor_goal = self.command.goal()
      self.actor_critic.update_normalizers(obs, actor_goal)
      action = self.actor_critic.act(obs, actor_goal)
      noise = torch.randn(
        action.shape, generator=self._rng, device=self.device, dtype=action.dtype
      )
      noisy_action = action + noise * self.cfg.action_noise
      # Official: the world-model normalizers see every collected step (the
      # noised action that was actually executed).
      self.world_model.update_normalizers(obs, noisy_action)
      clip_ids = self.command.clip_ids.clone()
      time_steps = self.command.time_steps.clone()
      pre_frames = self.command.global_frame_index(clip_ids, time_steps)
      # Official: the env computes the reward and the termination flags before
      # the (auto) reset, so the values belong to the terminal state.
      next_obs, reward, terminated, truncated, extras = self.env.step(noisy_action)
      if self.actor_obs_mode == "proprio":
        self._env_sensor = next_obs["proprio"]
      episode_extras = _episode_reward_extras(extras)
      if episode_extras:
        self._ep_extras.append(episode_extras)
      done = terminated | truncated
      self.buffer.add(
        state,
        noisy_action,
        reward,
        done,
        context,
        clip_ids,
        time_steps,
        terminated=terminated,
      )
      self.command.record_step(pre_frames, reward, terminated, truncated)
      self.env_steps += self.env.num_envs
      rollout_rewards.append(reward.mean())
      done_count += int(done.sum().item())
      self._step_reward_sum += float(reward.sum().item())
      self._step_reward_count += reward.numel()
      self._action_std_sum += float(action.std(dim=1).mean().item())
      self._action_std_steps += 1

      self._fail_len += 1
      self._fail_rew += reward
      flush = terminated | (self._fail_len >= self._max_episode_length)
      flush_ids = flush.nonzero(as_tuple=False).squeeze(-1)
      if flush_ids.numel() > 0:
        self._fail_len_buf.extend(self._fail_len[flush_ids].cpu().tolist())
        self._fail_rew_buf.extend(self._fail_rew[flush_ids].cpu().tolist())
        self._fail_len[flush_ids] = 0.0
        self._fail_rew[flush_ids] = 0.0

      self.episode_return += reward
      self.episode_length += 1

      self.last_action = noisy_action
      if bool(done.any()):
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        for env_id in done_ids.cpu().tolist():
          self._completed_returns.append(float(self.episode_return[env_id].item()))
          self._completed_lengths.append(int(self.episode_length[env_id].item()))
          self._completed_success.append(not bool(terminated[env_id].item()))
        # auto_reset=True: the env already reset these rows inside step().
        self.last_action[done_ids] = 0.0
        self.episode_return[done_ids] = 0.0
        self.episode_length[done_ids] = 0
        self.episode_len[done_ids] = 0
    return {
      "collect/reward": float(torch.stack(rollout_rewards).mean().item())
      if rollout_rewards
      else 0.0,
      "collect/done_rate": done_count / max(1, steps * self.env.num_envs),
    }

  # ------------------------------------------------------------------
  # world model
  # ------------------------------------------------------------------
  def _sample_wm_batch(self) -> dict[str, torch.Tensor] | None:
    """One fixed-shape window batch for the world-model update.

    Sampling happens outside the captured update (official ``supermimic_runner
    .py`` samples the batches and hands them to ``wm_learner.update``).  Rows
    the buffer had to drop are padded by repeating the last one so that a
    captured update always sees the same shapes.
    """
    cfg = self.cfg
    env_ids, positions = self.buffer.sample_windows(
      cfg.wm_batch_size, cfg.wm_window, generator=self._rng
    )
    if env_ids.numel() == 0:
      return None
    if env_ids.numel() < cfg.wm_batch_size:
      pad = cfg.wm_batch_size - env_ids.numel()
      env_ids = torch.cat([env_ids, env_ids[-1:].expand(pad)])
      positions = torch.cat([positions, positions[-1:].expand(pad)])
    states, actions, _ = self.buffer.gather_states(
      env_ids, positions, before=cfg.wm_window - 1
    )
    return {
      "states": states,
      "actions": actions,
      "context": self.buffer.contexts[env_ids, positions],
    }

  def _wm_rollout_loss(
    self, states: torch.Tensor, actions: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    """Autoregressive rollout + Bryson-weighted loss over ``k - 1`` steps.

    Tensor arguments only, like the official ``wm_learner._rollout`` /
    ``_compute_losses`` that ``torch.compile`` traces.
    """
    scored = max(1, self.cfg.wm_window - 1)  # official rollout stops one short
    reps = self.representation.represent(states)
    auxs = self.representation.extract_aux_state(states)
    pred_aux, pred_rep = auxs[:, 0], reps[:, 0]
    loss = torch.zeros((), device=reps.device)
    for j in range(scored):
      pred_aux, pred_rep = self.world_model(
        pred_aux, pred_rep, actions[:, j], context
      )
      loss = loss + self.world_model.loss(
        pred_aux, pred_rep, auxs[:, j + 1], reps[:, j + 1]
      )
    return loss / scored

  def _wm_update_from(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One world-model optimizer step (CUDA-graph / capture safe)."""
    cfg = self.cfg
    loss = self._wm_loss_fn(batch["states"], batch["actions"], batch["context"])
    for optimizer in self.wm_optimizers:
      optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), cfg.wm_grad_clip)
    for optimizer in self.wm_optimizers:
      optimizer.step()
    return {"world_model/loss": loss.detach()}

  def _update_world_model(self) -> dict[str, float]:
    cfg = self.cfg
    scored = max(1, cfg.wm_window - 1)
    losses = []
    self.world_model.train()
    for _ in range(cfg.wm_updates_per_iter):
      batch = self._sample_wm_batch()
      if batch is None:
        continue
      losses.append(self._wm_step(batch)["world_model/loss"])

    group_rmse: dict[str, float] = {}
    with torch.no_grad():
      batch = self._sample_wm_batch()
      if batch is None:
        window_error = 0.0
      else:
        states = batch["states"]
        actions = batch["actions"]
        context = batch["context"]
        reps = self.representation.represent(states)
        auxs = self.representation.extract_aux_state(states)
        pred_aux, pred_rep = auxs[:, 0], reps[:, 0]
        errors = []
        group_errors: dict[str, list[torch.Tensor]] = {}
        for j in range(scored):
          pred_aux, pred_rep = self.world_model(
            pred_aux, pred_rep, actions[:, j], context
          )
          pred_blocks = self.representation.loss_blocks(pred_rep, pred_aux)
          gt_blocks = self.representation.loss_blocks(reps[:, j + 1], auxs[:, j + 1])
          mse = {}
          for k_ in pred_blocks:
            value = ((pred_blocks[k_] - gt_blocks[k_]) ** 2).mean(dim=0).mean()
            mse[k_] = value
            group_errors.setdefault(k_, []).append(value)
          errors.append(sum(mse.values()) / len(mse))
        window_error = float(torch.stack(errors).mean().sqrt().item())
        group_rmse = {
          f"world_model/rmse_{k_}": float(torch.stack(v).mean().sqrt().item())
          for k_, v in group_errors.items()
        }
    return {
      "world_model/loss": float(torch.stack(losses).mean().item())
      if losses
      else 0.0,
      "world_model/window_error": window_error,
      **group_rmse,
    }

  # ------------------------------------------------------------------
  # imagination
  # ------------------------------------------------------------------
  def _sample_imagination_starts(self, batch_size: int) -> dict[str, torch.Tensor]:
    starts = self.buffer.sample_starts(batch_size, generator=self._rng)
    states, actions, dones = self.buffer.gather_states(
      starts["env_ids"], starts["positions"], before=1
    )
    prev_action = torch.where(
      dones[:, 0:1], torch.zeros_like(actions[:, 0]), actions[:, 0]
    )
    starts["prev_action"] = prev_action
    starts["rep"] = self.representation.represent(starts["state"])
    starts["aux"] = self.representation.extract_aux_state(starts["state"])
    starts["obs"] = starts["rep"]
    return starts

  def _goal_for(
    self, clip_ids: torch.Tensor, times: torch.Tensor, aux_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    reference = self.library.gather(
      clip_ids, times, self.cfg.goal_frames, stride=self.cfg.goal_stride
    )
    return self.command.goal_features.from_reference(
      reference, aux_state[..., 0:3], aux_state[..., 3:7]
    )

  def _imagine_step(
    self,
    aux: torch.Tensor,
    rep: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    context: torch.Tensor,
    reference: dict[str, torch.Tensor],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One imagined step: dynamics -> FK -> tracking reward -> soft termination.

    Pure function of its tensors, so ``torch.compile`` can trace it (the official
    runner compiles ``WorldModelEnv.step`` the same way).
    """
    cfg = self.cfg
    next_aux, next_rep = self.world_model(aux, rep, action, context)
    full_state = self.representation.reconstruct_full_state(next_aux, next_rep)
    root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel = (
      decompose_full_state(full_state, self.spec.num_joints)
    )
    body_pos, body_quat, body_lin_vel, body_ang_vel = self.forward_kinematics(
      root_pos,
      root_quat,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    reward, _ = self.reward.forward_from_bodies(
      full_state,
      action,
      prev_action,
      reference,
      body_pos,
      body_quat,
      body_lin_vel,
      body_ang_vel,
    )
    term_prob = soft_termination_prob(
      next_aux[..., 0:3],
      next_aux[..., 3:7],
      reference["body_pos"][:, self.anchor_index],
      reference["body_quat"][:, self.anchor_index],
      pos_threshold=cfg.termination_pos_threshold,
      ori_threshold=cfg.termination_ori_threshold,
      softness=cfg.termination_softness,
    )
    return next_aux, next_rep, reward, term_prob

  def _policy_inputs(self, batch_size: int) -> dict[str, torch.Tensor]:
    starts = self._sample_imagination_starts(batch_size)
    return {
      "state": starts["state"],
      "context": starts["context"],
      "clip_ids": starts["clip_ids"],
      "time_steps": starts["time_steps"],
      "prev_action": starts["prev_action"],
    }

  def _policy_update_from(
    self, inputs: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Rollout + actor step + critic fit; capture-safe (no host syncs)."""
    cfg = self.cfg
    state_in = inputs["state"]
    total = state_in.shape[0]
    chunk = max(1, min(cfg.policy_chunk_size, total))
    _set_requires_grad(self.world_model.parameters(), False)
    _set_requires_grad(self.actor_critic.critic.parameters(), False)
    for optimizer in self.actor_optimizers:
      # In-place zeroing (not ``set_to_none``): the zeroing kernels are part of
      # the captured graph, so each replay starts from zero gradients.
      optimizer.zero_grad()

    all_rep: list[list[torch.Tensor]] = []
    all_goals: list[list[torch.Tensor]] = []
    all_rewards: list[list[torch.Tensor]] = []
    all_terms: list[list[torch.Tensor]] = []
    all_contexts: list[torch.Tensor] = []
    objective_acc = state_in.new_zeros(())
    reward_acc = state_in.new_zeros(())
    term_acc = state_in.new_zeros(())
    loss_acc = state_in.new_zeros(())

    for start_index in range(0, total, chunk):
      end_index = min(start_index + chunk, total)
      weight = (end_index - start_index) / total
      state = state_in[start_index:end_index]
      context = inputs["context"][start_index:end_index]
      clip_ids = inputs["clip_ids"][start_index:end_index]
      time_steps = inputs["time_steps"][start_index:end_index]
      prev_action = inputs["prev_action"][start_index:end_index]
      rep = self.representation.represent(state)
      aux = self.representation.extract_aux_state(state)
      obs = (
        self.representation.add_proprio_noise(
          self._sensor_from_rep(rep, prev_action), self.spec.action_dim
        )
        if self.actor_obs_mode == "proprio"
        else rep
      )

      wm_goal, actor_goal = self._goal_for(clip_ids, time_steps, aux)
      chunk_rep: list[torch.Tensor] = [rep]
      chunk_goals: list[tuple[torch.Tensor, torch.Tensor]] = [(wm_goal, actor_goal)]
      chunk_rewards: list[torch.Tensor] = []
      chunk_terms: list[torch.Tensor] = []
      times = time_steps
      for _t in range(cfg.horizon):
        action = self._rollout_act(obs, actor_goal)
        reference = _squeeze(self.library.gather(clip_ids, times, 1))
        next_aux, next_rep, reward, term_prob = self._imagine_step(
          aux, rep, action, prev_action, context, reference
        )
        times = times + 1
        wm_goal, actor_goal = self._goal_for(clip_ids, times, next_aux)
        chunk_rewards.append(reward)
        chunk_terms.append(term_prob)
        chunk_rep.append(next_rep)
        chunk_goals.append((wm_goal, actor_goal))
        aux, rep = next_aux, next_rep
        prev_action = action
        obs = (
          self.representation.add_proprio_noise(
            self._sensor_from_rep(next_rep, action), self.spec.action_dim
          )
          if self.actor_obs_mode == "proprio"
          else next_rep
        )

      reward_sum = torch.zeros_like(chunk_rewards[0])
      survival = torch.ones_like(chunk_rewards[0])
      for t, r in enumerate(chunk_rewards):
        reward_sum = reward_sum + (self.gamma**t) * survival * r
        survival = survival * (1.0 - chunk_terms[t])
      if self.policy_mode == "actor-critic":
        wm_goal_H, _ = chunk_goals[-1]
        bootstrap = self.actor_critic.value(chunk_rep[-1], wm_goal_H, context)
      else:
        bootstrap = torch.zeros_like(reward_sum)
      objective = reward_sum + (self.gamma**cfg.horizon) * survival * bootstrap
      loss = -objective.mean() * weight
      loss.backward()

      objective_acc = objective_acc + objective.mean().detach() * weight
      reward_acc = reward_acc + reward_sum.mean().detach() * weight
      term_acc = term_acc + torch.stack(chunk_terms).mean().detach() * weight
      loss_acc = loss_acc + loss.detach()

      all_rep.append([t.detach() for t in chunk_rep])
      all_goals.append([(g[0].detach(), g[1].detach()) for g in chunk_goals])
      all_rewards.append([r.detach() for r in chunk_rewards])
      all_terms.append([d.detach() for d in chunk_terms])
      all_contexts.append(context.detach())

    torch.nn.utils.clip_grad_norm_(self.actor_params, cfg.grad_clip)
    for p in self.actor_params:
      if p.grad is not None:
        p.grad.clamp_(-cfg.grad_clip_elementwise, cfg.grad_clip_elementwise)
    for optimizer in self.actor_optimizers:
      optimizer.step()

    _set_requires_grad(self.world_model.parameters(), True)
    _set_requires_grad(self.actor_critic.critic.parameters(), True)

    reps = [
      torch.cat([chunk[i] for chunk in all_rep], dim=0)
      for i in range(cfg.horizon + 1)
    ]
    wm_goals = [
      torch.cat([chunk[i][0] for chunk in all_goals], dim=0)
      for i in range(cfg.horizon + 1)
    ]
    rewards = [
      torch.cat([chunk[i] for chunk in all_rewards], dim=0)
      for i in range(cfg.horizon)
    ]
    term_probs = [
      torch.cat([chunk[i] for chunk in all_terms], dim=0)
      for i in range(cfg.horizon)
    ]
    context = torch.cat(all_contexts, dim=0)
    if self.policy_mode == "actor-critic":
      critic_loss = self._critic_fit_from(reps, wm_goals, rewards, term_probs, context)
    else:
      critic_loss = state_in.new_zeros(())
    return {
      "objective": objective_acc,
      "reward": reward_acc,
      "term_prob": term_acc,
      "loss": loss_acc,
      "critic": critic_loss,
    }

  def _update_policy(self) -> dict[str, float]:
    inputs = self._policy_inputs(self.cfg.policy_batch_size)
    outputs = self._policy_step(inputs)
    bad_grads = sum(
      int((~torch.isfinite(p.grad)).sum().item())
      for p in self.actor_params
      if p.grad is not None
    )
    return {
      "policy/objective": float(outputs["objective"].item()),
      "policy/loss": float(outputs["loss"].item()),
      "policy/reward": float(outputs["reward"].item()),
      "policy/term_prob": float(outputs["term_prob"].item()),
      "policy/bad_grads": float(bad_grads),
      "critic/loss": float(outputs["critic"].item()),
    }

  def _critic_fit_from(
    self,
    reps: list[torch.Tensor],
    goals: list[torch.Tensor],
    rewards: list[torch.Tensor],
    term_probs: list[torch.Tensor],
    context: torch.Tensor,
  ) -> torch.Tensor:
    """GAE critic fit; returns the mean loss as a tensor (capture-safe)."""
    cfg = self.cfg
    horizon = cfg.horizon
    with torch.no_grad():
      # The official recursion uses raw critic values and clamps only the
      # final targets (``policy_learner.py:177-187``).
      values = [
        self.actor_critic.value_raw(reps[t], goals[t], context)
        for t in range(horizon + 1)
      ]
      targets = [torch.zeros_like(values[0]) for _ in range(horizon)]
      running = values[-1]
      for t in range(horizon - 1, -1, -1):
        running = rewards[t] + self.gamma * (1.0 - term_probs[t]) * (
          (1.0 - cfg.lam) * values[t + 1] + cfg.lam * running
        )
        targets[t] = running
      v_max = self.actor_critic.v_max
      targets = [target.clamp(max=v_max) for target in targets]

    rep_flat = torch.cat([reps[t] for t in range(horizon)], dim=0)
    goal_flat = torch.cat([goals[t] for t in range(horizon)], dim=0)
    target_flat = torch.cat(targets, dim=0)
    context_flat = context.repeat(horizon, 1)
    n = rep_flat.shape[0]
    minibatches = max(1, cfg.critic_minibatches)
    perm = torch.randperm(n // minibatches * minibatches, device=self.device)
    total = rep_flat.new_zeros(())
    steps = 0
    for _ in range(cfg.critic_epochs):
      for idx in torch.chunk(perm, minibatches):
        value = self.actor_critic.value_raw(
          rep_flat[idx], goal_flat[idx], context_flat[idx]
        )
        loss = torch.mean((value - target_flat[idx]) ** 2)
        for optimizer in self.critic_optimizers:
          optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          self.actor_critic.critic.parameters(), cfg.grad_clip
        )
        for optimizer in self.critic_optimizers:
          optimizer.step()
        total = total + loss.detach()
        steps += 1
    return total / max(1, steps)

  # ------------------------------------------------------------------
  # training loop
  # ------------------------------------------------------------------
  def _precompile_kinematics(self) -> None:
    if not self.cfg.compile:
      return
    state = self.extractor.state()
    batch = self.cfg.policy_batch_size
    if state.shape[0] >= batch:
      state = state[:batch]
    else:
      state = state[:1].expand(batch, -1)
    state = state.detach().clone().requires_grad_(True)
    start = time.perf_counter()
    with torch.enable_grad():
      root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel = (
        decompose_full_state(state, self.spec.num_joints)
      )
      outputs = self.forward_kinematics(
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
      )
      sum(output.square().mean() for output in outputs).backward()
    torch.cuda.synchronize()
    print(
      f"[INFO] Compiled differentiable FK in {time.perf_counter() - start:.1f}s.",
      flush=True,
    )

  def train(self) -> None:
    self.initialize()
    cfg = self.cfg
    print(
      f"[INFO] SuperMimic (GenCoord): {self.env.num_envs} environments, "
      f"{len(self.command.library.clip_names)} clips, horizon={cfg.horizon}, "
      f"window={cfg.wm_window}.",
      flush=True,
    )
    self._train_start_time = time.time()
    self._start_iteration = self.iteration
    self._start_env_steps = self.env_steps
    self._collect(cfg.warmup_steps)
    self._precompile_kinematics()
    # Official schedule: one env step per iteration, a training round fires with
    # probability `ratio` = N / num_batches / batch / horizon * replay_ratio.
    should_train = Ratio(
      self.env.num_envs
      / max(1, cfg.wm_updates_per_iter)
      / max(1, cfg.wm_batch_size)
      / max(1, cfg.wm_window)
      * cfg.replay_ratio
    )
    # Metrics accumulate between logs (official keeps one loss_dict per log
    # window); only the current iteration is not representative when a round
    # fires on 40 % of the steps.
    window_metrics: dict[str, float] = {}
    window_time = {"time/collect": 0.0, "time/world_model": 0.0, "time/policy": 0.0}
    window_rounds = 0.0
    window_iters = 0
    window_start = time.time()
    while self.iteration < cfg.num_iterations:
      logs: dict[str, float] = {}
      logs.update(self._collect(1))
      train_rounds = 0
      # Time each phase where it runs: with a fractional ratio a round fires on
      # only some iterations, so a single bracket around the loop would bill the
      # idle iterations to the world model.
      wm_seconds = 0.0
      policy_seconds = 0.0
      for _ in range(should_train(self.iteration)):
        if int(self.buffer.count.min().item()) < cfg.wm_window:
          continue
        t0 = time.perf_counter()
        logs.update(self._update_world_model())
        t1 = time.perf_counter()
        wm_seconds += t1 - t0
        logs.update(self._update_policy())
        policy_seconds += time.perf_counter() - t1
        train_rounds += 1
      logs["train/rounds"] = float(train_rounds)
      logs["time/world_model"] = wm_seconds
      logs["time/policy"] = policy_seconds
      self._set_lr(self.iteration)
      self.iteration += 1

      for key, value in logs.items():
        if key in window_time:
          window_time[key] += value
        else:
          window_metrics[key] = value
      window_rounds += train_rounds
      window_iters += 1

      # Official phase: ``step % interval == 0`` with ``step`` starting at 0, so
      # the first log and the first save both happen at step 0
      # (``supermimic_runner.py:726-748``).
      step = self.iteration - 1
      if cfg.log_interval > 0 and step % cfg.log_interval == 0:
        # Official accounting: collect is the window remainder, so the three
        # parts sum to the window's wall time
        # (``supermimic_runner.py:736-743``; ``iteration_time`` is their sum,
        # ``utils/logger.py:189``).  We then divide by the iteration count so
        # the printed numbers are per iteration and match "Iteration time"
        # instead of being a ``log_interval``-long window total.
        timings = window_timings(
          time.time() - window_start,
          window_time["time/world_model"],
          window_time["time/policy"],
        )
        n_iters = max(1, window_iters)
        averaged = {key: value / n_iters for key, value in timings.items()}
        merged = {**window_metrics, **averaged, "train/rounds": window_rounds}
        self._log_iteration(merged, sum(averaged.values()))
        window_metrics.clear()
        window_time = {"time/collect": 0.0, "time/world_model": 0.0, "time/policy": 0.0}
        window_rounds = 0.0
        window_iters = 0
        # Start the next window after the log itself, matching the official
        # runner (``start_time = time.time()`` right after ``logger.log``).
        window_start = time.time()
      if cfg.save_interval > 0 and step % cfg.save_interval == 0:
        self.save()

  def _log_iteration(self, logs: dict[str, float], iteration_time: float) -> None:
    cfg = self.cfg
    width = 86
    pad = 34
    it = self.iteration
    total = cfg.num_iterations
    elapsed = max(1.0e-6, time.time() - self._train_start_time)
    done_it = max(1, it - self._start_iteration)
    fps = (self.env_steps - self._start_env_steps) / elapsed
    remaining = max(0, total - it)
    eta = elapsed / done_it * remaining

    def line(label: str, value: str) -> str:
      return f"{label + ':':>{pad}} {value}\n"

    def mean(values: list[float]) -> str:
      return f"{statistics.mean(values):.4f}" if values else "n/a"

    mean_episode_reward = mean(self._completed_returns)
    mean_episode_length = (
      f"{statistics.mean(self._completed_lengths):.2f}"
      if self._completed_lengths
      else "n/a"
    )
    normalized: list[float] = []
    if self._completed_returns:
      normalized = [
        r / (0.1 * max(length, 1))
        for r, length in zip(
          self._completed_returns, self._completed_lengths, strict=False
        )
      ]
      mean_normalized = f"{statistics.mean(normalized):.4f}"
      success_rate = (
        f"{sum(self._completed_success) / len(self._completed_success):.4f}"
      )
    else:
      mean_normalized = "n/a"
      success_rate = "n/a"
    mean_step_reward = (
      self._step_reward_sum / self._step_reward_count if self._step_reward_count else 0.0
    )
    mean_action_std = (
      self._action_std_sum / self._action_std_steps if self._action_std_steps else 0.0
    )
    term_means: dict[str, float] = {}
    if self._ep_extras:
      keys = {k for extra in self._ep_extras for k in extra}
      for key in keys:
        values = [extra[key] for extra in self._ep_extras if key in extra]
        if values:
          term_means[key] = statistics.mean(values)

    def g(key: str) -> float:
      return float(logs.get(key, 0.0))

    lines = [
      f"{'#' * width}\n",
      f"{f' Learning iteration {it}/{total} ':^{width}}\n\n",
      line("Total steps", f"{self.env_steps:,}"),
      line("Steps per second", f"{fps:.0f}"),
      line("Collection time", f"{g('time/collect'):.3f}s"),
      line("WM Learning time", f"{g('time/world_model'):.3f}s"),
      line("Policy Learning time", f"{g('time/policy'):.3f}s"),
      line("Mean world model loss", f"{g('world_model/loss'):.4f}"),
      line(
        "Mean world model 7-step error",
        f"{g('world_model/window_error'):.4f}",
      ),
      line("Mean critic loss", f"{g('critic/loss'):.4f}"),
      line("Mean policy loss", f"{g('policy/loss'):.4f}"),
      line("Mean policy objective", f"{g('policy/objective'):.4f}"),
      line("Mean imagined reward", f"{g('policy/reward'):.4f}"),
      line("Mean episode reward", mean_episode_reward),
      line(
        "Mean episode length",
        f"{statistics.mean(self._fail_len_buf):.2f}" if self._fail_len_buf else "n/a",
      ),
      line("Mean true episode length", mean_episode_length),
      line("Mean normalized return", mean_normalized),
      line("Mean reward per step", f"{mean_step_reward:.4f}"),
      line("Mean action std", f"{mean_action_std:.4f}"),
      line("Success rate", success_rate),
      line("Training rounds", f"{int(g('train/rounds'))}"),
      line("Done rate", f"{g('collect/done_rate'):.4f}"),
      line("Bad gradients", f"{int(g('policy/bad_grads'))}"),
      line("Learning rate", f"{self.actor_optimizers[0].param_groups[0]['lr']:.2e}"),
    ]
    for name, value in term_means.items():
      lines.append(line(f"Episode reward {name.split('/')[-1]}", f"{value:.5f}"))
    command_metrics = {
      name: float(value.mean().item()) for name, value in self.command.metrics.items()
    }
    for name, value in command_metrics.items():
      lines.append(line(f"Command {name}", f"{value:.4f}"))
    lines.extend(
      [
        f"{'-' * width}\n",
        line("Iteration time", f"{iteration_time:.2f}s"),
        line("Time elapsed", time.strftime("%H:%M:%S", time.gmtime(elapsed))),
        line(
          "ETA",
          time.strftime("%H:%M:%S", time.gmtime(eta)) if remaining > 0 else "-",
        ),
        f"{'#' * width}\n",
      ]
    )
    print("".join(lines), flush=True)

    if self._writer is not None:
      for key, value in logs.items():
        self._writer.add_scalar(key, value, it)
      self._writer.add_scalar("time/iteration_s", iteration_time, it)
      self._writer.add_scalar("train/mean_reward_per_step", mean_step_reward, it)
      self._writer.add_scalar("train/mean_action_std", mean_action_std, it)
      if self._completed_returns:
        self._writer.add_scalar(
          "train/mean_episode_reward", statistics.mean(self._completed_returns), it
        )
        self._writer.add_scalar(
          "train/mean_episode_length", statistics.mean(self._completed_lengths), it
        )
        self._writer.add_scalar(
          "train/mean_normalized_return", statistics.mean(normalized), it
        )
        self._writer.add_scalar(
          "train/success_rate",
          sum(self._completed_success) / len(self._completed_success),
          it,
        )
      for name, value in term_means.items():
        self._writer.add_scalar(f"episode_term/{name.split('/')[-1]}", value, it)
      for name, value in command_metrics.items():
        self._writer.add_scalar(f"command/{name}", value, it)

    self._completed_returns.clear()
    self._completed_lengths.clear()
    self._completed_success.clear()
    self._action_std_sum = 0.0
    self._action_std_steps = 0
    self._step_reward_sum = 0.0
    self._step_reward_count = 0
    self._ep_extras.clear()

  def _set_lr(self, iteration: int) -> None:
    cfg = self.cfg
    if cfg.lr_decay_iters <= 0:
      scale = 1.0
    elif iteration >= cfg.lr_decay_iters:
      scale = cfg.lr_final_ratio
    else:
      progress = iteration / cfg.lr_decay_iters
      scale = cfg.lr_final_ratio + (1.0 - cfg.lr_final_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
      )
    for optimizers, base in (
      (self.wm_optimizers, cfg.wm_lr),
      (self.actor_optimizers, cfg.actor_lr),
      (self.critic_optimizers, cfg.critic_lr),
    ):
      for optimizer in optimizers:
        for group in optimizer.param_groups:
          if torch.is_tensor(group["lr"]):
            group["lr"].fill_(base * scale)
          else:
            group["lr"] = base * scale

  def save(self) -> Path:
    """Write the official checkpoint layout into the log directory.

    The official runner writes fixed-name ``world_model.pt`` / ``actor.pt`` /
    ``critic.pt`` at every ``save_interval`` (``supermimic_runner.py:750-777``)
    instead of a data-stamped single file.
    """
    self.log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        **stamp(
          {
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "actor_goal_dim": self.actor_goal_dim,
            "wm_goal_dim": self.wm_goal_dim,
            "cfg": asdict(self.cfg),
          }
        ),
        "world_model": self.world_model.state_dict(),
        "representation": self.representation.state_dict(),
      },
      self.log_dir / "world_model.pt",
    )
    torch.save(
      {
        **stamp(),
        "encoder": self.actor_critic.encoder.state_dict(),
        "decoder": self.actor_critic.decoder.state_dict(),
        "obs_norm": self.actor_critic.obs_norm.state_dict(),
        "goal_norm": self.actor_critic.goal_norm.state_dict(),
      },
      self.log_dir / "actor.pt",
    )
    torch.save(
      {**stamp(), "critic": self.actor_critic.critic.state_dict()},
      self.log_dir / "critic.pt",
    )
    print(f"[INFO] Saved checkpoints to {self.log_dir}", flush=True)
    return self.log_dir

  def load(self, path: str | Path) -> None:
    """Load a run directory (official three-file layout) or a legacy single file."""
    path = Path(path)
    if path.is_dir():
      bundle = torch.load(path / "world_model.pt", map_location=self.device)
      check_version(bundle.get("schema_version"), what="world_model.pt")
      self.world_model.load_state_dict(bundle["world_model"])
      if "representation" in bundle:
        self.representation.load_state_dict(bundle["representation"])
      actor = torch.load(path / "actor.pt", map_location=self.device)
      check_version(actor.get("schema_version"), what="actor.pt")
      self.actor_critic.encoder.load_state_dict(actor["encoder"])
      self.actor_critic.decoder.load_state_dict(actor["decoder"])
      self.actor_critic.obs_norm.load_state_dict(actor["obs_norm"])
      self.actor_critic.goal_norm.load_state_dict(actor["goal_norm"])
      critic_path = path / "critic.pt"
      if critic_path.is_file():
        critic = torch.load(critic_path, map_location=self.device)
        check_version(critic.get("schema_version"), what="critic.pt")
        self.actor_critic.critic.load_state_dict(critic["critic"])
      self.iteration = int(bundle.get("iteration", 0))
      self.env_steps = int(bundle.get("env_steps", 0))
      return
    legacy = torch.load(path, map_location=self.device)
    check_version(legacy.get("schema_version"), what="checkpoint")
    self.world_model.load_state_dict(legacy["world_model"], strict=False)
    missing, unexpected = self.actor_critic.load_state_dict(
      legacy["actor_critic"], strict=False
    )
    if missing or unexpected:
      print(
        f"[WARN] Loaded checkpoint with missing={len(missing)} "
        f"unexpected={len(unexpected)} keys."
      )
    self.iteration = int(legacy.get("iteration", 0))
    self.env_steps = int(legacy.get("env_steps", 0))

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()


def _episode_reward_extras(extras: dict | None) -> dict[str, float]:
  """``Episode_Reward/*`` scalars from an mjlab step's extras.

  mjlab nests manager logs under ``extras["log"]`` (``reward_manager.py`` writes
  ``Episode_Reward/<term>`` there), so a flat scan of ``extras`` finds nothing.
  """
  if not extras:
    return {}
  log = extras.get("log", extras)
  return {
    key: float(value)
    for key, value in log.items()
    if key.startswith("Episode_Reward/")
  }


def window_timings(
  window_total: float, wm_seconds: float, policy_seconds: float
) -> dict[str, float]:
  """Official timing accounting: collect is the remainder, so the three parts
  sum to ``window_total`` (supermimic_runner.py:736-743)."""
  collect = window_total - wm_seconds - policy_seconds
  if collect < 0.0:  # clock jitter: keep the invariant instead of clamping alone
    wm_seconds *= window_total / max(window_total + -collect, 1e-9)
    policy_seconds *= window_total / max(window_total + -collect, 1e-9)
    collect = 0.0
  return {
    "time/collect": collect,
    "time/world_model": wm_seconds,
    "time/policy": policy_seconds,
  }


class Ratio:
  """Fractional scheduler: returns how many repeats are due at ``step``."""

  def __init__(self, ratio: float) -> None:
    self._ratio = float(ratio)
    self._prev: float | None = None

  def __call__(self, step: int) -> int:
    if self._ratio <= 0:
      return 0
    if self._prev is None:
      self._prev = float(step)
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


def _set_requires_grad(parameters, value: bool) -> list[bool]:
  parameters = list(parameters)
  previous = [p.requires_grad for p in parameters]
  for p in parameters:
    p.requires_grad_(value)
  return previous


def _squeeze(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  return {k: v[:, 0] for k, v in reference.items()}
