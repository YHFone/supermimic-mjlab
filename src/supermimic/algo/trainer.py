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
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from supermimic.algo.buffer import ReplayBuffer
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.networks import (
  ActorCritic,
  ActorCriticCfg,
  WorldModel,
  WorldModelCfg,
)
from supermimic.algo.representation import GenCoordCfg, GenCoordRepresentation
from supermimic.algo.reward import (
  TRACKING_REWARD_TERMS,
  TrackingReward,
  TrackingRewardCfg,
  soft_termination_prob,
)
from supermimic.algo.schema import SCHEMA_VERSION, check_version, stamp
from supermimic.algo.state import (
  StateExtractor,
  SuperMimicStateSpec,
  build_kinematics,
  context_encoder_bias,
)
from supermimic.mdp.commands import MotionLibraryCommand
from supermimic.mdp.features import ProprioFeatures, ProprioNoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def build_optimizers(
  modules: list[nn.Module],
  lr: float,
  betas: tuple[float, float] = (0.9, 0.999),
  *,
  use_muon: bool = True,
) -> list[torch.optim.Optimizer]:
  """Muon on the hidden ``Linear`` weights, Adam on everything else.

  Mirrors the official ``learner_utils.build_optimizers``: only the weights
  between the first and the last ``Linear`` of each ``nn.Sequential`` go to
  Muon; the rest (embeddings, LayerNorms, output layers) use Adam.
  """
  hidden: list[torch.Tensor] = []
  if use_muon:
    hidden = [
      linear.weight
      for module in modules
      for seq in module.modules()
      if isinstance(seq, nn.Sequential)
      for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
    ]
  hidden_ids = {id(p) for p in hidden}
  rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
  optimizers: list[torch.optim.Optimizer] = [
    torch.optim.Adam(rest, lr=lr, betas=betas)
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
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.log_dir = Path(log_dir or cfg.log_dir) / cfg.experiment_name
    self.iteration = 0
    self.env_steps = 0
    self._rng = torch.Generator(device=self.device)
    self._rng.manual_seed(cfg.seed)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(cfg.seed)
    if cfg.use_tf32 and torch.cuda.is_available():
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      torch.set_float32_matmul_precision("high")

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

    self.reward = TrackingReward(
      self.kinematics,
      self.spec.tracked_body_indices,
      self.anchor_index,
      self.spec.soft_joint_pos_limits,
      TrackingRewardCfg(dt=env.step_dt),
    ).to(self.device)

    self.proprio = ProprioFeatures(
      self.spec.num_joints,
      self.spec.default_joint_pos,
      ProprioNoiseCfg(
        joint_pos=cfg.sensor_noise_joint_pos,
        joint_vel=cfg.sensor_noise_joint_vel,
        base_ang_vel=cfg.sensor_noise_base_ang_vel,
        projected_gravity=cfg.sensor_noise_projected_gravity,
      ),
    )
    self.proprio_noise_bounds = self.proprio.noise.bounds(self.spec.num_joints).to(
      self.device
    )
    self.history_len = cfg.obs_history
    self.obs_dim = (
      self.spec.num_joints * 3 + 6
      if cfg.actor_obs_mode == "proprio"
      else self.representation.rep_state_dim
    )
    if cfg.actor_obs_mode == "proprio" and self.obs_dim != self.proprio.dim:
      raise ValueError("proprio dimension mismatch")
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

    action_cfg = env.cfg.actions["joint_pos"]
    action_scale = torch.as_tensor(
      action_cfg.scale, dtype=torch.float32, device=self.device
    ).flatten()
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
        history_len=self.history_len,
        goal_dim=self.actor_goal_dim,
        wm_goal_dim=self.wm_goal_dim,
        rep_state_dim=self.representation.rep_state_dim,
        context_dim=self.spec.context_dim,
        action_dim=self.spec.action_dim,
        latent_dim=cfg.latent_dim,
        encoder_hidden=cfg.encoder_hidden_dims,
        decoder_hidden=cfg.decoder_hidden_dims,
        critic_hidden=cfg.critic_hidden_dims,
        max_step_reward=0.1,
        gamma=cfg.gamma,
      )
    ).to(self.device)

    self.wm_optimizers = build_optimizers(
      [self.world_model], cfg.wm_lr, use_muon=cfg.use_muon
    )
    self.actor_params = list(self.actor_critic.encoder.parameters()) + list(
      self.actor_critic.decoder.parameters()
    )
    self.actor_optimizers = build_optimizers(
      [self.actor_critic.encoder, self.actor_critic.decoder],
      cfg.actor_lr,
      betas=(0.9, 0.95),
      use_muon=cfg.use_muon,
    )
    self.critic_optimizers = build_optimizers(
      [self.actor_critic.critic], cfg.critic_lr, use_muon=cfg.use_muon
    )

    self.history = torch.zeros(
      env.num_envs, self.history_len, self.obs_dim, device=self.device
    )
    self.last_action = torch.zeros(
      env.num_envs, self.spec.action_dim, device=self.device
    )

    max_episode_steps = int(math.ceil(env.max_episode_length_s / env.step_dt)) + 2
    self.episode_len = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
    self.episode_return = torch.zeros(env.num_envs, device=self.device)
    self.episode_length = torch.zeros(
      env.num_envs, dtype=torch.long, device=self.device
    )
    self.reward_term_names = list(TRACKING_REWARD_TERMS)
    self._term_sums = torch.zeros(len(self.reward_term_names), device=self.device)
    self._term_steps = 0
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
  def actor_obs(
    self,
    full_state: torch.Tensor,
    last_action: torch.Tensor,
    context: torch.Tensor,
    *,
    noisy: bool,
  ) -> torch.Tensor:
    """One actor frame: the 68-dim rep state or the 93-dim sensor proprio."""
    if self.cfg.actor_obs_mode == "rep_state":
      return self.representation.represent(full_state)
    return self.proprio(
      full_state,
      last_action,
      context_encoder_bias(context, self.spec.num_joints),
      add_noise=noisy,
    )

  def push_history(
    self, history: torch.Tensor, obs: torch.Tensor
  ) -> torch.Tensor:
    if self.history_len == 1:
      return obs[:, None, :]
    return torch.cat([history[:, 1:], obs[:, None]], dim=1)

  def initialize(self) -> None:
    self.env.reset()
    state = self.extractor.state()
    context = self.extractor.context()
    obs = self.actor_obs(state, self.last_action, context, noisy=False)
    self.history[:] = obs[:, None, :].expand_as(self.history)
    self.episode_len.zero_()
    self.episode_return.zero_()
    self.episode_length.zero_()
    self._completed_returns.clear()
    self._completed_lengths.clear()
    self._completed_success.clear()
    self._term_sums.zero_()
    self._term_steps = 0
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
      obs = self.actor_obs(state, self.last_action, context, noisy=True)
      self.history = self.push_history(self.history, obs)
      wm_goal, actor_goal = self.command.goal()
      self.actor_critic.update_normalizers(self.history, actor_goal)
      action = self.actor_critic.act(self.history, actor_goal)
      noise = torch.randn(
        action.shape, generator=self._rng, device=self.device, dtype=action.dtype
      )
      noisy_action = action + noise * self.cfg.action_noise
      clip_ids = self.command.clip_ids.clone()
      time_steps = self.command.time_steps.clone()
      pre_frames = self.command.global_frame_index(clip_ids, time_steps)
      _, _, terminated, truncated, _ = self.env.step(noisy_action)
      next_state = self.extractor.state()
      reference = _squeeze(self.library.gather(clip_ids, time_steps, 1))
      reward, terms = self.reward(
        next_state, noisy_action, self.last_action, reference
      )
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
        next_state=next_state,
      )
      self.command.record_step(pre_frames, reward, terminated, truncated)
      self.env_steps += self.env.num_envs
      rollout_rewards.append(reward.mean())
      done_count += int(done.sum().item())
      self._step_reward_sum += float(reward.sum().item())
      self._step_reward_count += reward.numel()
      self._action_std_sum += float(action.std(dim=1).mean().item())
      self._action_std_steps += 1

      self.episode_return += reward
      self.episode_length += 1
      contributions = self.reward.term_contributions(terms)
      for term_index, term_name in enumerate(self.reward_term_names):
        self._term_sums[term_index] += float(contributions[term_name].mean().item())
      self._term_steps += 1

      self.last_action = noisy_action
      if bool(done.any()):
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        for env_id in done_ids.cpu().tolist():
          self._completed_returns.append(float(self.episode_return[env_id].item()))
          self._completed_lengths.append(int(self.episode_length[env_id].item()))
          self._completed_success.append(not bool(terminated[env_id].item()))
        self.env.reset(env_ids=done_ids)
        new_state = self.extractor.state()[done_ids]
        new_context = self.extractor.context()[done_ids]
        new_obs = self.actor_obs(
          new_state,
          torch.zeros_like(self.last_action[done_ids]),
          new_context,
          noisy=True,
        )
        self.history[done_ids] = new_obs[:, None, :].expand(-1, self.history_len, -1)
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
  def _update_world_model(self) -> dict[str, float]:
    cfg = self.cfg
    self._update_normalizers()
    k = cfg.wm_window
    scored = max(1, k - 1)  # official rollout stops one step short
    losses = []
    self.world_model.train()
    for _ in range(cfg.wm_updates_per_iter):
      env_ids, positions = self.buffer.sample_windows(
        cfg.wm_batch_size, k + 1, generator=self._rng
      )
      if env_ids.numel() == 0:
        continue
      states, actions, _ = self.buffer.gather_states(env_ids, positions, before=k)
      context = self.buffer.contexts[env_ids, positions]
      reps = self.representation.represent(states)
      auxs = self.representation.extract_aux_state(states)
      pred_aux = auxs[:, 0]
      pred_rep = reps[:, 0]
      loss = torch.zeros((), device=self.device)
      for j in range(scored):
        pred_aux, pred_rep = self.world_model(
          pred_aux, pred_rep, actions[:, j], context
        )
        loss = loss + self.world_model.loss(
          pred_aux, pred_rep, auxs[:, j + 1], reps[:, j + 1]
        )
      loss = loss / scored
      for optimizer in self.wm_optimizers:
        optimizer.zero_grad(set_to_none=True)
      if torch.isfinite(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          self.world_model.parameters(), cfg.wm_grad_clip
        )
        for optimizer in self.wm_optimizers:
          optimizer.step()
      losses.append(float(loss.item()))

    with torch.no_grad():
      env_ids, positions = self.buffer.sample_windows(
        cfg.wm_batch_size, k + 1, generator=self._rng
      )
      if env_ids.numel() == 0:
        window_error = 0.0
      else:
        states, actions, _ = self.buffer.gather_states(env_ids, positions, before=k)
        context = self.buffer.contexts[env_ids, positions]
        reps = self.representation.represent(states)
        auxs = self.representation.extract_aux_state(states)
        pred_aux, pred_rep = auxs[:, 0], reps[:, 0]
        errors = []
        for j in range(scored):
          pred_aux, pred_rep = self.world_model(
            pred_aux, pred_rep, actions[:, j], context
          )
          pred_blocks = self.representation.loss_blocks(pred_rep, pred_aux)
          gt_blocks = self.representation.loss_blocks(reps[:, j + 1], auxs[:, j + 1])
          errors.append(
            sum(((pred_blocks[k_] - gt_blocks[k_]) ** 2).mean() for k_ in pred_blocks)
            / len(pred_blocks)
          )
        window_error = float(torch.stack(errors).mean().sqrt().item())
    return {
      "world_model/loss": float(torch.tensor(losses).mean().item())
      if losses
      else 0.0,
      "world_model/window_error": window_error,
    }

  def _update_normalizers(self) -> None:
    if (
      self.iteration % self.cfg.norm_interval != 0
      and self.world_model.rep_norm.count.item() > 0
    ):
      return
    sample = self.buffer.sample_random(self.cfg.norm_batch_size, generator=self._rng)
    live = sample["live"]
    if int(live.sum().item()) < 64:
      return
    states = sample["state"][live]
    reps = self.representation.represent(states)
    self.world_model.update_normalizers(reps, sample["action"][live])

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
    starts["history"] = self._history_from_buffer(
      starts["env_ids"], starts["positions"], prev_action
    )
    return starts

  def _history_from_buffer(
    self,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    prev_action: torch.Tensor,
  ) -> torch.Tensor:
    n = self.history_len
    context = self.buffer.contexts[env_ids, positions]
    if self.cfg.actor_obs_mode == "rep_state":
      rep = self.representation.represent(self.buffer.states[env_ids, positions])
      return rep[:, None, :]
    states, actions, dones = self.buffer.gather_states(env_ids, positions, before=n)
    bias = context_encoder_bias(context, self.spec.num_joints)
    history = []
    for j in range(n):
      history.append(
        self.proprio(states[:, j + 1], actions[:, j], bias, add_noise=True)
      )
    history_t = torch.stack(history, dim=1)
    for k in range(n):
      mask = dones[:, k]
      if not bool(mask.any()):
        continue
      reset_obs = self.proprio(
        states[:, k + 1],
        torch.zeros_like(actions[:, k]),
        bias,
        add_noise=True,
      )
      for j in range(k + 1):
        history_t[:, j] = torch.where(mask[:, None], reset_obs, history_t[:, j])
    return history_t

  def _goal_for(
    self, clip_ids: torch.Tensor, times: torch.Tensor, aux_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    reference = self.library.gather(
      clip_ids, times, self.cfg.goal_frames, stride=self.cfg.goal_stride
    )
    return self.command.goal_features.from_reference(
      reference, aux_state[..., 0:3], aux_state[..., 3:7]
    )

  def _update_policy(self) -> dict[str, float]:
    cfg = self.cfg
    total = cfg.policy_batch_size
    starts = self._sample_imagination_starts(total)
    chunk = max(1, min(cfg.policy_chunk_size, total))

    _set_requires_grad(self.world_model.parameters(), False)
    _set_requires_grad(self.actor_critic.critic.parameters(), False)
    for optimizer in self.actor_optimizers:
      optimizer.zero_grad(set_to_none=True)

    all_aux: list[list[torch.Tensor]] = []
    all_rep: list[list[torch.Tensor]] = []
    all_goals: list[list[torch.Tensor]] = []
    all_rewards: list[list[torch.Tensor]] = []
    all_terms: list[list[torch.Tensor]] = []
    all_contexts: list[torch.Tensor] = []
    objective_value = 0.0
    reward_sum_value = 0.0
    term_prob_value = 0.0
    loss_value = 0.0
    bad_grads = 0

    for start_index in range(0, total, chunk):
      end_index = min(start_index + chunk, total)
      weight = (end_index - start_index) / total
      part = {k: v[start_index:end_index] for k, v in starts.items()}
      aux = part["aux"]
      rep = part["rep"]
      context = part["context"]
      clip_ids = part["clip_ids"]
      time_steps = part["time_steps"]
      prev_action = part["prev_action"]
      history = part["history"]

      wm_goal, actor_goal = self._goal_for(clip_ids, time_steps, aux)
      chunk_aux: list[torch.Tensor] = [aux]
      chunk_rep: list[torch.Tensor] = [rep]
      chunk_goals: list[tuple[torch.Tensor, torch.Tensor]] = [(wm_goal, actor_goal)]
      chunk_rewards: list[torch.Tensor] = []
      chunk_terms: list[torch.Tensor] = []
      times = time_steps
      with torch.enable_grad():
        for _t in range(cfg.horizon):
          action = self.actor_critic.act(history, actor_goal)
          next_aux, next_rep = self.world_model(aux, rep, action, context)
          full_state = self.representation.reconstruct_full_state(next_aux, next_rep)
          reference = _squeeze(self.library.gather(clip_ids, times, 1))
          body_pos, body_quat, body_lin_vel, body_ang_vel = self.forward_kinematics(
            full_state[:, 0:3],
            full_state[:, 3:7],
            full_state[:, 7:10],
            full_state[:, 10:13],
            full_state[:, 13 : 13 + self.spec.num_joints],
            full_state[:, 13 + self.spec.num_joints :],
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
          times = times + 1
          wm_goal, actor_goal = self._goal_for(clip_ids, times, next_aux)
          term_prob = soft_termination_prob(
            next_aux[..., 0:3],
            next_aux[..., 3:7],
            reference["body_pos"][:, self.anchor_index],
            reference["body_quat"][:, self.anchor_index],
            pos_threshold=cfg.termination_pos_threshold,
            ori_threshold=cfg.termination_ori_threshold,
            softness=cfg.termination_softness,
          )
          chunk_rewards.append(reward)
          chunk_terms.append(term_prob)
          chunk_aux.append(next_aux)
          chunk_rep.append(next_rep)
          chunk_goals.append((wm_goal, actor_goal))
          aux, rep = next_aux, next_rep
          prev_action = action
          if self.cfg.actor_obs_mode == "proprio":
            obs = self.representation.actor_sensor(
              next_rep,
              action,
              context_encoder_bias(context, self.spec.num_joints),
              noisy=True,
              noise_bounds=self.proprio_noise_bounds,
              generator=self._rng,
            )
            history = self.push_history(history, obs)
          else:
            history = next_rep[:, None, :]

        reward_sum = torch.zeros_like(chunk_rewards[0])
        survival = torch.ones_like(chunk_rewards[0])
        for t, r in enumerate(chunk_rewards):
          reward_sum = reward_sum + (cfg.gamma**t) * survival * r
          survival = survival * (1.0 - chunk_terms[t])
        wm_goal_H, _ = chunk_goals[-1]
        bootstrap = self.actor_critic.value(chunk_rep[-1], wm_goal_H, context)
        objective = reward_sum + (cfg.gamma**cfg.horizon) * survival * bootstrap
        loss = -objective.mean() * weight

      if torch.isfinite(loss):
        loss.backward()
        bad_grads += sum(
          int((~torch.isfinite(p.grad)).sum().item())
          for p in self.actor_params
          if p.grad is not None
        )
      else:
        bad_grads = -1
      objective_value += float(objective.mean().item()) * weight
      reward_sum_value += float(reward_sum.mean().item()) * weight
      term_prob_value += float(torch.stack(chunk_terms).mean().item()) * weight
      loss_value += float(loss.item())

      all_aux.append([s.detach() for s in chunk_aux])
      all_rep.append([s.detach() for s in chunk_rep])
      all_goals.append([(g[0].detach(), g[1].detach()) for g in chunk_goals])
      all_rewards.append([r.detach() for r in chunk_rewards])
      all_terms.append([d.detach() for d in chunk_terms])
      all_contexts.append(context.detach())

    if bad_grads == 0:
      torch.nn.utils.clip_grad_norm_(self.actor_params, cfg.grad_clip)
      for p in self.actor_params:
        if p.grad is not None:
          p.grad.clamp_(-cfg.grad_clip_elementwise, cfg.grad_clip_elementwise)
      for optimizer in self.actor_optimizers:
        optimizer.step()
    else:
      for optimizer in self.actor_optimizers:
        optimizer.zero_grad(set_to_none=True)

    _set_requires_grad(self.world_model.parameters(), True)
    _set_requires_grad(self.actor_critic.critic.parameters(), True)

    reps = [
      torch.cat([chunk[t] for chunk in all_rep], dim=0)
      for t in range(cfg.horizon + 1)
    ]
    wm_goals = [
      torch.cat([chunk[t][0] for chunk in all_goals], dim=0)
      for t in range(cfg.horizon + 1)
    ]
    rewards = [
      torch.cat([chunk[t] for chunk in all_rewards], dim=0)
      for t in range(cfg.horizon)
    ]
    term_probs = [
      torch.cat([chunk[t] for chunk in all_terms], dim=0)
      for t in range(cfg.horizon)
    ]
    context = torch.cat(all_contexts, dim=0)
    critic_loss = self._update_critic(reps, wm_goals, rewards, term_probs, context)
    return {
      "policy/objective": objective_value,
      "policy/loss": loss_value,
      "policy/reward": reward_sum_value,
      "policy/term_prob": term_prob_value,
      "policy/bad_grads": float(bad_grads),
      "critic/loss": critic_loss,
    }

  def _update_critic(
    self,
    reps: list[torch.Tensor],
    goals: list[torch.Tensor],
    rewards: list[torch.Tensor],
    term_probs: list[torch.Tensor],
    context: torch.Tensor,
  ) -> float:
    cfg = self.cfg
    horizon = cfg.horizon
    with torch.no_grad():
      values = [
        self.actor_critic.value(reps[t], goals[t], context)
        for t in range(horizon + 1)
      ]
      targets = [torch.zeros_like(values[0]) for _ in range(horizon)]
      running = values[-1]
      for t in range(horizon - 1, -1, -1):
        running = rewards[t] + cfg.gamma * (1.0 - term_probs[t]) * (
          (1.0 - cfg.lam) * values[t + 1] + cfg.lam * running
        )
        targets[t] = running

    rep_flat = torch.cat([reps[t] for t in range(horizon)], dim=0)
    goal_flat = torch.cat([goals[t] for t in range(horizon)], dim=0)
    target_flat = torch.cat(targets, dim=0)
    context_flat = context.repeat(horizon, 1)
    n = rep_flat.shape[0]
    losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.device)
      chunks = torch.chunk(perm, cfg.critic_minibatches)
      for idx in chunks:
        if idx.numel() == 0:
          continue
        value = self.actor_critic.value(rep_flat[idx], goal_flat[idx], context_flat[idx])
        loss = torch.mean((value - target_flat[idx]) ** 2)
        for optimizer in self.critic_optimizers:
          optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
          loss.backward()
          torch.nn.utils.clip_grad_norm_(
            self.actor_critic.critic.parameters(), cfg.grad_clip
          )
          for optimizer in self.critic_optimizers:
            optimizer.step()
        losses.append(float(loss.item()))
    return float(torch.tensor(losses).mean().item()) if losses else 0.0

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
      outputs = self.forward_kinematics(
        state[:, 0:3],
        state[:, 3:7],
        state[:, 7:10],
        state[:, 10:13],
        state[:, 13 : 13 + self.spec.num_joints],
        state[:, 13 + self.spec.num_joints :],
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
      f"window={cfg.wm_window}, actor_obs={cfg.actor_obs_mode}, "
      f"history={cfg.obs_history}.",
      flush=True,
    )
    self._train_start_time = time.time()
    self._start_iteration = self.iteration
    self._start_env_steps = self.env_steps
    self._collect(cfg.warmup_steps)
    self._precompile_kinematics()
    while self.iteration < cfg.num_iterations:
      iteration_start = time.time()
      logs: dict[str, float] = {}
      t_collect = time.perf_counter()
      self._step_accumulator += cfg.num_steps_per_env
      n_steps = int(math.floor(self._step_accumulator))
      self._step_accumulator -= n_steps
      logs.update(self._collect(n_steps))
      t_world_model = time.perf_counter()
      logs.update(self._update_world_model())
      t_policy = time.perf_counter()
      logs.update(self._update_policy())
      t_end = time.perf_counter()
      logs["time/collect"] = t_world_model - t_collect
      logs["time/world_model"] = t_policy - t_world_model
      logs["time/policy"] = t_end - t_policy
      self._set_lr(self.iteration)
      self.iteration += 1

      if cfg.log_interval > 0 and (
        self.iteration % cfg.log_interval == 0 or self.iteration == cfg.num_iterations
      ):
        self._log_iteration(logs, time.time() - iteration_start)
      if cfg.save_interval > 0 and self.iteration % cfg.save_interval == 0:
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
    term_means = (self._term_sums / max(1, self._term_steps)).cpu().tolist()

    lines = [
      f"{'#' * width}\n",
      f"{f' Learning iteration {it}/{total} ':^{width}}\n\n",
      line("Total steps", f"{self.env_steps:,}"),
      line("Steps per second", f"{fps:.0f}"),
      line("Collection time", f"{logs['time/collect']:.3f}s"),
      line("World-model time", f"{logs['time/world_model']:.3f}s"),
      line("Policy time", f"{logs['time/policy']:.3f}s"),
      line("Mean world model loss", f"{logs['world_model/loss']:.4f}"),
      line(
        "Mean world model 7-step error",
        f"{logs['world_model/window_error']:.4f}",
      ),
      line("Mean critic loss", f"{logs['critic/loss']:.4f}"),
      line("Mean policy loss", f"{logs['policy/loss']:.4f}"),
      line("Mean policy objective", f"{logs['policy/objective']:.4f}"),
      line("Mean imagined reward", f"{logs['policy/reward']:.4f}"),
      line("Mean episode reward", mean_episode_reward),
      line("Mean episode length", mean_episode_length),
      line("Mean normalized return", mean_normalized),
      line("Mean reward per step", f"{mean_step_reward:.4f}"),
      line("Mean action std", f"{mean_action_std:.4f}"),
      line("Success rate", success_rate),
      line("Done rate", f"{logs['collect/done_rate']:.4f}"),
      line("Bad gradients", f"{int(logs['policy/bad_grads'])}"),
      line("Learning rate", f"{self.actor_optimizers[0].param_groups[0]['lr']:.2e}"),
    ]
    for name, value in zip(self.reward_term_names, term_means, strict=True):
      lines.append(line(f"Episode reward {name}", f"{value:.5f}"))
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
      for name, value in zip(self.reward_term_names, term_means, strict=True):
        self._writer.add_scalar(f"episode_term/{name}", value, it)
      for name, value in command_metrics.items():
        self._writer.add_scalar(f"command/{name}", value, it)

    self._completed_returns.clear()
    self._completed_lengths.clear()
    self._completed_success.clear()
    self._term_sums.zero_()
    self._term_steps = 0
    self._action_std_sum = 0.0
    self._action_std_steps = 0
    self._step_reward_sum = 0.0
    self._step_reward_count = 0

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
          group["lr"] = base * scale

  def save(self, name: str | None = None) -> Path:
    self.log_dir.mkdir(parents=True, exist_ok=True)
    path = self.log_dir / (name or f"checkpoint_{self.iteration:07d}.pt")
    torch.save(
      {
        **stamp(
          {
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "actor_obs_mode": self.cfg.actor_obs_mode,
            "obs_history": self.cfg.obs_history,
            "actor_goal_dim": self.actor_goal_dim,
            "wm_goal_dim": self.wm_goal_dim,
          }
        ),
        "world_model": self.world_model.state_dict(),
        "actor_critic": self.actor_critic.state_dict(),
        "cfg": asdict(self.cfg),
      },
      path,
    )
    print(f"[INFO] Saved checkpoint to {path}", flush=True)
    return path

  def load(self, path: str | Path) -> None:
    checkpoint = torch.load(path, map_location=self.device)
    check_version(checkpoint.get("schema_version"), what="checkpoint")
    self.world_model.load_state_dict(checkpoint["world_model"], strict=False)
    missing, unexpected = self.actor_critic.load_state_dict(
      checkpoint["actor_critic"], strict=False
    )
    if missing or unexpected:
      print(
        f"[WARN] Loaded checkpoint with missing={len(missing)} "
        f"unexpected={len(unexpected)} keys."
      )
    self.iteration = int(checkpoint.get("iteration", 0))
    self.env_steps = int(checkpoint.get("env_steps", 0))
    print(f"[INFO] Loaded checkpoint {path} (schema v{SCHEMA_VERSION})")

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()


def _set_requires_grad(parameters, value: bool) -> list[bool]:
  parameters = list(parameters)
  previous = [p.requires_grad for p in parameters]
  for p in parameters:
    p.requires_grad_(value)
  return previous


def _squeeze(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  return {k: v[:, 0] for k, v in reference.items()}
