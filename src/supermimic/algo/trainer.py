"""SuperMimic training loop.

The loop alternates between the three phases of the paper (Algorithm A.1):

1. collect transitions with the current policy in the MuJoCo Warp simulator,
2. fit the differentiable world model on short replay-buffer windows,
3. improve the actor and critic by back-propagating an imagined return
   through the frozen world model.
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

from supermimic.algo.buffer import ReplayBuffer
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.networks import (
  ActorCritic,
  ActorCriticCfg,
  WorldModel,
  WorldModelCfg,
)
from supermimic.algo.reward import (
  TRACKING_REWARD_TERMS,
  TrackingReward,
  TrackingRewardCfg,
  TrackingTermination,
)
from supermimic.algo.state import (
  StateExtractor,
  SuperMimicStateSpec,
  build_kinematics,
)
from supermimic.mdp.commands import MotionLibraryCommand
from supermimic.mdp.features import ProprioFeatures, ProprioNoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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

    self.reward = TrackingReward(
      self.kinematics,
      self.spec.tracked_body_indices,
      self.anchor_index,
      self.spec.soft_joint_pos_limits,
      TrackingRewardCfg(dt=env.step_dt),
    ).to(self.device)
    self.termination = TrackingTermination(
      self.kinematics,
      self.spec.tracked_body_indices,
      self.anchor_index,
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
    self.history_len = cfg.obs_history
    self.proprio_dim = self.proprio.dim
    self.goal_dim = self.command.goal_features.dim
    self.obs_dim = self.history_len * self.proprio_dim

    self.buffer = ReplayBuffer(
      num_envs=env.num_envs,
      capacity=cfg.buffer_steps,
      state_dim=self.spec.state_dim,
      action_dim=self.spec.action_dim,
      context_dim=self.spec.context_dim,
      device=self.device,
    )

    self.world_model = WorldModel(
      WorldModelCfg(
        state_dim=self.spec.state_dim,
        action_dim=self.spec.action_dim,
        context_dim=self.spec.context_dim,
        hidden_dims=cfg.wm_hidden_dims,
      )
    ).to(self.device)
    self.actor_critic = ActorCritic(
      ActorCriticCfg(
        obs_dim=self.obs_dim,
        goal_dim=self.goal_dim,
        state_dim=self.spec.state_dim,
        context_dim=self.spec.context_dim,
        action_dim=self.spec.action_dim,
        critic_goal_dim=self.goal_dim,
        latent_dim=cfg.latent_dim,
        encoder_hidden=cfg.encoder_hidden_dims,
        decoder_hidden=cfg.decoder_hidden_dims,
        critic_hidden=cfg.critic_hidden_dims,
      )
    ).to(self.device)

    self.wm_optimizer = torch.optim.Adam(
      self.world_model.net.parameters(), lr=cfg.wm_lr
    )
    self.actor_params = list(self.actor_critic.encoder.parameters()) + list(
      self.actor_critic.decoder.parameters()
    )
    self.actor_optimizer = torch.optim.Adam(
      self.actor_params, lr=cfg.actor_lr, betas=(0.9, 0.95)
    )
    self.critic_optimizer = torch.optim.Adam(
      self.actor_critic.critic.parameters(), lr=cfg.critic_lr
    )

    self.history = torch.zeros(
      env.num_envs, self.history_len, self.proprio_dim, device=self.device
    )
    self.last_action = torch.zeros(
      env.num_envs, self.spec.action_dim, device=self.device
    )

    max_episode_steps = int(math.ceil(env.max_episode_length_s / env.step_dt)) + 2
    self.episode_rewards = torch.zeros(
      env.num_envs, max_episode_steps, device=self.device
    )
    self.episode_len = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
    self.episode_start_clip = torch.zeros(
      env.num_envs, dtype=torch.long, device=self.device
    )
    self.episode_start_time = torch.zeros(
      env.num_envs, dtype=torch.long, device=self.device
    )

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

  def initialize(self) -> None:
    self.env.reset()
    state = self.extractor.state()
    context = self.extractor.context()
    bias = context[:, 2 : 2 + self.spec.num_joints]
    obs = self.proprio(state, self.last_action, bias, add_noise=False)
    self.history[:] = obs[:, None, :].expand_as(self.history)
    self.episode_start_clip[:] = self.command.clip_ids
    self.episode_start_time[:] = self.command.time_steps
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

  @torch.no_grad()
  def _collect(self, steps: int) -> dict[str, float]:
    if steps <= 0:
      return {"collect/reward": 0.0, "collect/done_rate": 0.0}
    rollout_rewards = []
    done_count = 0
    for _ in range(steps):
      state = self.extractor.state()
      context = self.extractor.context()
      bias = context[:, 2 : 2 + self.spec.num_joints]
      obs = self.proprio(state, self.last_action, bias, add_noise=True)
      self.history = torch.cat([self.history[:, 1:], obs[:, None]], dim=1)
      goal = self.command.goal()
      action = self.actor_critic.act(self.history.reshape(self.env.num_envs, -1), goal)
      noisy_action = action + torch.randn_like(action) * self.cfg.action_noise
      clip_ids = self.command.clip_ids.clone()
      time_steps = self.command.time_steps.clone()
      _, _, terminated, truncated, _ = self.env.step(noisy_action)
      next_state = self.extractor.state()
      reference = self.library.gather(clip_ids, time_steps, 1)
      reward, terms = self.reward(
        next_state, noisy_action, self.last_action, _squeeze(reference)
      )
      done = terminated | truncated
      self.buffer.add(state, noisy_action, reward, done, context, clip_ids, time_steps)
      self.env_steps += self.env.num_envs
      rollout_rewards.append(reward.mean())
      done_count += int(done.sum().item())
      self._step_reward_sum += float(reward.sum().item())
      self._step_reward_count += reward.numel()
      self._action_std_sum += float(action.std(dim=1).mean().item())
      self._action_std_steps += 1

      idx = self.episode_len
      self.episode_rewards[torch.arange(self.env.num_envs), idx] = reward
      self.episode_len += 1
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
        self._finish_episodes(done_ids)
        self.env.reset(env_ids=done_ids)
        new_state = self.extractor.state()[done_ids]
        new_context = self.extractor.context()[done_ids]
        new_bias = new_context[:, 2 : 2 + self.spec.num_joints]
        new_obs = self.proprio(
          new_state,
          torch.zeros_like(self.last_action[done_ids]),
          new_bias,
          add_noise=True,
        )
        self.history[done_ids] = new_obs[:, None, :].expand(-1, self.history_len, -1)
        self.last_action[done_ids] = 0.0
        self.episode_len[done_ids] = 0
        self.episode_return[done_ids] = 0.0
        self.episode_length[done_ids] = 0
        self.episode_start_clip[done_ids] = self.command.clip_ids[done_ids]
        self.episode_start_time[done_ids] = self.command.time_steps[done_ids]
    return {
      "collect/reward": float(torch.stack(rollout_rewards).mean().item())
      if rollout_rewards
      else 0.0,
      "collect/done_rate": done_count / max(1, steps * self.env.num_envs),
    }

  def _finish_episodes(self, done_ids: torch.Tensor) -> None:
    cfg = self.cfg
    bin_frames = max(1, round(self.command.sampler.bin_size * self.library.dt))
    clip_list: list[int] = []
    time_list: list[int] = []
    value_list: list[float] = []
    for env_id in done_ids.cpu().tolist():
      length = int(self.episode_len[env_id].item())
      if length == 0:
        continue
      rewards = self.episode_rewards[env_id, :length]
      disc = 0.0
      clip = int(self.episode_start_clip[env_id].item())
      start_time = int(self.episode_start_time[env_id].item())
      values = torch.zeros(length, device=self.device)
      for t in range(length - 1, -1, -1):
        disc = float(rewards[t].item()) + cfg.gamma * disc
        values[t] = disc
      for t in range(0, length, bin_frames):
        clip_list.append(clip)
        time_list.append(start_time + t)
        value_list.append(float(values[t].item()))
    if clip_list:
      self.command.update_returns(
        torch.tensor(clip_list, dtype=torch.long, device=self.device),
        torch.tensor(time_list, dtype=torch.long, device=self.device),
        torch.tensor(value_list, dtype=torch.float32, device=self.device),
      )

  def _update_world_model(self) -> dict[str, float]:
    cfg = self.cfg
    self._update_normalizers()
    k = cfg.wm_window
    losses = []
    self.world_model.train()
    for _ in range(cfg.wm_updates_per_iter):
      env_ids, positions = self.buffer.sample_windows(cfg.wm_batch_size, k + 1)
      states, actions, _ = self.buffer.gather_states(env_ids, positions, before=k)
      context = self.buffer.contexts[env_ids, positions]
      pred = states[:, 0]
      loss = torch.zeros((), device=self.device)
      for j in range(k):
        pred = self.world_model(pred, actions[:, j], context)
        loss = loss + self.world_model.state_loss(pred, states[:, j + 1])
      loss = loss / k
      self.wm_optimizer.zero_grad(set_to_none=True)
      if torch.isfinite(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          self.world_model.net.parameters(), cfg.wm_grad_clip
        )
        self.wm_optimizer.step()
      losses.append(float(loss.item()))
    with torch.no_grad():
      env_ids, positions = self.buffer.sample_windows(cfg.wm_batch_size, k + 1)
      states, actions, _ = self.buffer.gather_states(env_ids, positions, before=k)
      context = self.buffer.contexts[env_ids, positions]
      pred = states[:, 0]
      errors = []
      for j in range(k):
        pred = self.world_model(pred, actions[:, j], context)
        diff = self.world_model.state_norm.normalize(
          pred
        ) - self.world_model.state_norm.normalize(states[:, j + 1])
        errors.append(diff.pow(2).mean())
      window_error = float(torch.stack(errors).mean().sqrt().item())
    return {
      "world_model/loss": float(torch.tensor(losses).mean().item()),
      "world_model/window_error": window_error,
    }

  def _update_normalizers(self) -> None:
    if (
      self.iteration % self.cfg.norm_interval != 0
      and self.world_model.state_norm.count.item() > 0
    ):
      return
    sample = self.buffer.sample_random(self.cfg.norm_batch_size)
    live = sample["live"]
    if int(live.sum().item()) < 64:
      return
    self.world_model.update_normalizers(
      sample["state"][live], sample["action"][live], sample["context"][live]
    )

  def _sample_imagination_starts(self, batch_size: int) -> dict[str, torch.Tensor]:
    starts = self.buffer.sample_starts(batch_size, generator=self._rng)
    states, actions, dones = self.buffer.gather_states(
      starts["env_ids"], starts["positions"], before=1
    )
    prev_action = torch.where(
      dones[:, 0:1], torch.zeros_like(actions[:, 0]), actions[:, 0]
    )
    starts["prev_action"] = prev_action
    starts["history"] = self._history_from_buffer(
      starts["env_ids"], starts["positions"]
    )
    return starts

  def _history_from_buffer(
    self, env_ids: torch.Tensor, positions: torch.Tensor
  ) -> torch.Tensor:
    n = self.history_len
    states, actions, dones = self.buffer.gather_states(env_ids, positions, before=n)
    bias = self.buffer.contexts[env_ids, positions][:, 2 : 2 + self.spec.num_joints]
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

  def _update_policy(self) -> dict[str, float]:
    cfg = self.cfg
    total = cfg.policy_batch_size
    starts = self._sample_imagination_starts(total)
    chunk = max(1, min(cfg.policy_chunk_size, total))

    _set_requires_grad(self.world_model.parameters(), False)
    _set_requires_grad(self.actor_critic.critic.parameters(), False)
    self.actor_optimizer.zero_grad(set_to_none=True)

    all_states: list[list[torch.Tensor]] = []
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
      state = part["state"]
      context = part["context"]
      clip_ids = part["clip_ids"]
      time_steps = part["time_steps"]
      prev_action = part["prev_action"]
      history = part["history"]

      goal = self._goal_for(clip_ids, time_steps, state)
      chunk_states: list[torch.Tensor] = [state]
      chunk_goals: list[torch.Tensor] = [goal]
      chunk_rewards: list[torch.Tensor] = []
      chunk_terms: list[torch.Tensor] = []
      times = time_steps
      with torch.enable_grad():
        for _t in range(cfg.horizon):
          obs = self.proprio(
            state, prev_action, context[:, 2 : 2 + self.spec.num_joints]
          )
          history = torch.cat([history[:, 1:], obs[:, None]], dim=1)
          action = self.actor_critic.act(history.reshape(history.shape[0], -1), goal)
          next_state = self.world_model(state, action, context)
          reference = _squeeze(self.library.gather(clip_ids, times, 1))
          body_pos, body_quat, body_lin_vel, body_ang_vel = self.forward_kinematics(
            next_state[:, 0:3],
            next_state[:, 3:7],
            next_state[:, 7:10],
            next_state[:, 10:13],
            next_state[:, 13 : 13 + self.spec.num_joints],
            next_state[:, 13 + self.spec.num_joints :],
          )
          reward, _ = self.reward.forward_from_bodies(
            next_state,
            action,
            prev_action,
            reference,
            body_pos,
            body_quat,
            body_lin_vel,
            body_ang_vel,
          )
          _, term_prob = self.termination.forward_from_anchor(
            body_pos[:, self.anchor_index],
            body_quat[:, self.anchor_index],
            reference,
          )
          chunk_rewards.append(reward)
          chunk_terms.append(term_prob)
          chunk_states.append(next_state)
          times = times + 1
          goal = self._goal_for(clip_ids, times, next_state)
          chunk_goals.append(goal)
          state = next_state
          prev_action = action

        reward_sum = torch.zeros_like(chunk_rewards[0])
        survival = torch.ones_like(chunk_rewards[0])
        for t, r in enumerate(chunk_rewards):
          reward_sum = reward_sum + (cfg.gamma**t) * survival * r
          survival = survival * (1.0 - chunk_terms[t])
        bootstrap = self.actor_critic.value(chunk_states[-1], chunk_goals[-1], context)
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

      all_states.append([s.detach() for s in chunk_states])
      all_goals.append([g.detach() for g in chunk_goals])
      all_rewards.append([r.detach() for r in chunk_rewards])
      all_terms.append([d.detach() for d in chunk_terms])
      all_contexts.append(context.detach())

    if bad_grads == 0:
      torch.nn.utils.clip_grad_norm_(self.actor_params, cfg.grad_clip)
      for p in self.actor_params:
        if p.grad is not None:
          p.grad.clamp_(-cfg.grad_clip_elementwise, cfg.grad_clip_elementwise)
      self.actor_optimizer.step()
    else:
      self.actor_optimizer.zero_grad(set_to_none=True)

    _set_requires_grad(self.world_model.parameters(), True)
    _set_requires_grad(self.actor_critic.critic.parameters(), True)

    states = [
      torch.cat([chunk_states[t] for chunk_states in all_states], dim=0)
      for t in range(cfg.horizon + 1)
    ]
    goals = [
      torch.cat([chunk_goals[t] for chunk_goals in all_goals], dim=0)
      for t in range(cfg.horizon + 1)
    ]
    rewards = [
      torch.cat([chunk_rewards[t] for chunk_rewards in all_rewards], dim=0)
      for t in range(cfg.horizon)
    ]
    term_probs = [
      torch.cat([chunk_terms[t] for chunk_terms in all_terms], dim=0)
      for t in range(cfg.horizon)
    ]
    context = torch.cat(all_contexts, dim=0)
    critic_loss = self._update_critic(states, goals, rewards, term_probs, context)
    return {
      "policy/objective": objective_value,
      "policy/loss": loss_value,
      "policy/reward": reward_sum_value,
      "policy/term_prob": term_prob_value,
      "policy/bad_grads": float(bad_grads),
      "critic/loss": critic_loss,
    }

  def _goal_for(
    self, clip_ids: torch.Tensor, times: torch.Tensor, state: torch.Tensor
  ) -> torch.Tensor:
    reference = self.library.gather(clip_ids, times, self.cfg.goal_frames)
    return self.command.goal_features.from_reference(
      reference, state[:, 0:3], state[:, 3:7]
    )

  def _update_critic(
    self,
    states: list[torch.Tensor],
    goals: list[torch.Tensor],
    rewards: list[torch.Tensor],
    term_probs: list[torch.Tensor],
    context: torch.Tensor,
  ) -> float:
    cfg = self.cfg
    horizon = cfg.horizon
    with torch.no_grad():
      values = [
        self.actor_critic.value(states[t], goals[t], context)
        for t in range(horizon + 1)
      ]
      targets = [torch.zeros_like(values[0]) for _ in range(horizon)]
      running = values[-1]
      for t in range(horizon - 1, -1, -1):
        running = rewards[t] + cfg.gamma * (1.0 - term_probs[t]) * (
          (1.0 - cfg.lam) * values[t + 1] + cfg.lam * running
        )
        targets[t] = running

    state_flat = torch.cat([states[t] for t in range(horizon)], dim=0)
    goal_flat = torch.cat([goals[t] for t in range(horizon)], dim=0)
    target_flat = torch.cat(targets, dim=0)
    context_flat = context.repeat(horizon, 1)
    n = state_flat.shape[0]
    losses = []
    for _ in range(cfg.critic_epochs):
      perm = torch.randperm(n, device=self.device)
      chunks = torch.chunk(perm, cfg.critic_minibatches)
      for idx in chunks:
        if idx.numel() == 0:
          continue
        value = self.actor_critic.value(
          state_flat[idx], goal_flat[idx], context_flat[idx]
        )
        loss = torch.mean((value - target_flat[idx]) ** 2)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
          loss.backward()
          torch.nn.utils.clip_grad_norm_(
            self.actor_critic.critic.parameters(), cfg.grad_clip
          )
          self.critic_optimizer.step()
        losses.append(float(loss.item()))
    return float(torch.tensor(losses).mean().item()) if losses else 0.0

  def _precompile_kinematics(self) -> None:
    """Compile the FK forward and backward before the first timed iteration."""
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
      f"[INFO] SuperMimic: {self.env.num_envs} environments, "
      f"{len(self.command.library.clip_names)} clips, "
      f"horizon={cfg.horizon}, window={cfg.wm_window}.",
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

    def mean_int(values: list[int]) -> str:
      return f"{statistics.mean(values):.2f}" if values else "n/a"

    mean_episode_reward = mean(self._completed_returns)
    mean_episode_length = mean_int(self._completed_lengths)
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
      self._step_reward_sum / self._step_reward_count
      if self._step_reward_count
      else 0.0
    )
    mean_action_std = (
      self._action_std_sum / self._action_std_steps if self._action_std_steps else 0.0
    )
    term_means = (self._term_sums / max(1, self._term_steps)).cpu().tolist()

    lines = [
      f"{'#' * width}\n",
      f"{f' Learning iteration {it}/{total} ':^{width}}\n\n",
      line("Stage", "policy + critic"),
      line("Total steps", f"{self.env_steps:,}"),
      line("Steps per second", f"{fps:.0f}"),
      line("Collection time", f"{logs['time/collect']:.3f}s"),
      line("World-model time", f"{logs['time/world_model']:.3f}s"),
      line("Policy time", f"{logs['time/policy']:.3f}s"),
      line("Mean world model loss", f"{logs['world_model/loss']:.4f}"),
      line(
        "Mean world model 8-step error",
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
      line("Learning rate", f"{self.actor_optimizer.param_groups[0]['lr']:.2e}"),
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
    for optimizer, base in (
      (self.wm_optimizer, cfg.wm_lr),
      (self.actor_optimizer, cfg.actor_lr),
      (self.critic_optimizer, cfg.critic_lr),
    ):
      for group in optimizer.param_groups:
        group["lr"] = base * scale

  def save(self, name: str | None = None) -> Path:
    self.log_dir.mkdir(parents=True, exist_ok=True)
    path = self.log_dir / (name or f"checkpoint_{self.iteration:07d}.pt")
    torch.save(
      {
        "iteration": self.iteration,
        "env_steps": self.env_steps,
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
    print(f"[INFO] Loaded checkpoint {path} (iteration {self.iteration})")

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
