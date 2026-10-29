"""World model, actor, and critic networks for SuperMimic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from mjlab.utils.lab_api.math import quat_mul


def _mlp(
  in_dim: int,
  hidden: tuple[int, ...],
  out_dim: int,
  activation: str = "silu",
  *,
  output_gain: float = 1.0,
) -> nn.Sequential:
  act_cls = {"silu": nn.SiLU, "elu": nn.ELU, "relu": nn.ReLU}[activation]
  layers: list[nn.Module] = []
  last = in_dim
  for width in hidden:
    layers.append(nn.Linear(last, width))
    layers.append(act_cls())
    last = width
  out = nn.Linear(last, out_dim)
  if output_gain != 1.0:
    nn.init.orthogonal_(out.weight, gain=output_gain)
    nn.init.zeros_(out.bias)
  layers.append(out)
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  """Running mean/variance normalizer with optional frozen dimensions."""

  def __init__(
    self,
    dim: int,
    *,
    frozen: tuple[int, ...] = (),
    eps: float = 1e-5,
  ) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros((), dtype=torch.float64))
    frozen_mask = torch.zeros(dim, dtype=torch.bool)
    frozen_mask[list(frozen)] = True
    self.register_buffer("frozen_mask", frozen_mask)
    self.eps = eps

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.detach().float()
    if x.numel() == 0:
      return
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    batch_count = torch.tensor(float(x.shape[0]), device=x.device)
    if self.count.item() == 0.0:
      self.mean.copy_(batch_mean)
      self.var.copy_(batch_var)
      self.count.copy_(batch_count)
      return
    total = self.count + batch_count
    delta = batch_mean - self.mean
    self.mean.add_(delta * batch_count / total)
    m2 = self.var * self.count + batch_var * batch_count
    m2 = m2 + delta.pow(2) * self.count * batch_count / total
    self.var.copy_(m2 / total)
    self.count.copy_(total)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    std = (self.var + self.eps).sqrt()
    out = (x - self.mean) / std
    return torch.where(self.frozen_mask, x, out)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.normalize(x)

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    std = (self.var + self.eps).sqrt()
    out = x * std + self.mean
    return torch.where(self.frozen_mask, x, out)


def _rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
  """Exponential map from a rotation vector, smooth at the origin."""
  theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
  half = 0.5 * theta
  small = theta < 1e-4
  safe_theta = theta.clamp_min(1e-4)
  scale = torch.where(
    small,
    0.5 - theta.pow(2) / 48.0,
    torch.sin(half) / safe_theta,
  )
  quat = torch.cat([torch.cos(half), rotvec * scale], dim=-1)
  return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-6)


@dataclass
class WorldModelCfg:
  state_dim: int = 71
  action_dim: int = 29
  context_dim: int = 33
  goal_dim: int = 730
  hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  termination_hidden: tuple[int, ...] = (256, 128)
  activation: str = "silu"
  output_gain: float = 0.01


class WorldModel(nn.Module):
  """Learned residual dynamics ``s_{t+1} = f(s_t, a_t, xi)`` plus a termination head.

  The network predicts a residual for every state component except the root
  quaternion, for which it predicts an incremental rotation vector so that the
  next quaternion stays on the unit sphere.  A small head maps ``(s, g)`` to the
  logit of the termination probability ``d_hat(s, g)``; it is trained with BCE
  on the simulator's termination labels and, unlike a thresholded analytic
  test, is differentiable with respect to the state.
  """

  def __init__(self, cfg: WorldModelCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.state_norm = RunningNorm(cfg.state_dim, frozen=tuple(range(3, 7)))
    self.action_norm = RunningNorm(cfg.action_dim)
    self.context_norm = RunningNorm(cfg.context_dim)
    self.goal_norm = RunningNorm(cfg.goal_dim)
    self.num_joints = (cfg.state_dim - 13) // 2
    self.out_dim = (cfg.state_dim - 4) + 3
    self.net = _mlp(
      cfg.state_dim + cfg.action_dim + cfg.context_dim,
      cfg.hidden_dims,
      self.out_dim,
      cfg.activation,
      output_gain=cfg.output_gain,
    )
    self.term_net = _mlp(
      cfg.state_dim + cfg.goal_dim,
      cfg.termination_hidden,
      1,
      cfg.activation,
    )
    # A randomly initialized head outputs d_hat ~ 0.5, so the survival product
    # 0.5**t would cut the effective imagination horizon to a couple of steps
    # while BCE slowly moves the logit towards the (low) base rate.  Start the
    # bias at a 5% termination prior instead; the head still learns from BCE.
    nn.init.constant_(self.term_net[-1].bias, math.log(0.05 / 0.95))

  def forward(
    self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    x = torch.cat(
      [
        self.state_norm.normalize(state),
        self.action_norm.normalize(action),
        self.context_norm.normalize(context),
      ],
      dim=-1,
    )
    delta = self.net(x)
    non_quat = delta[:, : self.out_dim - 3]
    rotvec = delta[:, self.out_dim - 3 :]
    rot = _rotvec_to_quat(rotvec)
    quat = quat_mul(rot, state[:, 3:7])
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-6)

    num_joints = self.num_joints
    pos = state[:, 0:3] + non_quat[:, 0:3]
    lin_vel = state[:, 7:10] + non_quat[:, 3:6]
    ang_vel = state[:, 10:13] + non_quat[:, 6:9]
    joint_pos = state[:, 13 : 13 + num_joints] + non_quat[:, 9 : 9 + num_joints]
    joint_vel = (
      state[:, 13 + num_joints :] + non_quat[:, 9 + num_joints : 9 + 2 * num_joints]
    )
    return torch.cat([pos, quat, lin_vel, ang_vel, joint_pos, joint_vel], dim=-1)

  @torch.no_grad()
  def update_normalizers(
    self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> None:
    self.state_norm.update(state)
    self.action_norm.update(action)
    self.context_norm.update(context)

  def termination_logits(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    x = torch.cat(
      [self.state_norm.normalize(state), self.goal_norm.normalize(goal)], dim=-1
    )
    return self.term_net(x).squeeze(-1)

  def termination_prob(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Differentiable termination probability ``d_hat(s, g)`` in ``[0, 1]``."""
    return torch.sigmoid(self.termination_logits(state, goal))

  def termination_loss(
    self, state: torch.Tensor, goal: torch.Tensor, label: torch.Tensor
  ) -> torch.Tensor:
    """BCE of ``d_hat`` against the simulator's ``{0, 1}`` termination label."""
    logits = self.termination_logits(state, goal)
    return torch.nn.functional.binary_cross_entropy_with_logits(
      logits, label.float()
    )

  def state_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-aware state loss with a sign-invariant quaternion term."""
    pred_n = self.state_norm.normalize(predicted)
    target_n = self.state_norm.normalize(target)
    pred_non_quat = torch.cat([pred_n[:, 0:3], pred_n[:, 7:]], dim=-1)
    target_non_quat = torch.cat([target_n[:, 0:3], target_n[:, 7:]], dim=-1)
    loss_non_quat = torch.mean((pred_non_quat - target_non_quat) ** 2)
    pred_quat = predicted[:, 3:7]
    target_quat = target[:, 3:7]
    dot = torch.sum(pred_quat * target_quat, dim=-1, keepdim=True)
    sign = torch.where(dot >= 0, 1.0, -1.0)
    loss_quat = torch.mean((sign * pred_quat - target_quat) ** 2)
    return loss_non_quat + 4.0 * loss_quat


@dataclass
class ActorCriticCfg:
  obs_dim: int = 930
  goal_dim: int = 730
  state_dim: int = 71
  context_dim: int = 33
  action_dim: int = 29
  critic_goal_dim: int = 730
  latent_dim: int = 32
  history_len: int = 10
  encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden: tuple[int, ...] = (256, 128)
  activation: str = "elu"
  min_std: float = 1e-3


class ActorCritic(nn.Module):
  """Deterministic goal-conditioned actor plus a state-goal critic.

  The observation and goal are normalized with running statistics before the
  actor: the observation normalizer is per frame, with statistics shared across
  the history window, matching the original SuperMimic implementation.  The
  critic normalizes its ``(state, goal, context)`` input as a whole, so that
  meters, radians, and rad/s enter the first layer at comparable scales.
  """

  def __init__(self, cfg: ActorCriticCfg) -> None:
    super().__init__()
    self.cfg = cfg
    if cfg.obs_dim % cfg.history_len != 0:
      raise ValueError(
        f"obs_dim {cfg.obs_dim} is not divisible by history_len {cfg.history_len}"
      )
    self.obs_norm = RunningNorm(cfg.obs_dim // cfg.history_len)
    self.goal_norm = RunningNorm(cfg.goal_dim)
    self.critic_norm = RunningNorm(
      cfg.state_dim + cfg.critic_goal_dim + cfg.context_dim
    )
    self.encoder = _mlp(
      cfg.goal_dim, cfg.encoder_hidden, cfg.latent_dim, cfg.activation
    )
    self.decoder = _mlp(
      cfg.latent_dim + cfg.obs_dim,
      cfg.decoder_hidden,
      cfg.action_dim,
      cfg.activation,
      output_gain=0.01,
    )
    self.critic = _mlp(
      cfg.state_dim + cfg.critic_goal_dim + cfg.context_dim,
      cfg.critic_hidden,
      1,
      cfg.activation,
    )

  def act(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    batch = obs.shape[0]
    obs = obs.reshape(batch, self.cfg.history_len, -1)
    obs = self.obs_norm(obs).reshape(batch, -1)
    z = torch.tanh(self.encoder(self.goal_norm(goal)))
    return self.decoder(torch.cat([z, obs], dim=-1))

  def value(
    self, state: torch.Tensor, goal: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    x = torch.cat([state, goal, context], dim=-1)
    return self.critic(self.critic_norm(x)).squeeze(-1)
