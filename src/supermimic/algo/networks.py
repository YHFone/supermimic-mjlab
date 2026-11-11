"""World model, actor, and critic networks (official GenCoord layout).

Everything here follows ``HybridRobotics/supermimic``:

* the dynamics network consumes ``rep(68) + action(29) + ξ(33)`` and predicts
  ``[Δv_b, Δω_b, Δvj]`` (35), integrated by
  :class:`~supermimic.algo.representation.GenCoordRepresentation`;
* MLPs are ``Linear → LayerNorm → GELU`` with a plain output layer;
* normalizers use the empirical (train-only update) rule;
* the critic reads ``rep + wm_goal + ξ`` **unnormalized** and its value is
  clamped at ``v_max = max_step_reward / (1 - γ)``;
* termination is analytic (see :mod:`supermimic.algo.reward`), so the world
  model takes no goal and has no termination head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from supermimic.algo.empirical_norm import EmpiricalNorm
from supermimic.algo.representation import GenCoordRepresentation


def _mlp(
  in_dim: int,
  hidden: tuple[int, ...],
  out_dim: int,
  activation: str = "gelu",
) -> nn.Sequential:
  act_cls = {"gelu": nn.GELU, "elu": nn.ELU, "silu": nn.SiLU}[activation]
  layers: list[nn.Module] = []
  last = in_dim
  for width in hidden:
    layers += [nn.Linear(last, width), nn.LayerNorm(width), act_cls()]
    last = width
  layers.append(nn.Linear(last, out_dim))
  return nn.Sequential(*layers)


@dataclass
class WorldModelCfg:
  context_dim: int = 33
  loss_mode: str = "paper"
  hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  activation: str = "gelu"
  # Bryson "nominal error" per channel group; the loss weights are 1/nominal².
  nominal_error_root_pos: float = 0.01
  nominal_error_root_ori: float = 0.02
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14


class WorldModel(nn.Module):
  """``(aux, rep, a, ξ) → (aux', rep')`` in generalized coordinates."""

  def __init__(
    self,
    representation: GenCoordRepresentation,
    cfg: WorldModelCfg,
    *,
    action_scale: torch.Tensor,
    action_offset: torch.Tensor,
  ) -> None:
    super().__init__()
    self.cfg = cfg
    self.rep = representation
    self.register_buffer("action_scale", action_scale.clone().float())
    self.register_buffer("action_offset", action_offset.clone().float())
    self.rep_norm = EmpiricalNorm(representation.rep_state_dim)
    self.action_norm = EmpiricalNorm(representation.num_joints)
    self.net = _mlp(
      representation.rep_state_dim
      + representation.num_joints
      + cfg.context_dim,
      cfg.hidden_dims,
      representation.prediction_dim,
      cfg.activation,
    )
    self.loss_mode = cfg.loss_mode
    self.loss_nominal = {
      "root_pos": cfg.nominal_error_root_pos,
      "root_ori": cfg.nominal_error_root_ori,
      "joint_pos": cfg.nominal_error_joint_pos,
      "joint_vel": cfg.nominal_error_joint_vel,
    }

  def proc(self, action: torch.Tensor) -> torch.Tensor:
    """Raw policy output → processed joint command (official ``Dynamics.proc``)."""
    return action * self.action_scale + self.action_offset

  def forward(
    self,
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat(
      [self.rep_norm(rep_state), self.action_norm(self.proc(action)), context], dim=-1
    )
    prediction = self.net(x)
    return self.rep.apply_prediction(aux_state, rep_state, prediction)

  @torch.no_grad()
  def update_normalizers(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
  ) -> None:
    self.rep_norm.update(rep_state)
    self.action_norm.update(self.proc(action))

  def loss(
    self,
    pred_aux: torch.Tensor,
    pred_rep: torch.Tensor,
    gt_aux: torch.Tensor,
    gt_rep: torch.Tensor,
  ) -> torch.Tensor:
    """Bryson-weighted state loss over the official channel groups.

    ``loss = Σ_g w_g n_g mse_g / Σ_g n_g`` with ``w_g = 1 / nominal_g²``.
    """
    pred_blocks = self.rep.loss_blocks(pred_rep, pred_aux)
    gt_blocks = self.rep.loss_blocks(gt_rep, gt_aux)
    total = pred_rep.new_zeros(())
    width = pred_rep.new_zeros(())
    for name, pred in pred_blocks.items():
      target = gt_blocks[name]
      mse = ((pred - target) ** 2).mean()
      n = pred.shape[-1]
      if self.loss_mode == "bryson":
        weight = 1.0 / self.loss_nominal[name] ** 2
        total = total + weight * n * mse
      else:
        total = total + n * mse
      width = width + n
    return total / width


@dataclass
class ActorCriticCfg:
  obs_dim: int = 93
  """Per-frame actor sensor width (93 for proprio, 68 for rep_state)."""
  history_len: int = 1
  goal_dim: int = 640
  """Actor goal: joint targets + anchor orientation (official ``actor_goal``)."""
  wm_goal_dim: int = 670
  """World-model goal: adds the anchor position (critic input)."""
  rep_state_dim: int = 68
  context_dim: int = 33
  action_dim: int = 29
  latent_dim: int = 32
  encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden: tuple[int, ...] = (256, 128)
  activation: str = "elu"
  max_step_reward: float = 0.1
  gamma: float = 0.99


class ActorCritic(nn.Module):
  """Deterministic goal-conditioned actor plus a state-goal critic."""

  def __init__(self, cfg: ActorCriticCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.obs_norm = EmpiricalNorm(cfg.obs_dim)
    self.goal_norm = EmpiricalNorm(cfg.goal_dim)
    self.encoder = _mlp(cfg.goal_dim, cfg.encoder_hidden, cfg.latent_dim, cfg.activation)
    self.decoder = _mlp(
      cfg.latent_dim + cfg.history_len * cfg.obs_dim,
      cfg.decoder_hidden,
      cfg.action_dim,
      cfg.activation,
    )
    self.critic = _mlp(
      cfg.rep_state_dim + cfg.wm_goal_dim + cfg.context_dim,
      cfg.critic_hidden,
      1,
      cfg.activation,
    )
    self.v_max = cfg.max_step_reward / max(1e-6, 1.0 - cfg.gamma)

  def act(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """``history`` is ``(B, H, obs_dim)`` frame-major, newest last."""
    batch = history.shape[0]
    obs = self.obs_norm(history).reshape(batch, -1)
    z = torch.tanh(self.encoder(self.goal_norm(goal)))
    return self.decoder(torch.cat([z, obs], dim=-1))

  def value(
    self, rep_state: torch.Tensor, goal: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    x = torch.cat([rep_state, goal, context], dim=-1)
    return self.critic(x).squeeze(-1).clamp(max=self.v_max)

  @torch.no_grad()
  def update_normalizers(self, history: torch.Tensor, goal: torch.Tensor) -> None:
    self.obs_norm.update(history[:, -1])
    self.goal_norm.update(goal)
