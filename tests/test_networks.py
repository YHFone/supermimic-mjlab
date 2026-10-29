"""Shape, normalization, and fitting checks for the SuperMimic networks."""

import torch

from supermimic.algo.networks import (
  ActorCritic,
  ActorCriticCfg,
  WorldModel,
  WorldModelCfg,
)


def test_world_model_fits_linear_dynamics():
  torch.manual_seed(0)
  state_dim, action_dim, context_dim = 15, 2, 3
  cfg = WorldModelCfg(
    state_dim=state_dim,
    action_dim=action_dim,
    context_dim=context_dim,
    hidden_dims=(64, 64),
  )
  model = WorldModel(cfg)
  batch = 256
  state = torch.zeros(batch, state_dim)
  state[:, 0:3] = torch.randn(batch, 3) * 0.2
  state[:, 3:7] = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
  state[:, 7:13] = torch.randn(batch, 6) * 0.2
  state[:, 13:] = torch.randn(batch, 2) * 0.2
  action = torch.randn(batch, action_dim)
  context = torch.zeros(batch, context_dim)
  model.update_normalizers(state, action, context)
  target = state.clone()
  target[:, 0:3] += 0.1 * action[:, :1]
  target[:, 13:] += 0.05 * action
  optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-2)
  initial = None
  for step in range(60):
    prediction = model(state, action, context)
    loss = model.state_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0:
      initial = float(loss.item())
  final = float(loss.item())
  assert final < 0.5 * initial
  prediction = model(state, action, context)
  assert torch.allclose(
    torch.linalg.vector_norm(prediction[:, 3:7], dim=-1),
    torch.ones(batch),
    atol=1e-5,
  )


def test_actor_critic_shapes():
  cfg = ActorCriticCfg(
    obs_dim=40,
    goal_dim=21,
    state_dim=15,
    context_dim=3,
    action_dim=2,
    critic_goal_dim=21,
    latent_dim=4,
    encoder_hidden=(32,),
    decoder_hidden=(32,),
    critic_hidden=(16,),
  )
  model = ActorCritic(cfg)
  obs = torch.randn(5, 40)
  goal = torch.randn(5, 21)
  action = model.act(obs, goal)
  value = model.value(torch.randn(5, 15), goal, torch.randn(5, 3))
  assert action.shape == (5, 2)
  assert value.shape == (5,)
  (action.sum() + value.sum()).backward()
  assert any(p.grad is not None for p in model.parameters())


def test_actor_normalizes_obs_per_frame_and_goal():
  cfg = ActorCriticCfg(
    obs_dim=30,
    goal_dim=7,
    state_dim=15,
    context_dim=3,
    action_dim=2,
    critic_goal_dim=7,
    latent_dim=4,
    history_len=10,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    critic_hidden=(8,),
  )
  model = ActorCritic(cfg)
  # One mean/var per frame dimension, shared across the 10 history frames.
  assert model.obs_norm.mean.shape == (3,)
  assert model.goal_norm.mean.shape == (7,)
  model.obs_norm.update(torch.ones(7, 3))
  model.goal_norm.update(torch.full((7, 7), 3.0))
  normalized = model.obs_norm(torch.ones(2, 10, 3))
  assert torch.allclose(normalized, torch.zeros(2, 10, 3), atol=1e-5)
  assert torch.allclose(
    model.goal_norm(torch.full((2, 7), 3.0)), torch.zeros(2, 7), atol=1e-5
  )
  action = model.act(torch.randn(2, 30), torch.randn(2, 7))
  assert action.shape == (2, 2)


def test_critic_normalizes_its_input():
  cfg = ActorCriticCfg(
    obs_dim=30,
    goal_dim=7,
    state_dim=15,
    context_dim=3,
    action_dim=2,
    critic_goal_dim=7,
    history_len=10,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    critic_hidden=(8,),
  )
  model = ActorCritic(cfg)
  # One statistic per concatenated (state, goal, context) entry.
  assert model.critic_norm.mean.shape == (15 + 7 + 3,)
  model.critic_norm.update(torch.ones(7, 15 + 7 + 3))
  ones = torch.ones(2, 15 + 7 + 3)
  state, goal, context = ones[:, :15], ones[:, 15:22], ones[:, 22:]
  # A constant input normalizes to zero, so the critic sees only its bias.
  expected = model.critic(torch.zeros(2, 15 + 7 + 3)).squeeze(-1)
  assert torch.allclose(model.value(state, goal, context), expected, atol=1e-5)
  model.value(state, goal, context).sum().backward()
  assert any(p.grad is not None for p in model.critic.parameters())


def test_termination_head_is_a_differentiable_probability():
  cfg = WorldModelCfg(
    state_dim=15,
    action_dim=2,
    context_dim=3,
    goal_dim=7,
    hidden_dims=(8,),
    termination_hidden=(16,),
  )
  model = WorldModel(cfg)
  state = torch.randn(5, 15, requires_grad=True)
  goal = torch.randn(5, 7)
  prob = model.termination_prob(state, goal)
  assert prob.shape == (5,)
  assert bool((prob >= 0).all() and (prob <= 1).all())
  # A freshly initialized head must not output d_hat ~ 0.5, which would shrink
  # the survival product and with it the imagined horizon to a couple of steps.
  assert prob.detach().mean().item() < 0.2
  loss = model.termination_loss(
    state, goal, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
  )
  loss.backward()
  assert state.grad is not None
  assert float(state.grad.abs().sum()) > 0.0
