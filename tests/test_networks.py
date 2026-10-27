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
