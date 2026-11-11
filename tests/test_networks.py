"""World-model fitting and actor shapes under the GenCoord contract."""

import torch

from supermimic.algo.networks import ActorCritic, ActorCriticCfg, WorldModel, WorldModelCfg
from supermimic.algo.representation import GenCoordCfg, GenCoordRepresentation

J = 29


def test_world_model_learns_a_linear_residual():
  torch.manual_seed(0)
  rep = GenCoordRepresentation(GenCoordCfg(num_joints=J, dt=0.02, decimation=4))
  wm = WorldModel(
    rep,
    WorldModelCfg(context_dim=3, hidden_dims=(64, 64)),
    action_scale=torch.ones(J),
    action_offset=torch.zeros(J),
  )
  batch = 512
  aux = torch.zeros(batch, 7)
  aux[:, 3] = 1.0
  rep_state = torch.zeros(batch, 68)
  action = torch.randn(batch, J) * 0.1
  context = torch.zeros(batch, 3)
  # Target: the action's first three entries are the root acceleration.
  gt = rep_state.clone()
  gt[:, 4:7] = action[:, :3]
  optimizer = torch.optim.Adam(wm.net.parameters(), lr=3e-3)
  for _ in range(200):
    optimizer.zero_grad()
    loss = wm.loss(aux, rep_state, aux, gt) if False else None
    pred_aux, pred_rep = wm(aux, rep_state, action, context)
    loss = wm.loss(pred_aux, pred_rep, aux, gt)
    loss.backward()
    optimizer.step()
  assert loss.item() < 1e-2


def test_actor_shapes():
  cfg = ActorCriticCfg(
    obs_dim=93,
    history_len=10,
    goal_dim=640,
    wm_goal_dim=670,
    action_dim=J,
    latent_dim=8,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    critic_hidden=(16,),
  )
  net = ActorCritic(cfg)
  history = torch.zeros(5, 10, 93)
  action = net.act(history, torch.zeros(5, 640))
  assert action.shape == (5, J)
  value = net.value(torch.zeros(5, 68), torch.zeros(5, 670), torch.zeros(5, 33))
  assert value.shape == (5,)
