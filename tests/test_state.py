"""Official context layout: [friction, com(3), encoder bias(J)]."""

import torch

from supermimic.algo.state import context_encoder_bias
from supermimic.mdp.features import ProprioFeatures


def test_context_encoder_bias_starts_after_friction_and_com():
  num_joints = 3
  bias = torch.tensor([0.1, -0.02, 0.03])
  context = torch.cat([torch.tensor([0.7]), torch.zeros(3), bias])
  got = context_encoder_bias(context[None], num_joints)
  assert torch.allclose(got[0], bias)


def test_proprio_order_is_gravity_first():
  num_joints = 3
  state = torch.zeros(1, 13 + 2 * num_joints)
  state[0, 3] = 1.0
  state[0, 13 : 13 + num_joints] = torch.tensor([0.2, 0.3, 0.4])
  bias = torch.tensor([0.01, 0.02, 0.03])
  obs = ProprioFeatures(num_joints, torch.zeros(num_joints))(
    state, torch.zeros(1, num_joints), bias[None], add_noise=False
  )
  assert obs.shape == (1, 3 * num_joints + 6)
  # gravity (identity root => [0, 0, -1]), then ang_vel, then q - q0 + bias.
  assert torch.allclose(obs[0, :3], torch.tensor([0.0, 0.0, -1.0]))
  assert torch.allclose(obs[0, 6:9], torch.tensor([0.21, 0.32, 0.43]), atol=1e-6)
