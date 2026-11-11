"""Official context layout: [friction, com(3), encoder bias(J)]."""

import torch

from supermimic.algo.state import context_encoder_bias


def test_context_encoder_bias_starts_after_friction_and_com():
  num_joints = 3
  bias = torch.tensor([0.1, -0.02, 0.03])
  context = torch.cat([torch.tensor([0.7]), torch.zeros(3), bias])
  got = context_encoder_bias(context[None], num_joints)
  assert torch.allclose(got[0], bias)
