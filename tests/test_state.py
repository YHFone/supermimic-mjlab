"""Context layout: [friction, encoder bias(J), com offset(3)]."""

import torch

from supermimic.algo.state import context_encoder_bias
from supermimic.mdp.features import ProprioFeatures


def test_context_encoder_bias_starts_after_friction():
  num_joints = 3
  bias = torch.tensor([0.1, -0.02, 0.03])
  context = torch.cat([torch.tensor([0.7]), bias, torch.tensor([0.012, -0.02, 0.03])])
  got = context_encoder_bias(context[None], num_joints)
  assert torch.allclose(got[0], bias)


def test_proprio_adds_the_true_encoder_bias():
  num_joints = 3
  default = torch.zeros(num_joints)
  state = torch.zeros(1, 13 + 2 * num_joints)
  state[0, 3] = 1.0
  state[0, 13 : 13 + num_joints] = torch.tensor([0.2, 0.3, 0.4])
  bias = torch.tensor([0.01, 0.02, 0.03])
  obs = ProprioFeatures(num_joints, default)(
    state, torch.zeros(1, num_joints), bias[None], add_noise=False
  )
  assert torch.allclose(
    obs[0, :num_joints], torch.tensor([0.21, 0.32, 0.43]), atol=1e-6
  )


def test_buffered_history_already_ends_at_the_current_observation():
  """F3: `_history_from_buffer` returns o_{t-H+1}..o_t, so the loop must not
  append o_t again at the first imagined step."""
  from types import SimpleNamespace

  from supermimic.algo.buffer import ReplayBuffer
  from supermimic.algo.trainer import SuperMimicTrainer
  from supermimic.mdp.features import ProprioFeatures

  num_joints, history_len = 2, 3
  buffer = ReplayBuffer(
    1, 16, 13 + 2 * num_joints, num_joints, 1 + num_joints + 3, "cpu"
  )
  for t in range(6):
    state = torch.zeros(1, 13 + 2 * num_joints)
    state[0, 3] = 1.0
    state[0, 13 : 13 + num_joints] = float(t)
    buffer.add(
      state=state,
      action=torch.full((1, num_joints), float(t)),
      reward=torch.zeros(1),
      done=torch.zeros(1, dtype=torch.bool),
      context=torch.zeros(1, 1 + num_joints + 3),
      clip_id=torch.zeros(1, dtype=torch.long),
      time_step=torch.full((1,), t, dtype=torch.long),
    )
  zero_noise = SimpleNamespace(
    joint_pos=0.0, joint_vel=0.0, base_ang_vel=0.0, projected_gravity=0.0
  )
  trainer = SuperMimicTrainer.__new__(SuperMimicTrainer)
  trainer.buffer = buffer
  trainer.history_len = history_len
  trainer.spec = SimpleNamespace(num_joints=num_joints)
  trainer.proprio = ProprioFeatures(num_joints, torch.zeros(num_joints), zero_noise)
  positions = torch.tensor([5])
  history = trainer._history_from_buffer(torch.tensor([0]), positions)[0]
  expected = trainer.proprio(
    buffer.states[0, 5][None], buffer.actions[0, 4][None], None, add_noise=False
  )[0]
  assert torch.allclose(history[-1], expected)
  assert not torch.allclose(history[-2], expected)
