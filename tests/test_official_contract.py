"""Contract pins for the official GenCoord implementation.

Every number here is read off ``HybridRobotics/supermimic``: 68/7/35 GenCoord
shapes, 67-dim goal frames at stride 5 (actor 640, world model 670), analytic
soft termination, Bryson-weighted world-model loss, empirical normalization,
clamped critic, and Muon parameter groups.
"""

import pytest
import torch

from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.empirical_norm import EmpiricalNorm
from supermimic.algo.networks import ActorCritic, ActorCriticCfg, WorldModel, WorldModelCfg
from supermimic.algo.representation import GenCoordCfg, GenCoordRepresentation
from supermimic.algo.reward import soft_termination_prob
from supermimic.algo.schema import SCHEMA_VERSION, SchemaMismatchError, check_version
from supermimic.mdp.features import GoalFeatures

J = 29


def _rep() -> GenCoordRepresentation:
  return GenCoordRepresentation(
    GenCoordCfg(num_joints=J, dt=0.02, decimation=4)
  )


def test_representation_dims():
  rep = _rep()
  assert (rep.rep_state_dim, rep.aux_state_dim, rep.prediction_dim) == (68, 7, 35)
  assert rep.full_state_dim == 71
  assert rep.substep_coeff == pytest.approx(0.625)


def test_represent_drops_world_xy_and_quat():
  rep = _rep()
  state = torch.zeros(2, 71)
  state[:, 3] = 1.0
  state[:, 0] = 3.0
  state[:, 2] = 0.9
  out = rep.represent(state)
  assert out.shape == (2, 68)
  # z is kept; horizontal position is not part of the representation.
  assert torch.allclose(out[:, 0], torch.tensor([0.9, 0.9]))
  assert not torch.isnan(out).any()


def test_roundtrip_is_exact():
  rep = _rep()
  torch.manual_seed(0)
  state = torch.zeros(16, 71)
  state[:, 3:7] = torch.nn.functional.normalize(torch.randn(16, 4), dim=-1)
  state[:, 0:3] = torch.randn(16, 3)
  state[:, 7:13] = torch.randn(16, 6)
  state[:, 13:] = torch.randn(16, 2 * J)
  aux = rep.extract_aux_state(state)
  back = rep.reconstruct_full_state(aux, rep.represent(state))
  assert torch.allclose(back, state, atol=1e-6)


def test_semi_implicit_integration():
  rep = _rep()
  dt = 0.02
  aux = torch.zeros(1, 7)
  aux[0, 3] = 1.0
  aux[0, 2] = 1.0
  rep_state = torch.zeros(1, 68)
  rep_state[0, 4] = 1.0  # v_b = +x, 1 m/s (root frame == world frame here)
  rep_state[0, 10 : 10 + J] = 0.3  # qj
  rep_state[0, 10 + J :] = 2.0  # vj
  next_aux, next_rep = rep.apply_prediction(aux, rep_state, torch.zeros(1, 35))
  # Position advances by v_b * dt; joints by vj * dt; no delta => no twist change.
  assert next_aux[0, 0].item() == pytest.approx(1.0 * dt, abs=1e-5)
  assert next_rep[0, 10].item() == pytest.approx(0.3 + 2.0 * dt, abs=1e-5)
  assert torch.allclose(next_aux[0, 3:7], aux[0, 3:7], atol=1e-6)


def test_goal_dims_and_layout():
  goal = GoalFeatures(num_joints=J, anchor_index=0)
  assert goal.wm_dim == 670
  assert goal.actor_dim == 640
  # Per-frame content is 2 J; the actor drops the 30 position dims.
  assert goal.wm_dim - goal.actor_dim == 30


def test_goal_stride_offsets():
  goal = GoalFeatures(num_joints=J, anchor_index=0)
  frames = goal.horizon
  assert frames == 10


def test_soft_termination_thresholds():
  root_pos = torch.zeros(3, 3)
  root_quat = torch.zeros(3, 4)
  root_quat[:, 0] = 1.0
  ref_pos = torch.zeros(3, 3)
  ref_pos[:, 2] = torch.tensor([0.0, 0.25, 0.30])
  ref_quat = root_quat.clone()
  p = soft_termination_prob(root_pos, root_quat, ref_pos, ref_quat)
  # Far below the threshold the probability is ~0; at it, exactly 0.5; above
  # it, sigmoid((norm - 1) / 0.05).
  assert p[0].item() < 1e-6
  assert p[1].item() == pytest.approx(0.5, abs=1e-3)
  assert p[2].item() > 0.9


def test_critic_value_is_clamped():
  cfg = ActorCriticCfg(
    obs_dim=68,
    history_len=1,
    goal_dim=640,
    wm_goal_dim=670,
    action_dim=J,
    latent_dim=8,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    critic_hidden=(16,),
    max_step_reward=0.1,
    gamma=0.99,
  )
  net = ActorCritic(cfg)
  assert net.v_max == pytest.approx(10.0)
  out = net.value(torch.zeros(4, 68), torch.zeros(4, 670), torch.zeros(4, 33))
  assert out.max().item() <= 10.0


def test_world_model_shapes_and_bryson_loss():
  rep = _rep()
  wm = WorldModel(
    rep,
    WorldModelCfg(context_dim=33, hidden_dims=(16,), loss_mode="bryson"),
    action_scale=torch.ones(J),
    action_offset=torch.zeros(J),
  )
  aux = torch.zeros(2, 7)
  aux[:, 3] = 1.0
  rep_state = torch.zeros(2, 68)
  aux2, rep2 = wm(aux, rep_state, torch.zeros(2, J), torch.zeros(2, 33))
  assert aux2.shape == (2, 7) and rep2.shape == (2, 68)
  # Root position nominal is the tightest channel (0.01), joint velocity the
  # loosest (3.14): a unit error on root_pos must cost 1/0.01**2 = 1e4.
  gt_aux = aux.clone()
  gt_aux[:, 0] += 1.0
  loss = wm.loss(aux, rep_state, gt_aux, rep_state)
  # mse over the 3-dim root_pos block is 1/3, so the block contributes
  # 1e4 * 3 * (1/3) = 1e4 to a total width of 3 + 9 + J + J.
  assert loss.item() == pytest.approx(1e4 / (3 + 9 + J + J), rel=1e-4)


def test_empirical_norm_eval_is_noop():
  norm = EmpiricalNorm(4)
  norm.train()
  norm.update(torch.randn(1024, 4) * 3.0 + 1.0)
  x = torch.randn(8, 4)
  norm.eval()
  before = norm.mean.clone(), norm.std.clone()
  norm.update(torch.randn(1024, 4) * 100.0)
  assert torch.allclose(norm.mean, before[0]) and torch.allclose(norm.std, before[1])
  assert torch.allclose(norm.inverse(norm(x)), x, atol=1e-5)


def test_history_beyond_one_requires_proprio():
  with pytest.raises(ValueError):
    SuperMimicCfg(actor_obs_mode="rep_state", obs_history=2)
  cfg = SuperMimicCfg(actor_obs_mode="proprio", obs_history=10)
  assert cfg.obs_history == 10


def test_schema_version_guard():
  assert SCHEMA_VERSION == 2
  with pytest.raises(SchemaMismatchError):
    check_version(1, what="checkpoint")
  with pytest.raises(SchemaMismatchError):
    check_version(None, what="checkpoint")
  check_version(2, what="checkpoint")
