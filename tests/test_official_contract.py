"""Contract pins for the official GenCoord implementation.

Every number here is read off ``HybridRobotics/supermimic``: 68/7/35 GenCoord
shapes, 67-dim goal frames at stride 5 (actor 640, world model 670), analytic
soft termination, Bryson-weighted world-model loss, empirical normalization,
clamped critic, and Muon parameter groups.
"""

import pytest
import torch

from supermimic.algo.empirical_norm import EmpiricalNorm
from supermimic.algo.networks import ActorCritic, ActorCriticCfg, WorldModel, WorldModelCfg
from supermimic.algo.representation import (
  GenCoordCfg,
  GenCoordRepresentation,
  quat_integrate,
)
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


def test_quat_integration_is_world_frame_left_multiply():
  """``quat_integrate`` must left-multiply: ``R_next = R(delta) R(q)``.

  The caller feeds a *world-frame* angular velocity
  (``quat_apply(root_quat, ...)``), so right-multiplying would integrate about
  the body axes.  At a 90° yaw the two conventions differ by 1.6° per step and
  ~51° after a 32-step imagination rollout.
  """
  import torch

  from mjlab.utils.lab_api.math import matrix_from_quat, quat_mul

  def rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
      [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(-1, 3, 3)
    eye = torch.eye(3).expand_as(k)
    s = torch.sin(angle)[:, None, None]
    c = torch.cos(angle)[:, None, None]
    return eye + s * k + (1.0 - c) * (k @ k)

  half_turn = torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5])  # 90° about z
  q = half_turn.expand(1, 4)
  w_world = torch.tensor([[1.0, 0.0, 0.0]])  # world-frame, not along yaw axis
  dt = 0.02

  q_next = quat_integrate(q, w_world, dt)
  r_ref = rodrigues(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0.02])) @ (
    matrix_from_quat(q)
  )
  assert torch.allclose(matrix_from_quat(q_next), r_ref, atol=1e-5)

  # Guard the bug: right-multiplying (body-frame convention) differs by >1°.
  delta = torch.tensor(
    [[torch.cos(torch.tensor(0.01)), torch.sin(torch.tensor(0.01)), 0.0, 0.0]]
  )
  q_body_frame = quat_mul(q, delta)
  assert (matrix_from_quat(q_body_frame) - r_ref).abs().max() > 0.01


def test_actor_critic_use_gelu_like_official():
  """Official ``build_mlp`` is ``Linear → LN → GELU`` for actor *and* critic."""
  cfg = ActorCriticCfg(
    obs_dim=68,
    goal_dim=640,
    wm_goal_dim=670,
    action_dim=J,
    latent_dim=8,
    encoder_hidden=(16,),
    decoder_hidden=(16,),
    critic_hidden=(16,),
  )
  net = ActorCritic(cfg)
  for module in (net.encoder, net.decoder, net.critic):
    assert any(isinstance(m, torch.nn.GELU) for m in module.modules())
    assert not any(isinstance(m, torch.nn.ELU) for m in module.modules())


def test_critic_raw_value_is_not_clamped_but_value_is():
  """GAE uses raw V; only the bootstrap and the final targets are clamped."""
  cfg = ActorCriticCfg(
    obs_dim=68,
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
  with torch.no_grad():
    last = net.critic[-1]
    last.bias.fill_(50.0)
    last.weight.zero_()
  raw = net.value_raw(torch.zeros(2, 68), torch.zeros(2, 670), torch.zeros(2, 33))
  clamped = net.value(torch.zeros(2, 68), torch.zeros(2, 670), torch.zeros(2, 33))
  assert raw.min().item() > 10.0
  assert clamped.max().item() == pytest.approx(10.0)


def test_world_model_shapes_and_bryson_loss():
  rep = _rep()
  wm = WorldModel(
    rep,
    WorldModelCfg(context_dim=33, hidden_dims=(16,)),
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


def test_schema_version_guard():
  assert SCHEMA_VERSION == 2
  with pytest.raises(SchemaMismatchError):
    check_version(1, what="checkpoint")
  with pytest.raises(SchemaMismatchError):
    check_version(None, what="checkpoint")
  check_version(2, what="checkpoint")


def test_full_state_layout_matches_official():
  """Official full state: ``[p(3), q(4), qj(J), v(3), w(3), vj(J)]``."""
  from supermimic.algo.representation import decompose_full_state, full_state_offsets
  from supermimic.mdp.features import make_state

  idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(J)
  assert (idx_qj, idx_lin, idx_ang, idx_vj) == (7, 36, 39, 42)

  qj = torch.arange(J, dtype=torch.float32)[None]
  v = torch.full((1, 3), 1.0)
  w = torch.full((1, 3), 2.0)
  state = make_state(
    torch.zeros(1, 3),
    torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    v,
    w,
    qj,
    qj + 0.5,
  )
  assert torch.equal(state[0, idx_qj:idx_lin], qj[0])
  assert torch.equal(state[0, idx_lin:idx_ang], v[0])
  assert torch.equal(state[0, idx_ang:idx_vj], w[0])
  p, q, qj2, v2, w2, vj2 = decompose_full_state(state, J)
  assert torch.equal(qj2, qj) and torch.equal(v2, v) and torch.equal(w2, w)


def test_proprio_dim_and_noise_bounds():
  """Official actor sensor: 93 dims, uniform noise 0.05/0.2/0.01/0.5/0.0."""
  rep = _rep()
  assert rep.proprio_obs_dim(J) == 93
  bounds = rep.proprio_noise_bounds(J, "cpu")
  assert bounds.shape == (93,)
  expected = torch.cat(
    [
      torch.full((3,), 0.05),
      torch.full((3,), 0.2),
      torch.full((J,), 0.01),
      torch.full((J,), 0.5),
      torch.full((J,), 0.0),
    ]
  )
  assert torch.allclose(bounds, expected)
  obs = torch.zeros(256, 93)
  noisy = rep.add_proprio_noise(obs, J)
  assert torch.all((noisy - obs).abs() <= expected + 1e-6)
  assert float((noisy - obs).abs()[:, -J:].sum()) == 0.0


def test_history_length_above_one_is_rejected():
  from supermimic.algo.config import SuperMimicCfg

  with pytest.raises(ValueError):
    SuperMimicCfg(history_length=2)
  assert SuperMimicCfg().actor_obs_mode == "rep_state"


def test_reward_ceiling_is_derived_from_env_weights():
  """Official: ``step_dt * sum(positive weights)`` (``world_model_env.py:59-63``)."""
  from types import SimpleNamespace

  from supermimic.mdp.commands import max_step_reward_ceiling

  env = SimpleNamespace(
    step_dt=0.02,
    cfg=SimpleNamespace(
      rewards={
        "a": SimpleNamespace(weight=0.5),
        "b": SimpleNamespace(weight=1.0),
        "penalty": SimpleNamespace(weight=-2.0),
      }
    ),
  )
  assert max_step_reward_ceiling(env) == pytest.approx(0.02 * 1.5)
