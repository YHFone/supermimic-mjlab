"""Export path checks: shape inference, the flat policy wrapper, and ONNX parity."""

import pytest
import torch

from supermimic.algo import export
from supermimic.algo.networks import ActorCritic, ActorCriticCfg


def _actor(num_joints: int = 3, history: int = 2, goal_frames: int = 2) -> ActorCritic:
  obs_frame = 3 * num_joints + 6
  goal_frame = 2 * num_joints + 15
  cfg = ActorCriticCfg(
    obs_dim=history * obs_frame,
    goal_dim=goal_frames * goal_frame,
    state_dim=13 + 2 * num_joints,
    context_dim=1 + num_joints + 3,
    action_dim=num_joints,
    critic_goal_dim=goal_frames * goal_frame,
    latent_dim=8,
    history_len=history,
    encoder_hidden=(16, 8),
    decoder_hidden=(24, 12),
    critic_hidden=(10,),
  )
  actor = ActorCritic(cfg)
  # A non-trivial normalizer, so the export cannot silently drop it.
  with torch.no_grad():
    actor.obs_norm.mean.fill_(0.25)
    actor.obs_norm.var.fill_(4.0)
    actor.goal_norm.mean.fill_(-0.5)
    actor.goal_norm.var.fill_(0.25)
  return actor.eval()


def test_spec_inference_matches_the_actor():
  actor = _actor()
  spec = export.infer_actor_spec(actor.state_dict())
  assert spec.num_joints == actor.cfg.action_dim
  assert spec.obs_dim == actor.cfg.obs_dim
  assert spec.goal_dim == actor.cfg.goal_dim
  assert spec.action_dim == actor.cfg.action_dim
  assert spec.history == actor.cfg.history_len
  assert spec.goal_frames == actor.cfg.goal_dim // spec.goal_frame_dim
  assert spec.goal_frame_dim == 2 * spec.num_joints + 15
  assert spec.input_dim == actor.cfg.obs_dim + actor.cfg.goal_dim
  assert spec.encoder_hidden == (16, 8)
  assert spec.decoder_hidden == (24, 12)
  assert spec.critic_hidden == (10,)


def test_spec_rejects_a_mismatched_goal_dim():
  actor = _actor()
  state = dict(actor.state_dict())
  state["encoder.0.weight"] = torch.zeros(8, actor.cfg.goal_dim + 1)
  with pytest.raises(ValueError):
    export.infer_actor_spec(state)


def test_flat_policy_matches_actor_act():
  actor = _actor()
  policy = export.ActorPolicy(actor, actor.cfg.obs_dim, actor.cfg.goal_dim)
  x = torch.randn(4, actor.cfg.obs_dim + actor.cfg.goal_dim)
  with torch.no_grad():
    expected = actor.act(x[:, : actor.cfg.obs_dim], x[:, actor.cfg.obs_dim :])
    got = policy(x)
  assert torch.allclose(got, expected, atol=1e-6)


def test_onnx_roundtrip(tmp_path):
  pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")
  actor = _actor()
  path = export.export_actor_onnx(actor, tmp_path / "policy.onnx")
  assert path.is_file()

  shapes_in, shapes_out = export.onnx_io_shapes(path)
  assert shapes_in[1] == actor.cfg.obs_dim + actor.cfg.goal_dim
  assert shapes_out[1] == actor.cfg.action_dim

  deviation = export.verify_onnx(actor, path, batch=5)
  assert deviation < 1e-4, deviation
