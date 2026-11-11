"""ONNX export of the deployable (proprio) actor."""

import pytest
import torch

from supermimic.algo.export import (
  ActorPolicy,
  infer_actor_spec,
)
from supermimic.algo.networks import ActorCritic, ActorCriticCfg

J = 29


def _actor(obs_dim: int, history: int) -> ActorCritic:
  return ActorCritic(
    ActorCriticCfg(
      obs_dim=obs_dim,
      history_len=history,
      goal_dim=640,
      wm_goal_dim=670,
      action_dim=J,
      latent_dim=8,
      encoder_hidden=(16,),
      decoder_hidden=(16,),
      critic_hidden=(16,),
    )
  )


def test_infer_spec_from_proprio_actor():
  actor = _actor(93, 1)
  spec = infer_actor_spec(actor.state_dict())
  assert spec.obs_dim == 93 and spec.history == 1
  assert spec.goal_dim == 640 and spec.wm_goal_dim == 670
  assert spec.rep_state_dim == 68 and spec.context_dim == 33


def test_rep_state_actor_is_not_deployable():
  actor = _actor(68, 1)
  with pytest.raises(ValueError, match="rep_state"):
    infer_actor_spec(actor.state_dict())


def test_policy_input_reshapes_history():
  actor = _actor(93, 3)
  policy = ActorPolicy(actor, 3 * 93, 640).eval()
  out = policy(torch.zeros(4, 3 * 93 + 640))
  assert out.shape == (4, J)
