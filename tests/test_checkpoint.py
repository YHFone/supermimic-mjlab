"""Official three-file checkpoint layout (``world_model.pt`` / ``actor.pt`` / ``critic.pt``)."""

import torch

from supermimic.algo import trainer as trainer_mod
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.networks import ActorCritic, ActorCriticCfg, WorldModel, WorldModelCfg
from supermimic.algo.representation import GenCoordCfg, GenCoordRepresentation

J = 29


def _representation() -> GenCoordRepresentation:
  return GenCoordRepresentation(GenCoordCfg(num_joints=J), default_joint_pos=torch.zeros(J))


def _world_model(rep: GenCoordRepresentation) -> WorldModel:
  return WorldModel(
    rep,
    WorldModelCfg(hidden_dims=(8,)),
    action_scale=torch.ones(J),
    action_offset=torch.zeros(J),
  )


def _actor_critic() -> ActorCritic:
  return ActorCritic(
    ActorCriticCfg(
      obs_dim=68,
      goal_dim=640,
      wm_goal_dim=670,
      action_dim=J,
      latent_dim=8,
      encoder_hidden=(8,),
      decoder_hidden=(8,),
      critic_hidden=(8,),
    )
  )


def _trainer(tmp_path):
  trainer = trainer_mod.SuperMimicTrainer.__new__(trainer_mod.SuperMimicTrainer)
  trainer.device = torch.device("cpu")
  trainer.cfg = SuperMimicCfg()
  trainer.log_dir = tmp_path
  trainer.iteration = 7
  trainer.env_steps = 11
  trainer.actor_goal_dim = 640
  trainer.wm_goal_dim = 670
  trainer.representation = _representation()
  trainer.world_model = _world_model(trainer.representation)
  trainer.actor_critic = _actor_critic()
  return trainer


def test_save_writes_the_official_three_files(tmp_path):
  trainer = _trainer(tmp_path)
  path = trainer.save()
  assert path == tmp_path
  for name in ("world_model.pt", "actor.pt", "critic.pt"):
    assert (tmp_path / name).is_file()
  assert not list(tmp_path.glob("checkpoint_*.pt"))


def test_directory_round_trip_restores_every_module(tmp_path):
  trainer = _trainer(tmp_path)
  reference = {
    "world_model": {
      k: v.clone() for k, v in trainer.world_model.state_dict().items()
    },
    "encoder": {
      k: v.clone() for k, v in trainer.actor_critic.encoder.state_dict().items()
    },
    "critic": {
      k: v.clone() for k, v in trainer.actor_critic.critic.state_dict().items()
    },
  }
  trainer.save()

  restored = _trainer(tmp_path / "reloaded")
  restored.log_dir = tmp_path / "reloaded"
  restored.load(tmp_path)
  for key, value in restored.world_model.state_dict().items():
    assert torch.equal(value, reference["world_model"][key])
  for key, value in restored.actor_critic.encoder.state_dict().items():
    assert torch.equal(value, reference["encoder"][key])
  for key, value in restored.actor_critic.critic.state_dict().items():
    assert torch.equal(value, reference["critic"][key])
  assert restored.iteration == 7
  assert restored.env_steps == 11
