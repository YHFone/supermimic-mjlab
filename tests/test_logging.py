"""Plumbing that silently dropped data: mjlab nests manager logs under "log"."""

from supermimic.algo.trainer import _episode_reward_extras


def test_episode_reward_extras_reads_mjlab_log_level():
  """mjlab writes ``Episode_Reward/<term>`` into ``extras["log"]``."""
  extras = {
    "log": {
      "Episode_Reward/tracking": 1.5,
      "Episode_Termination/time_out": 2.0,
      "Metrics/anchor_pos": 0.1,
    }
  }
  assert _episode_reward_extras(extras) == {"Episode_Reward/tracking": 1.5}


def test_episode_reward_extras_accepts_flat_dict_and_none():
  assert _episode_reward_extras(None) == {}
  assert _episode_reward_extras({}) == {}
  assert _episode_reward_extras({"Episode_Reward/x": 3.0}) == {"Episode_Reward/x": 3.0}
