"""Export the SuperMimic actor to ONNX for the deployment stack.

The deployment controller (``g1_deploy``) runs on an :mod:`onnxruntime` C++ runtime
and only assembles *raw* observations, so everything else travels inside the
graph:

``[obs | goal] -> action``

* ``obs`` is the flattened proprioceptive history, ``history * (3 J + 6)``
  numbers in the order ``q - q0 | dq | omega_b | g_b | a_{t-1}``.
* ``goal`` is the flattened reference window, ``goal_frames * (2 J + 15)``
  numbers in the order ``q_ref - q0 | dq_ref | anchor_pos_ref |
  anchor_ori_root | anchor_lin_vel_ref | anchor_ang_vel_ref``.
* The observation and goal :class:`RunningNorm` statistics of the actor are
  baked into the graph, and the encoder's ``tanh`` is applied as in
  :meth:`ActorCritic.act`.

The sizes are inferred from the checkpoint, so an exported model always matches
the actor that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .networks import ActorCritic, ActorCriticCfg

# Per-frame observation terms, in the order the actor consumes them.
OBS_TERMS: tuple[str, ...] = (
  "joint_pos_rel",
  "joint_vel",
  "base_ang_vel_b",
  "projected_gravity_b",
  "last_action",
)

# Per-frame goal terms, in the order the goal encoder consumes them.
GOAL_TERMS: tuple[str, ...] = (
  "joint_pos_ref_rel",
  "joint_vel_ref",
  "anchor_pos_ref",
  "anchor_ori_root",
  "anchor_lin_vel_ref",
  "anchor_ang_vel_ref",
)

ANCHOR_POS_DIM = 3
ANCHOR_ORI_DIM = 6
ANCHOR_VEL_DIM = 6
#: ``3 + 6`` base-pose terms that are not per-joint in the observation.
BASE_OBS_DIM = 6


@dataclass(frozen=True)
class ActorSpec:
  """Shapes of a trained actor, inferred from its state dict."""

  obs_dim: int
  goal_dim: int
  action_dim: int
  history: int
  obs_frame_dim: int
  goal_frames: int
  goal_frame_dim: int
  latent_dim: int
  state_dim: int
  context_dim: int
  critic_goal_dim: int
  encoder_hidden: tuple[int, ...]
  decoder_hidden: tuple[int, ...]
  critic_hidden: tuple[int, ...]

  @property
  def num_joints(self) -> int:
    return (self.obs_frame_dim - BASE_OBS_DIM) // 3

  @property
  def input_dim(self) -> int:
    return self.obs_dim + self.goal_dim

  def actor_cfg(self) -> ActorCriticCfg:
    return ActorCriticCfg(
      obs_dim=self.obs_dim,
      goal_dim=self.goal_dim,
      state_dim=self.state_dim,
      context_dim=self.context_dim,
      action_dim=self.action_dim,
      critic_goal_dim=self.critic_goal_dim,
      latent_dim=self.latent_dim,
      history_len=self.history,
      encoder_hidden=self.encoder_hidden,
      decoder_hidden=self.decoder_hidden,
      critic_hidden=self.critic_hidden,
    )

  def describe(self) -> str:
    return (
      f"obs {self.obs_dim} = {self.history} x {self.obs_frame_dim}, "
      f"goal {self.goal_dim} = {self.goal_frames} x {self.goal_frame_dim}, "
      f"action {self.action_dim}, input {self.input_dim}"
    )


def _mlp_layer_indices(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
  layers = sorted(
    int(key[len(prefix) : -len(".weight")])
    for key in state
    if key.startswith(prefix) and key.endswith(".weight")
  )
  if not layers:
    raise ValueError(f"state dict has no {prefix} weights")
  return layers


def _mlp_dims(state: dict[str, torch.Tensor], prefix: str) -> tuple[int, ...]:
  """Hidden widths of a ``_mlp`` sequential, read from its ``Linear`` weights."""
  layers = _mlp_layer_indices(state, prefix)
  return tuple(int(state[f"{prefix}{i}.weight"].shape[0]) for i in layers[:-1])


def _mlp_output_dim(state: dict[str, torch.Tensor], prefix: str) -> int:
  last = _mlp_layer_indices(state, prefix)[-1]
  return int(state[f"{prefix}{last}.weight"].shape[0])


def infer_actor_spec(state: dict[str, torch.Tensor]) -> ActorSpec:
  """Recover :class:`ActorSpec` from an ``actor_critic`` state dict."""
  obs_frame_dim = int(state["obs_norm.mean"].numel())
  encoder_in = int(state["encoder.0.weight"].shape[1])
  latent_dim = _mlp_output_dim(state, "encoder.")
  obs_dim = int(state["decoder.0.weight"].shape[1]) - latent_dim
  action_dim = _mlp_output_dim(state, "decoder.")

  if obs_dim <= 0 or obs_dim % obs_frame_dim != 0:
    raise ValueError(
      f"obs_dim {obs_dim} is not a multiple of the per-frame size {obs_frame_dim}"
    )
  history = obs_dim // obs_frame_dim
  num_joints = (obs_frame_dim - BASE_OBS_DIM) // 3
  goal_frame_dim = 2 * num_joints + ANCHOR_POS_DIM + ANCHOR_ORI_DIM + ANCHOR_VEL_DIM
  if encoder_in % goal_frame_dim != 0:
    raise ValueError(
      f"goal_dim {encoder_in} is not a multiple of the per-frame size "
      f"{goal_frame_dim} (joints {num_joints})"
    )
  goal_frames = encoder_in // goal_frame_dim

  state_dim = 13 + 2 * num_joints
  context_dim = 1 + num_joints + 3
  critic_dim = int(state["critic_norm.mean"].numel())
  critic_goal_dim = critic_dim - state_dim - context_dim
  if critic_goal_dim != encoder_in:
    raise ValueError(
      f"critic normalizer has {critic_dim} dims, expected "
      f"{state_dim + context_dim + encoder_in}"
    )

  return ActorSpec(
    obs_dim=obs_dim,
    goal_dim=encoder_in,
    action_dim=action_dim,
    history=history,
    obs_frame_dim=obs_frame_dim,
    goal_frames=goal_frames,
    goal_frame_dim=goal_frame_dim,
    latent_dim=latent_dim,
    state_dim=state_dim,
    context_dim=context_dim,
    critic_goal_dim=critic_goal_dim,
    encoder_hidden=_mlp_dims(state, "encoder."),
    decoder_hidden=_mlp_dims(state, "decoder."),
    critic_hidden=_mlp_dims(state, "critic."),
  )


def load_actor(
  checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, ActorSpec, dict[str, Any]]:
  """Rebuild the trained actor from a training checkpoint."""
  payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
  if "actor_critic" not in payload:
    raise KeyError(f"{checkpoint} is not a SuperMimic checkpoint (no 'actor_critic')")
  state = payload["actor_critic"]
  spec = infer_actor_spec(state)
  actor = ActorCritic(spec.actor_cfg())
  missing, unexpected = actor.load_state_dict(state, strict=False)
  if missing or unexpected:
    raise RuntimeError(
      f"actor state dict mismatch: missing={list(missing)} "
      f"unexpected={list(unexpected)}"
    )
  actor.to(device).eval()
  return actor, spec, payload


class ActorPolicy(nn.Module):
  """Flat ``[obs | goal]`` policy, the deployment-side view of the actor."""

  def __init__(self, actor: ActorCritic, obs_dim: int, goal_dim: int) -> None:
    super().__init__()
    self.actor = actor
    self.obs_dim = obs_dim
    self.input_dim = obs_dim + goal_dim

  def forward(self, policy_input: torch.Tensor) -> torch.Tensor:
    return self.actor.act(
      policy_input[:, : self.obs_dim], policy_input[:, self.obs_dim :]
    )


def export_actor_onnx(
  actor: ActorCritic,
  output: str | Path,
  *,
  opset: int = 17,
  input_name: str = "policy_input",
  output_name: str = "actions",
) -> Path:
  """Write ``[obs | goal] -> action`` as a batch-1-ready ONNX graph."""
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  policy = ActorPolicy(actor, actor.cfg.obs_dim, actor.cfg.goal_dim).eval()
  sample = torch.zeros(1, policy.input_dim)
  torch.onnx.export(
    policy,
    (sample,),
    str(output),
    input_names=[input_name],
    output_names=[output_name],
    dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
    opset_version=opset,
    do_constant_folding=True,
    dynamo=False,
  )
  return output


def onnx_io_shapes(path: str | Path) -> tuple[list[int | str], list[int | str]]:
  """Input and output shapes of an exported graph (``None`` for dynamic axes)."""
  import onnx

  model = onnx.load(str(path), load_external_data=False)
  graph = model.graph

  def shape_of(value) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
      if dim.HasField("dim_value") and dim.dim_value > 0:
        dims.append(int(dim.dim_value))
      else:
        dims.append(dim.dim_param or "?")
    return dims

  return shape_of(graph.input[0]), shape_of(graph.output[0])


def verify_onnx(
  actor: ActorCritic,
  onnx_path: str | Path,
  *,
  batch: int = 8,
  seed: int = 0,
) -> float:
  """Max absolute deviation between the actor and its ONNX graph."""
  import numpy as np
  import onnxruntime as ort

  generator = torch.Generator().manual_seed(seed)
  policy = ActorPolicy(actor, actor.cfg.obs_dim, actor.cfg.goal_dim).eval()
  samples = [
    torch.zeros(batch, policy.input_dim),
    torch.randn(batch, policy.input_dim, generator=generator),
    torch.randn(batch, policy.input_dim, generator=generator) * 3.0,
  ]
  with torch.no_grad():
    reference = [policy(x).numpy() for x in samples]

  session = ort.InferenceSession(
    str(onnx_path), providers=["CPUExecutionProvider"]
  )
  input_name = session.get_inputs()[0].name
  worst = 0.0
  for sample, expected in zip(samples, reference):
    got = session.run(None, {input_name: sample.numpy()})[0]
    worst = max(worst, float(np.max(np.abs(got - expected))))
  return worst
