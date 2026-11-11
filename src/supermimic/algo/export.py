"""Export the SuperMimic actor to ONNX for the deployment stack.

The deployment controller (``g1_deploy``) runs on an :mod:`onnxruntime` C++ runtime
and only assembles *raw* observations, so everything else travels inside the
graph::

    [obs | goal] -> action

* ``obs`` is the flattened sensor history, ``history * (3 J + 6)`` numbers in
  the official order ``g_b | omega_b | q - q0 | dq | a_{t-1}``;
* ``goal`` is the official 640-dim actor goal: ``K * 2 J`` numbers of absolute
  ``[q_ref, dq_ref]`` followed by ``K * 6`` numbers of 6D anchor orientation
  relative to the robot root (block-major, stride 5);
* the actor's empirical normalizers are baked into the graph and the encoder's
  ``tanh`` is applied as in :meth:`ActorCritic.act`.

Only ``actor_obs_mode="proprio"`` actors can be exported: the ``rep_state``
actor consumes privileged simulator state and is not deployable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .networks import ActorCritic, ActorCriticCfg

#: GenCoord representation width (1 + 3 + 3 + 3 + 2 J for J = 29).
REP_STATE_DIM = 68
#: World-model goal minus actor goal: the anchor position block.
WM_EXTRA_GOAL_DIM = 30
#: Base-pose terms per proprio frame that are not per-joint.
BASE_OBS_DIM = 6
#: Goal window constants (official ``GOAL_FRAME_STRIDE`` / ``NUM_GOAL_FRAMES``).
GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5


@dataclass(frozen=True)
class ActorSpec:
  """Shapes of a trained actor, inferred from its state dict."""

  obs_dim: int
  goal_dim: int
  action_dim: int
  history: int
  obs_frame_dim: int
  latent_dim: int
  rep_state_dim: int
  context_dim: int
  wm_goal_dim: int
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
      obs_dim=self.obs_frame_dim,
      history_len=self.history,
      goal_dim=self.goal_dim,
      wm_goal_dim=self.wm_goal_dim,
      rep_state_dim=self.rep_state_dim,
      context_dim=self.context_dim,
      action_dim=self.action_dim,
      latent_dim=self.latent_dim,
      encoder_hidden=self.encoder_hidden,
      decoder_hidden=self.decoder_hidden,
      critic_hidden=self.critic_hidden,
    )

  def describe(self) -> str:
    return (
      f"obs {self.obs_dim} = {self.history} x {self.obs_frame_dim}, "
      f"goal {self.goal_dim} (actor) / {self.wm_goal_dim} (critic), "
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
  layers = _mlp_layer_indices(state, prefix)
  return tuple(int(state[f"{prefix}{i}.weight"].shape[0]) for i in layers[:-1])


def _mlp_output_dim(state: dict[str, torch.Tensor], prefix: str) -> int:
  last = _mlp_layer_indices(state, prefix)[-1]
  return int(state[f"{prefix}{last}.weight"].shape[0])


def infer_actor_spec(state: dict[str, torch.Tensor]) -> ActorSpec:
  """Recover :class:`ActorSpec` from an ``actor_critic`` state dict."""
  obs_frame_dim = int(state["obs_norm._mean"].numel())
  latent_dim = _mlp_output_dim(state, "encoder.")
  goal_dim = int(state["encoder.0.weight"].shape[1])
  decoder_in = int(state["decoder.0.weight"].shape[1])
  obs_dim = decoder_in - latent_dim
  action_dim = _mlp_output_dim(state, "decoder.")
  if obs_dim <= 0 or obs_dim % obs_frame_dim != 0:
    raise ValueError(
      f"obs_dim {obs_dim} is not a multiple of the per-frame size {obs_frame_dim}"
    )
  history = obs_dim // obs_frame_dim
  if obs_frame_dim == REP_STATE_DIM:
    raise ValueError(
      "this actor was trained with actor_obs_mode='rep_state' and consumes "
      "privileged simulator state; retrain with actor_obs_mode='proprio' to "
      "export a deployable policy."
    )
  critic_in = int(state["critic.0.weight"].shape[1])
  wm_goal_dim = goal_dim + WM_EXTRA_GOAL_DIM
  context_dim = critic_in - REP_STATE_DIM - wm_goal_dim
  if context_dim <= 0:
    raise ValueError(
      f"critic input {critic_in} does not fit rep {REP_STATE_DIM} + "
      f"wm_goal {wm_goal_dim}"
    )
  return ActorSpec(
    obs_dim=obs_dim,
    goal_dim=goal_dim,
    action_dim=action_dim,
    history=history,
    obs_frame_dim=obs_frame_dim,
    latent_dim=latent_dim,
    rep_state_dim=REP_STATE_DIM,
    context_dim=context_dim,
    wm_goal_dim=wm_goal_dim,
    encoder_hidden=_mlp_dims(state, "encoder."),
    decoder_hidden=_mlp_dims(state, "decoder."),
    critic_hidden=_mlp_dims(state, "critic."),
  )


def load_actor(
  checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, ActorSpec, dict[str, Any]]:
  """Rebuild the trained actor from a training checkpoint."""
  from .schema import check_version

  payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
  if "actor_critic" not in payload:
    raise KeyError(f"{checkpoint} is not a SuperMimic checkpoint (no 'actor_critic')")
  check_version(payload.get("schema_version"), what="checkpoint")
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
    self.goal_dim = goal_dim
    self.history = actor.cfg.history_len
    self.frame_dim = obs_dim // self.history
    self.input_dim = obs_dim + goal_dim

  def forward(self, policy_input: torch.Tensor) -> torch.Tensor:
    obs = policy_input[:, : self.obs_dim].reshape(
      policy_input.shape[0], self.history, self.frame_dim
    )
    goal = policy_input[:, self.obs_dim :]
    return self.actor.act(obs, goal)


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
