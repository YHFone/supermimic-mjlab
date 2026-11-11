"""CUDA-graph capture for the learner updates (ported from the official code).

The official ``supermimic/learners/learner_utils.py`` wraps every learner update in
:class:`CapturedUpdate`: the first few calls run eagerly, then the call is recorded
once as a CUDA graph and replayed afterwards.  Requirements on the captured
function are strict and worth stating explicitly:

* **no host syncs** — nothing may call ``.item()``, ``.cpu()`` or read a boolean
  derived from tensor data (``if tensor.any():``);
* **fixed shapes** — every tensor shape is a Python-level constant;
* **in-place state** — parameters, buffers, optimizer state and the learning rate
  must be updated in place: the graph binds the memory addresses it recorded.

The learning rate is therefore kept as a device tensor and changed with
``group["lr"].fill_(lr)`` instead of assigning a new Python float.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable

import torch
from torch import nn


def graph_state(
  modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]
) -> list[torch.Tensor]:
  """Tensors the captured function reads that must keep their addresses."""
  state: list[torch.Tensor] = []
  for module in modules:
    state += [*module.parameters(), *module.buffers()]
  for optimizer in optimizers:
    state += [t for s in optimizer.state.values() for t in s.values()]
    state += [
      group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])
    ]
  return state


def _identity(state: list[torch.Tensor]) -> list[int]:
  return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
  return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
  """Eager warm-up, one capture, then graph replays."""

  def __init__(
    self,
    fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    state: Callable[[], list[torch.Tensor]],
    *,
    warmup: int = 3,
  ) -> None:
    self._fn = fn
    self._state = state
    self._warmup = warmup
    self._inputs: dict[str, torch.Tensor] | None = None
    self._graph: torch.cuda.CUDAGraph | None = None
    self._identity: list[int] = []
    self._outputs: dict[str, torch.Tensor] = {}
    self.captured = False

  def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = self._state()
    if not state or not state[0].is_cuda:
      return self._fn(inputs)
    device = state[0].device
    with torch.cuda.device(device):
      stream, pool = _capture_stream_and_pool(device)
      if self._inputs is None:
        self._inputs = {k: v.clone() for k, v in inputs.items()}
      else:
        for key, value in inputs.items():
          self._inputs[key].copy_(value)
      if self._warmup > 0:
        self._warmup -= 1
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
          outputs = self._fn(self._inputs)
        torch.cuda.current_stream().wait_stream(stream)
        return outputs
      if self._graph is None:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
          self._outputs = self._fn(self._inputs)
        self._graph, self._identity = graph, _identity(self._state())
        self.captured = True
      elif _identity(state) != self._identity:
        raise RuntimeError("a captured tensor moved")
      self._graph.replay()
      torch.cuda.current_stream().synchronize()
      return self._outputs
