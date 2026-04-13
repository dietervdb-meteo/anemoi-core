# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""TensorSnapshotCallback — save full forward/backward tensors for targeted diagnostic steps.

Saves four tensor types per hooked module to a compressed .npz file:

  fwd_act_{i}          Forward activation (i-th output tensor of the module).
  upstream_grad_{i}    dL/d(module_output_i)  — grad_output from full_backward_hook.
  downstream_grad_{i}  dL/d(module_input_i)   — grad_input  from full_backward_hook.
  weight_grad_{param}  Post-allreduce weight gradient for each owned parameter.

All arrays are stored as float32.  Original dtype is recorded in _meta JSON.

Auto-discovery
--------------
No module list is needed in the config.  On the first training step the callback
scans pl_module.named_modules() and hooks every module whose name matches the
same structural patterns used by PerLayerGradientMonitor:

  * encoder (top-level aggregate)
  * encoder.<sublayer>  where sublayer in _ENC_DEC_SUBS
  * decoder (top-level aggregate)
  * decoder.<sublayer>  where sublayer in _ENC_DEC_SUBS
  * processor.proc.<N>  (one entry per processor block)
  * processor.proc.<N>.<sublayer>  where sublayer in _GT_SUBS

Sharding is inferred automatically:
  * "processor.proc." in module name -> sharded (node dim split across model_comm_group)
  * otherwise                        -> not sharded (full tensor on every rank)

For sharded modules, activation shards are buffered to CPU in the hooks and
then gathered via dist.all_gather_object() in on_before_optimizer_step (after
the backward is complete and the model_comm_group is idle). Weight gradients are
never gathered (DDP all-reduce already makes them identical on every rank).

Only model_comm_group rank-0 writes the .npz file.

File naming:  <output_dir>/tensors_step_{step:06d}.npz

Norm reconstruction
-------------------
To reconstruct a PerLayerGradientMonitor group norm from the saved file:

    import numpy as np, json
    snap = np.load("tensors_step_000001.npz", allow_pickle=True)
    # example: processor.proc.0.lin_key group (weight + bias)
    # Keys use the full Python module path as returned by named_modules(),
    # so they carry the "model." prefix, e.g.:
    #   "model.processor.proc.0.lin_key|weight_grad_weight"
    #   "model.encoder.data.proc.lin_key|weight_grad_weight"
    prefix = "model.processor.proc.0.lin_key|weight_grad_"
    keys = [k for k in snap.files if k.startswith(prefix)]
    norm = np.linalg.norm([np.linalg.norm(snap[k].ravel()) for k in keys])

Configuration example
---------------------
.. code-block:: yaml

    diagnostics:
      callbacks:
        - _target_: anemoi.training.diagnostics.callbacks.tensor_snapshot.TensorSnapshotCallback
          steps: [1, 2, 3, 10, 50, 100]
          output_dir: /scratch/project_465002133/vandenbl/records/tensor_snapshots/my-run
          save_activations: false  # set true only if you need fwd/bwd tensors (high CPU RAM cost)
          # name_filter restricts which modules are hooked when save_activations=true.
          # Applied as re.search() against the full module path from named_modules().
          # When set, bypasses the auto-discovery _classify logic and hooks ANY
          # matching module; sharding is inferred: modules with 'processor' in
          # the name are treated as sharded, all others as not sharded.
          # Use this to limit to the two boundary modules (safe CPU RAM):
          #   name_filter: 'model\.encoder$|model\.processor$'
          # Other examples:
          #   name_filter: 'encoder|decoder'   # skip processor entirely
          #   name_filter: 'proc\.(0|1)\.'     # proc blocks 0+1 sublayers only
          name_filter: null

Memory note
-----------
``save_activations=False`` (default): only weight gradients are saved.  These
are small — O(param_count × 4 bytes) — and are always safe to capture.

``save_activations=True``: also saves the forward activation and the upstream /
downstream gradient tensors for each hooked module.  For large hidden states
(~100 K nodes × 1024 channels) this can easily exceed 10–20 GB of CPU RAM per
captured step.  Use only when the node count is small, or limit ``steps`` to a
single step.
"""

import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

# Keep in sync with gradient_detail.py
_GT_SUBS: frozenset = frozenset([
    "lin_key", "lin_query", "lin_value", "lin_edge",
    "edge_pre_mlp", "node_dst_mlp", "node_src_mlp",
])
_ENC_DEC_SUBS: frozenset = _GT_SUBS | frozenset(["lin_self", "projection"])


@dataclass
class _HookState:
    name: str          # module name as returned by named_modules()
    sharded: bool      # True -> node dim is split across model_comm_group
    output_index: Optional[int] = None  # if set, only capture output[output_index] from tuples
    handles: list = field(default_factory=list)
    fwd_buf:          Optional[list] = None   # list[Tensor|None], CPU float32
    bwd_upstream_buf: Optional[list] = None
    bwd_dn_buf:       Optional[list] = None


class _ForwardHook:
    """Picklable forward hook — stores module output to state buffer."""

    def __init__(self, state: _HookState, cb) -> None:
        self.state = state
        self.cb = cb

    def __call__(self, mod, inp, output) -> None:
        if self.cb._capturing:
            idx = self.state.output_index
            if idx is not None and isinstance(output, (tuple, list)):
                output = output[idx]
            self.state.fwd_buf = TensorSnapshotCallback._to_cpu(output)


class _BackwardHook:
    """Picklable full-backward hook — stores upstream and downstream grads."""

    def __init__(self, state: _HookState, cb) -> None:
        self.state = state
        self.cb = cb

    def __call__(self, mod, grad_input, grad_output) -> None:
        if self.cb._capturing:
            idx = self.state.output_index
            go = grad_output[idx] if (idx is not None and isinstance(grad_output, (tuple, list)) and len(grad_output) > idx) else grad_output
            gi = grad_input[idx]  if (idx is not None and isinstance(grad_input,  (tuple, list)) and len(grad_input)  > idx) else grad_input
            self.state.bwd_upstream_buf = TensorSnapshotCallback._to_cpu(go)
            self.state.bwd_dn_buf       = TensorSnapshotCallback._to_cpu(gi)


class TensorSnapshotCallback(pl.Callback):
    """Save full forward/backward tensors for a targeted list of training steps.

    Parameters
    ----------
    config : DictConfig
        Full job configuration (required by anemoi callback contract).
    steps : list[int]
        Global optimizer steps at which to capture tensors.
    output_dir : str
        Directory where .npz files are written.  Created if absent.
    """

    def __init__(
        self,
        config: DictConfig,
        steps: list,
        output_dir: str,
        save_activations: bool = False,
        name_filter: str | None = None,
    ) -> None:
        super().__init__()
        assert not config.diagnostics.get("enable_checkpointing", True), (
            "TensorSnapshotCallback: set diagnostics.enable_checkpointing: False "
            "in your config — checkpointing pickle-serialises the model including "
            "registered hooks, which fails with ProcessGroup objects."
        )
        self._target_steps: set = set(int(s) for s in steps)
        self._output_dir = Path(output_dir)
        self._save_activations: bool = save_activations
        self._name_filter: re.Pattern | None = re.compile(name_filter) if name_filter else None
        self._hooks: list = []
        self._capturing: bool = False

    # ── auto-discovery ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(name: str) -> Optional[tuple]:
        """Return (sharded: bool, output_index: int | None) or None (skip).

        Matches the structural patterns used by PerLayerGradientMonitor.

        output_index selects a single tensor from a module that returns a tuple.
        encoder.{ds} and decoder.{ds} (the actual mapper modules, not the
        ModuleDict containers) return (x_src, x_dst); index 1 is the tensor
        that crosses the enc→proc and proc→dec boundary.
        """
        if not name:
            return None
        last = name.split(".")[-1]

        # processor blocks and their sublayers (node-sharded)
        if re.search(r"(?:^|\.)processor\.proc\.\d+$", name):
            return True, None
        if re.search(r"(?:^|\.)processor\.proc\.\d+\.", name) and last in _GT_SUBS:
            return True, None

        # encoder mapper (e.g. encoder.data) — NOT the ModuleDict container!
        # forward() returns (x_src_data, x_dst_hidden_sharded).
        # Capture index 1: the hidden representation going into the processor.
        if re.search(r"(?:^|\.)encoder\.\w+$", name) and "processor" not in name:
            return True, 1
        # encoder sublayers — leaf param-bearing modules (weight grads only)
        if (re.search(r"(?:^|\.)encoder\.", name)
                and "processor" not in name
                and last in _ENC_DEC_SUBS):
            return False, None

        # decoder mapper (e.g. decoder.data) — NOT the ModuleDict container!
        # forward() returns (x_latent_sharded, x_dst_data_gathered).
        # Capture index 1: the final grid prediction.
        if re.search(r"(?:^|\.)decoder\.\w+$", name) and "processor" not in name:
            return False, 1
        # decoder sublayers — leaf param-bearing modules (weight grads only)
        if (re.search(r"(?:^|\.)decoder\.", name)
                and "processor" not in name
                and last in _ENC_DEC_SUBS):
            return False, None

        return None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_cpu(tensors) -> list:
        """Detach tensors to CPU float32 non-blocking, preserving None entries.

        Uses non_blocking=True so the GPU is not stalled waiting for the copy
        to finish.  Callers must torch.cuda.synchronize() before reading the
        returned tensors.
        """
        if tensors is None:
            return [None]
        if isinstance(tensors, torch.Tensor):
            return [tensors.detach().float().to('cpu', non_blocking=True)]
        return [
            t.detach().float().to('cpu', non_blocking=True) if isinstance(t, torch.Tensor) else None
            for t in tensors
        ]

    @staticmethod
    def _gather_cpu_shard(local: torch.Tensor, group) -> torch.Tensor:
        """Gather CPU tensor shards from all ranks in group along dim 0.

        Uses all_gather_object which handles uneven shard sizes correctly
        and requires no GPU memory.  Returns the full concatenated tensor.
        """
        if group is None or dist.get_world_size(group) <= 1:
            return local
        shards = [None] * dist.get_world_size(group)
        dist.all_gather_object(shards, local, group=group)
        return torch.cat(shards, dim=0)

    @staticmethod
    def _mcg(m):       return getattr(m, "model_comm_group",      None)
    @staticmethod
    def _mcg_rank(m):  return getattr(m, "model_comm_group_rank",  0)
    @staticmethod
    def _mcg_size(m):  return getattr(m, "model_comm_group_size",  1)

    # ── lightning hooks ───────────────────────────────────────────────────────

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pl_module = pl_module  # kept for just-in-time hook registration

        # Pass 1: discover ALL weight-grad modules via _classify (always,
        # regardless of name_filter / save_activations).
        mod_map = dict(pl_module.named_modules())
        self._mod_map = mod_map
        states: dict[str, _HookState] = {}
        for name in mod_map:
            result = self._classify(name)
            if result is None:
                continue
            sharded, output_index = result
            state = _HookState(name=name, sharded=sharded, output_index=output_index)
            states[name] = state
            self._hooks.append(state)

        # Pass 2 (activation hooks): build the list of (module, state) pairs
        # that need hooks, but DO NOT register them yet.  Hooks are registered
        # just-in-time in on_train_batch_start immediately before a capture step
        # and removed right after saving.  This gives zero Python overhead on
        # every non-capturing step.
        self._act_hook_targets: list[tuple] = []  # [(module, state), ...]
        if self._save_activations:
            for name, module in mod_map.items():
                if self._name_filter is not None and not self._name_filter.search(name):
                    continue
                if name in states:
                    state = states[name]
                else:
                    if self._name_filter is None:
                        continue
                    sharded = "processor" in name
                    state = _HookState(name=name, sharded=sharded)
                    states[name] = state
                    self._hooks.append(state)
                self._act_hook_targets.append((module, state))

        LOGGER.info(
            "TensorSnapshotCallback: %d weight-grad modules  %d activation-hook candidates  "
            "save_activations=%s  name_filter=%r  (model_comm_group_size=%d)  target steps=%s",
            len(states),
            len(self._act_hook_targets),
            self._save_activations,
            self._name_filter.pattern if self._name_filter else None,
            self._mcg_size(pl_module),
            sorted(self._target_steps),
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        self._capturing = step in self._target_steps
        if self._capturing and self._save_activations:
            # Register hooks just-in-time — zero overhead on all other steps.
            for module, state in self._act_hook_targets:
                state.handles = [
                    module.register_forward_hook(_ForwardHook(state, self)),
                    module.register_full_backward_hook(_BackwardHook(state, self)),
                ]
            LOGGER.debug("TensorSnapshotCallback: registered %d activation hooks for step %d",
                         len(self._act_hook_targets), step)

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
    ) -> None:
        step = trainer.global_step
        if step not in self._target_steps:
            self._capturing = False
            return

        # Deregister activation hooks immediately — they must not persist beyond
        # this step (and we don't want the overhead on subsequent steps).
        if self._save_activations:
            for _, state in self._act_hook_targets:
                for h in state.handles:
                    h.remove()
                state.handles = []

        # Flush all pending non-blocking CPU copies before reading buffers.
        if self._save_activations and torch.cuda.is_available():
            torch.cuda.synchronize()

        mcg      = self._mcg(pl_module)
        mcg_size = self._mcg_size(pl_module)
        # Use true global rank 0 as the single writer.
        # model_comm_group_rank == 0 is a necessary condition (the gather
        # collects shards from the 8 model-parallel ranks), but with
        # data_parallel > 1 there are multiple model_comm_group_rank-0 processes
        # and they would all write to the same filename.  Global rank 0 is
        # always also model_comm_group_rank 0 (first data-parallel replica,
        # first model-parallel rank), so the gather result is correct.
        global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_rank0 = global_rank == 0

        flat: dict = {}
        meta: dict = {}

        def _store(key: str, t: torch.Tensor) -> None:
            meta[key + "|dtype"] = str(t.dtype).replace("torch.", "")
            flat[key] = t.float().numpy()

        mod_by_name = dict(pl_module.named_modules())

        for state in self._hooks:
            nm = state.name

            # forward activation
            if state.fwd_buf is not None:
                for i, t in enumerate(state.fwd_buf):
                    if t is not None:
                        full = self._gather_cpu_shard(t, mcg) if state.sharded else t
                        if is_rank0:
                            _store(f"{nm}|fwd_act_{i}", full)

            # upstream grad dL/d(output)
            if state.bwd_upstream_buf is not None:
                for i, t in enumerate(state.bwd_upstream_buf):
                    if t is not None:
                        full = self._gather_cpu_shard(t, mcg) if state.sharded else t
                        if is_rank0:
                            _store(f"{nm}|upstream_grad_{i}", full)

            # downstream grad dL/d(input)
            if state.bwd_dn_buf is not None:
                for i, t in enumerate(state.bwd_dn_buf):
                    if t is not None:
                        full = self._gather_cpu_shard(t, mcg) if state.sharded else t
                        if is_rank0:
                            _store(f"{nm}|downstream_grad_{i}", full)

            # weight grads — all-reduced by DDP, identical on all ranks, no gather
            if is_rank0:
                mod = mod_by_name.get(nm)
                if mod is not None:
                    for pname, param in mod.named_parameters(recurse=False):
                        if param.grad is not None:
                            safe = pname.replace(".", "_")
                            _store(f"{nm}|weight_grad_{safe}", param.grad.detach().cpu())

        # clear buffers before any early return
        for state in self._hooks:
            state.fwd_buf = state.bwd_upstream_buf = state.bwd_dn_buf = None
        self._capturing = False

        if not is_rank0:
            return

        flat["_meta"] = np.frombuffer(
            json.dumps({
                "step": step,
                "model_comm_group_size": mcg_size,
                "global_rank": global_rank,
                "hooked_modules": [
                    {"name": s.name, "sharded": s.sharded} for s in self._hooks
                ],
                "dtype_map": meta,
                "note_sharding": (
                    "Sharded activation tensors reconstructed from all "
                    "model_comm_group shards via all_gather_object on CPU. "
                    "Weight grads are post-DDP-allreduce and need no gathering."
                ),
                "note_norm_reconstruction": (
                    "PerLayerGradientMonitor norm for a group = "
                    "np.linalg.norm([np.linalg.norm(snap[k].ravel()) "
                    "for k in snap.files if k.startswith(group+'|weight_grad_')])"
                ),
            }).encode(),
            dtype=np.uint8,
        )

        out = self._output_dir / f"tensors_step_{step:06d}.npz"
        np.savez_compressed(out, **flat)
        LOGGER.info(
            "TensorSnapshotCallback: step %d — %d arrays saved to %s",
            step, len(flat) - 1, out,
        )

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Hooks are removed immediately after each capture step; this is just
        # a safety net in case training ends mid-capture.
        for state in self._hooks:
            for h in state.handles:
                h.remove()
            state.handles = []
        self._hooks.clear()
        LOGGER.info("TensorSnapshotCallback: hooks removed.")
