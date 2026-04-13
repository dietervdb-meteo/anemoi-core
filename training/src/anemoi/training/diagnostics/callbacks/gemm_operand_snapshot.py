# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""GemmOperandSnapshotCallback — save the two operands going into a targeted
Linear layer's weight-grad GEMM.

For a ``torch.nn.Linear`` layer computing ``K = X @ W^T``, the weight-gradient
GEMM is:

    dL/dW  =  (dL/dK)^T  @  X

With gradient checkpointing the mapper's ``num_chunks`` loop calls lin_key once
per destination-node chunk.  Each call contributes independently::

    dL/dW  =  sum_i  (dL/dK_i)^T  @  X_i

With gradient checkpointing enabled (the default for all mapper layers), the
recompute forward for chunk *i* fires ``_FwdInputHook`` (``fwd_inp_buf = X_i``)
immediately before the backward hook fires for that chunk.  The backward hook
reads the correctly paired ``X_i`` and appends ``(X_i, dL/dK_i)`` to
``state.chunk_pairs``.  After the full backward, all chunk tensors are
**concatenated along the node axis** before saving::

    X_cat    = cat(X_0, ..., X_{n-1})      shape: (sum_i chunk_i_src, in_features)
    dLdK_cat = cat(dL/dK_0, ..., dL/dK_n) shape: (sum_i chunk_i_src, out_features)

Because chunks partition the source–destination graph, the reconstruction
``dLdK_cat.T @ X_cat == sum_i (dL/dK_i).T @ X_i == dL/dW`` holds exactly.
The same identity applies to the FTZ simulation::

    dL/dW_ftz = ftz(dLdK_cat).T @ ftz(X_cat) == sum_i ftz(dL/dK_i).T @ ftz(X_i)

This callback captures:

  gemm_X_0     — concatenated forward input X  (shape: total_src, in_features), f32.
  gemm_dLdK_0  — concatenated upstream grad dL/dK (shape: total_src, out_features), f32,
                 **at GPU-native scale** (i.e. still multiplied by the AMP GradScaler
                 loss scale).  The loss scale is stored in ``_meta["loss_scale"]``.
                 To compare with ``param.grad`` (which is unscaled), divide by
                 ``loss_scale`` after loading.

Saving as float32 lets you:
  - Reconstruct dL/dW: wg_recon = (dLdK_0 / loss_scale).T @ X_0.
  - Simulate FTZ/bf16 at GPU scale: apply quantisation directly to dLdK_0
    (already at the correct scale), compute matmul, divide by loss_scale.
  - Measure the f16 subnormal fraction of dLdK_0 without any rescaling.
  - Compare dL/dK magnitude between DIAG and REF runs.

The file format mirrors TensorSnapshotCallback: a single .npz per step with
a JSON ``_meta`` blob.  Keys::

    <layer_name>|gemm_X_0       concatenated forward input X
    <layer_name>|gemm_dLdK_0   concatenated upstream grad dL/dK

**Requirement**: gradient checkpointing must be enabled on the targeted mapper
(``gradient_checkpointing=True``, which is the default).  With checkpointing
disabled the forward recompute does not run and ``fwd_inp_buf`` will not contain
the correct ``X_i`` when the backward hook fires.

Configuration example
---------------------
.. code-block:: yaml

    diagnostics:
      callbacks:
        - _target_: anemoi.training.diagnostics.callbacks.gemm_operand_snapshot.GemmOperandSnapshotCallback
          steps: [0, 1, 2]
          output_dir: /scratch/project_465002133/vandenbl/records/gemm_operands/my-run
          # layer_pattern: regex applied via re.search() to named_modules() paths.
          # Can match multiple layers; each gets its own keys in the .npz.
          # Default targets the encoder lin_key whose elevation we are diagnosing.
          layer_pattern: 'encoder\.data\.proc\.lin_key$'
          # sharding is inferred automatically:
          #   - modules whose name contains 'processor' (the processor blocks)
          #   - modules matching encoder.<ds>.proc.* or decoder.<ds>.proc.*
          # are treated as sharded (their source x_src is split across the
          # model_comm_group by shard_edges_1hop); all others are not sharded.

Memory note
-----------
At 1-node scale, enc.lin_key operands are ~(1183044, 1024) × 4 bytes ≈ 4.8 GB
each (x2 for X and dLdK) per step.  Use ``steps: [0]`` only when diagnosing
at step 0.  CPU RAM overhead per rank is roughly 1/model_comm_group_size of that.
"""

import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class _LayerState:
    name: str
    sharded: bool
    handles: list = field(default_factory=list)
    # Transient: set by every forward hook call, including gradient-checkpoint
    # recomputes.  The backward hook reads this immediately to form a pair.
    fwd_inp_buf: list = field(default_factory=list)
    # Accumulated (X_cpu, dLdK_cpu) pairs — one per gradient-checkpoint chunk.
    # Concatenated along dim 0 before saving; len == num_chunks after backward.
    chunk_pairs: list = field(default_factory=list)



class _FwdInputHook:
    """Captures ``inp[0]`` (= X) of a Linear forward call."""

    def __init__(self, state: _LayerState, cb) -> None:
        self.state = state
        self.cb = cb

    def __call__(self, mod, inp, output) -> None:
        if not self.cb._capturing:
            return
        self.state.fwd_inp_buf = _to_cpu(inp)


class _BwdGradHook:
    """Captures ``grad_output[0]`` (= dL/dK) of a Linear backward call.

    With gradient checkpointing the mapper recomputes the forward for chunk *i*
    (firing ``_FwdInputHook`` → ``fwd_inp_buf = X_i``) immediately before this
    hook runs for chunk *i*.  We snapshot ``fwd_inp_buf`` here to obtain the
    correctly paired ``(X_i, dL/dK_i)`` and append to ``chunk_pairs``.
    """

    def __init__(self, state: _LayerState, cb) -> None:
        self.state = state
        self.cb = cb

    def __call__(self, mod, grad_input, grad_output) -> None:
        if not self.cb._capturing:
            return
        # Snapshot fwd_inp_buf before it can be overwritten by the next chunk.
        X_snapshot = list(self.state.fwd_inp_buf)
        self.state.chunk_pairs.append((X_snapshot, _to_cpu(grad_output)))


def _to_cpu(tensors) -> list:
    """Detach to CPU float32 non-blocking, preserving None entries."""
    if tensors is None:
        return [None]
    if isinstance(tensors, torch.Tensor):
        return [tensors.detach().float().to("cpu", non_blocking=True)]
    return [
        t.detach().float().to("cpu", non_blocking=True)
        if isinstance(t, torch.Tensor)
        else None
        for t in tensors
    ]


def _gather(local: torch.Tensor, group) -> torch.Tensor:
    """Gather CPU shards along dim 0 (same pattern as TensorSnapshotCallback)."""
    if group is None or dist.get_world_size(group) <= 1:
        return local
    shards = [None] * dist.get_world_size(group)
    dist.all_gather_object(shards, local, group=group)
    return torch.cat(shards, dim=0)


class GemmOperandSnapshotCallback(pl.Callback):
    """Save the two operands of the weight-grad GEMM for a targeted Linear layer.

    Parameters
    ----------
    config : DictConfig
        Full job configuration (required by anemoi callback contract).
    steps : list[int]
        Global optimizer steps at which to capture.
    output_dir : str
        Directory for .npz output files.
    layer_pattern : str
        Regex applied via ``re.search()`` to module paths from ``named_modules()``.
        Every matching module gets its operands captured.
    """

    def __init__(
        self,
        config: DictConfig,
        steps: list,
        output_dir: str,
        layer_pattern: str = r"encoder\.data\.proc\.lin_key$",
        gather_before_save: bool = True,
    ) -> None:
        super().__init__()
        self._target_steps: set = set(int(s) for s in steps)
        self._output_dir = Path(output_dir)
        self._pattern = re.compile(layer_pattern)
        self._gather_before_save = gather_before_save
        self._states: list[_LayerState] = []
        self._capturing = False

    # ── Lightning lifecycle ───────────────────────────────────────────────────

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pl_module = pl_module
        mod_map = dict(pl_module.named_modules())
        matches = [
            name for name in mod_map if self._pattern.search(name)
        ]
        for name in matches:
            # Sharded = operands are split across model_comm_group by
            # shard_edges_1hop and must be gathered before reconstruction.
            # Covers:
            #   1. processor.proc.*  (hidden-hidden blocks)
            #   2. encoder.<ds>.proc.*  (data->hidden mapper sublayers)
            #   3. decoder.<ds>.proc.*  (hidden->data mapper sublayers)
            sharded = (
                "processor" in name
                or bool(re.search(r"(?:encoder|decoder)\.[^.]+\.proc\.", name))
            )
            self._states.append(_LayerState(name=name, sharded=sharded))
        LOGGER.info(
            "GemmOperandSnapshotCallback: matched %d layer(s): %s  "
            "target steps=%s",
            len(self._states),
            [(s.name, "sharded" if s.sharded else "local") for s in self._states],
            sorted(self._target_steps),
        )
        self._mod_map = mod_map

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        self._capturing = step in self._target_steps
        if not self._capturing:
            return
        # Register hooks JIT — zero overhead on non-capture steps.
        for state in self._states:
            mod = self._mod_map[state.name]
            state.handles = [
                mod.register_forward_hook(_FwdInputHook(state, self)),
                mod.register_full_backward_hook(_BwdGradHook(state, self)),
            ]
        LOGGER.debug(
            "GemmOperandSnapshotCallback: registered hooks for step %d", step
        )

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

        # Remove hooks immediately.
        for state in self._states:
            for h in state.handles:
                h.remove()
            state.handles = []

        # Wait for all non-blocking CPU copies to land.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        mcg = getattr(pl_module, "model_comm_group", None)
        mcg_size = getattr(pl_module, "model_comm_group_size", 1)
        global_rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        is_rank0 = global_rank == 0

        flat: dict = {}

        def _store(key: str, t: torch.Tensor) -> None:
            flat[key] = t.float().numpy()

        # Read loss_scale for _meta only — dLdK is saved at GPU-native scale
        # (NOT divided here).  The analysis script divides by loss_scale when
        # comparing against param.grad.
        loss_scale = 1.0
        pp = getattr(trainer, "precision_plugin", None)
        if pp is not None:
            sc = getattr(pp, "scaler", None)
            if sc is not None:
                loss_scale = float(sc.get_scale())
        LOGGER.debug("GemmOperandSnapshotCallback: step %d  loss_scale=%.0f", step, loss_scale)

        # Record chunk counts before clearing (needed for metadata).
        n_chunks_map = {s.name: len(s.chunk_pairs) for s in self._states}

        for state in self._states:
            nm = state.name

            # Concatenate all chunk tensors along the node axis.
            # dLdK_cat.T @ X_cat == sum_i (dLdK_i.T @ X_i) == dL/dW.
            all_X    = [t for X_list, _ in state.chunk_pairs for t in X_list    if t is not None]
            all_dLdK = [t for _, dK_list in state.chunk_pairs for t in dK_list  if t is not None]

            if self._gather_before_save:
                # All ranks participate in the collective; only rank-0 stores.
                group = mcg if state.sharded else None
                if all_X:
                    X_cat = torch.cat([_gather(t, group) if state.sharded else t for t in all_X], dim=0)
                    if is_rank0:
                        _store(f"{nm}|gemm_X_0", X_cat)
                if all_dLdK:
                    dLdK_cat = torch.cat([_gather(t, group) if state.sharded else t for t in all_dLdK], dim=0)
                    if is_rank0:
                        _store(f"{nm}|gemm_dLdK_0", dLdK_cat)
            else:
                # No gather: each rank writes its local chunk-concatenation.
                # Non-sharded layers are identical on all ranks; only rank-0 writes.
                if state.sharded or is_rank0:
                    if all_X:
                        _store(f"{nm}|gemm_X_0", torch.cat(all_X, dim=0))
                    if all_dLdK:
                        _store(f"{nm}|gemm_dLdK_0", torch.cat(all_dLdK, dim=0))

        # Clear buffers.
        for state in self._states:
            state.fwd_inp_buf = []
            state.chunk_pairs = []
        self._capturing = False

        # Determine output path. gather_before_save=True  → single gathered
        # file on rank-0 only.  gather_before_save=False → per-rank shard file.
        if self._gather_before_save:
            if not is_rank0:
                return
            out = self._output_dir / f"gemm_operands_step_{step:06d}.npz"
        else:
            out = self._output_dir / f"gemm_operands_step_{step:06d}_rank{global_rank:04d}.npz"

        if not flat:
            return

        flat["_meta"] = np.frombuffer(
            json.dumps({
                "step": step,
                "model_comm_group_size": mcg_size,
                "global_rank": global_rank,
                "gather_before_save": self._gather_before_save,
                "loss_scale": loss_scale,
                "captured_layers": [
                    {"name": s.name, "sharded": s.sharded,
                     "n_chunks": n_chunks_map[s.name]}
                    for s in self._states
                ],
                "note": (
                    "gemm_X_0   = forward input X concatenated over all gradient-checkpoint chunks. "
                    "gemm_dLdK_0 = upstream grad dL/dK at GPU-native AMP scale (NOT divided by loss_scale). "
                    "To compare with param.grad: wg_recon = (dLdK_0 / loss_scale).T @ X_0. "
                    "For FTZ/bf16 simulation: apply quantisation directly to dLdK_0 (already at GPU scale), "
                    "compute matmul, divide by loss_scale. "
                    "All arrays stored as float32. "
                    "Per-rank files (gather_before_save=False): cat across ranks before reconstructing."
                ),
            }).encode(),
            dtype=np.uint8,
        )

        np.savez_compressed(out, **flat)
        LOGGER.info(
            "GemmOperandSnapshotCallback: step %d — %d arrays saved to %s",
            step,
            len(flat) - 1,
            out,
        )

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        for state in self._states:
            for h in state.handles:
                h.remove()
            state.handles = []
        LOGGER.info("GemmOperandSnapshotCallback: hooks removed.")
