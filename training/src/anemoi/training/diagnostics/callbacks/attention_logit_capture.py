# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""AttentionLogitCaptureCallback — save raw pre-softmax attention logits to disk.

At each configured training step, enables per-edge logit capture inside the
Triton GT kernel, waits for the next forward pass, then saves the results and
clears the capture state.

The Triton kernel (``_gt_fwd``) is extended with an ``emit_logits`` compile-time
flag and a ``LOGITS_ptr`` output buffer.  When the flag is False Triton
dead-code-eliminates the store, so there is zero overhead in normal training.

Each captured tensor is a float32 array of shape ``[M, H]``:

  M  —  number of edges in the CSC adjacency for that block
  H  —  number of attention heads (may be divided across model_comm_group ranks)

Multiple tensors are captured per forward pass — one per GT call, in the order
the model calls them (typically: decoder, encoder, then processor layers 0..N-1).

Files are saved as compressed .npz under ``output_dir``:

  attn_logits_step_{step:06d}.npz

Keys inside the npz: ``block_0``, ``block_1``, ... — in forward-call order.
A ``_meta`` key stores a JSON dict with step, num_blocks, and H.

Offline analysis (entropy):
    import numpy as np, json
    from scipy.special import softmax
    snap = np.load("attn_logits_step_007750.npz", allow_pickle=True)
    meta = json.loads(str(snap["_meta"]))
    # logits shape: [M, H];  reconstruct per-dst softmax requires edge->dst map
    # For a quick entropy proxy: treat each column (head) as a distribution
    logits = snap["block_0"]  # [M, H]
    # ... compute per-dst softmax and entropy offline using the saved edge->dst
    # mapping (row array from the CSC structure, also available in snapshots)

Configuration
-------------
.. code-block:: yaml

    diagnostics:
      callbacks:
        - _target_: anemoi.training.diagnostics.callbacks.attention_logit_capture.AttentionLogitCaptureCallback
          # Capture every this many global optimizer steps.
          every_n_steps: 50
          # Directory to save .npz files.
          output_dir: /scratch/project_465002133/vandenbl/records/attn_logits/my-run
          # Optional: only keep the first N captured blocks (e.g. 1 = decoder only).
          # Null = keep all.
          max_blocks: null
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

LOGGER = logging.getLogger(__name__)


def _get_gt_function():
    """Import GraphTransformerFunction lazily (avoids hard dep at import time)."""
    try:
        from anemoi.models.triton.gt import GraphTransformerFunction
        return GraphTransformerFunction
    except ImportError:
        return None


class AttentionLogitCaptureCallback(pl.callbacks.Callback):
    """Capture pre-softmax attention logits from the Triton GT kernel.

    Parameters
    ----------
    config : DictConfig
        Full job configuration (required by the anemoi callback contract).
    every_n_steps : int
        Capture every this many global optimizer steps.
    output_dir : str
        Directory for .npz output files.
    max_blocks : int or None
        If set, only retain the first ``max_blocks`` captured tensors per step.
        Useful to limit disk usage (e.g. ``max_blocks=1`` keeps decoder only).
    """

    def __init__(
        self,
        config: DictConfig,
        every_n_steps: int,
        output_dir: str,
        max_blocks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.output_dir = Path(output_dir)
        self.max_blocks = max_blocks
        self._gt_fn = None  # resolved lazily

    # ------------------------------------------------------------------

    def _resolve_gt(self) -> bool:
        if self._gt_fn is None:
            self._gt_fn = _get_gt_function()
        if self._gt_fn is None:
            LOGGER.warning(
                "AttentionLogitCaptureCallback: GraphTransformerFunction not "
                "available (triton not installed?). Callback is a no-op."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Arm capture flag BEFORE the forward pass that produces this step's
    # gradients.  on_train_batch_start fires before forward+backward.
    # ------------------------------------------------------------------

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if self.every_n_steps <= 0 or step % self.every_n_steps != 0:
            return
        if not self._resolve_gt():
            return
        LOGGER.info("AttentionLogitCaptureCallback: arming capture at step %d", step)
        self._gt_fn.captured_logits = []
        self._gt_fn.capture_logits = True

    # ------------------------------------------------------------------
    # Collect and save AFTER backward (before optimizer step so gradients
    # are still live, but logits were captured during forward).
    # ------------------------------------------------------------------

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: object,
    ) -> None:
        step = trainer.global_step
        if self.every_n_steps <= 0 or step % self.every_n_steps != 0:
            return
        if not self._resolve_gt():
            return

        # Disarm capture immediately.
        self._gt_fn.capture_logits = False
        captured = self._gt_fn.captured_logits[:]
        self._gt_fn.captured_logits = []

        if not captured:
            LOGGER.warning(
                "AttentionLogitCaptureCallback: no logits captured at step %d "
                "(triton backend not used, or no forward pass this step?)", step
            )
            return

        if self.max_blocks is not None:
            captured = captured[: self.max_blocks]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"attn_logits_step_{step:06d}.npz"

        arrays = {}
        for i, (buf, row) in enumerate(captured):
            arrays[f"block_{i}"] = buf.cpu().numpy()      # [M, H] float32
            arrays[f"row_{i}"]   = row.numpy().astype(np.int32)  # [M] int32 edge→dst

        meta = json.dumps({
            "step": step,
            "num_blocks": len(captured),
            "H": captured[0][0].shape[1] if captured else 0,
        })
        arrays["_meta"] = np.array(meta)

        np.savez_compressed(str(out_path), **arrays)
        LOGGER.info(
            "AttentionLogitCaptureCallback: saved %d block(s) → %s",
            len(captured), out_path,
        )
