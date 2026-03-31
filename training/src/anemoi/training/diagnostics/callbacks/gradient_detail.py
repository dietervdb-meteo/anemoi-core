# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-module-group gradient-norm monitor for training-instability diagnostics."""

import logging
import re
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

LOGGER = logging.getLogger(__name__)

# Sub-module names inside a processor GraphTransformer block tracked individually.
_GT_SUBS: frozenset[str] = frozenset(
    [
        "lin_key",
        "lin_query",
        "lin_value",
        "lin_edge",
        "edge_pre_mlp",
        "node_dst_mlp",
        "node_src_mlp",
    ]
)

# Sub-module names tracked within encoder and decoder blocks.
# Superset of _GT_SUBS: encoder/decoder also have lin_self and projection.
_ENC_DEC_SUBS: frozenset[str] = _GT_SUBS | frozenset(["lin_self", "projection"])


class PerLayerGradientMonitor(pl.callbacks.Callback):
    """Per-module-group gradient norm monitor for diagnosing training instability.

    Logs gradient norms broken down by model component every ``every_n_steps``
    global optimizer steps:

    * Processor aggregate — ``processor`` (all parameters across all N blocks)
    * Each main processor block — ``proc.{N}`` (aggregate of the full block)
    * Key sub-modules within each processor block — ``proc.{N}.{sub}`` for each
      sub in ``_GT_SUBS`` that is present: ``lin_key``, ``lin_query``,
      ``lin_value``, ``lin_edge``, ``edge_pre_mlp``, ``node_dst_mlp``,
      ``node_src_mlp``
    * Encoder aggregate — ``encoder`` (all encoder parameters)
    * Encoder sub-modules — ``encoder.{sub}`` for each sub in ``_ENC_DEC_SUBS``
      that is present (superset of ``_GT_SUBS``: also ``lin_self``, ``projection``)
    * Decoder aggregate — ``decoder`` (all decoder parameters)
    * Decoder sub-modules — ``decoder.{sub}`` for each sub in ``_ENC_DEC_SUBS``
    * Global L2 norm (mirrors ``GradientMonitor`` — logged as ``train/grad_norm``)
    * AMP GradScaler scale — logged every step unconditionally when ``log_scaler``
      is True, regardless of ``every_n_steps``

    All metrics are written to the Lightning logger (MLflow) under
    ``train/glayer/{group_key}``.  Groups are built dynamically on the first
    optimizer step by scanning ``pl_module.named_parameters()``.

    Parameters
    ----------
    config : DictConfig
        Full job configuration (required by the anemoi callback contract).
    every_n_steps : int
        Log gradient norms every N global optimizer steps. The GradScaler scale
        is always logged every step when ``log_scaler=True``.
    log_scaler : bool
        If True, log the AMP GradScaler scale factor every optimization step.

    Add to ``config.diagnostics.callbacks``:

    .. code-block:: yaml

        diagnostics:
          callbacks:
            - _target_: anemoi.training.diagnostics.callbacks.gradient_detail.PerLayerGradientMonitor
              every_n_steps: 1
              log_scaler: true
    """

    def __init__(
        self,
        config: DictConfig,
        every_n_steps: int = 1,
        log_scaler: bool = True,
    ) -> None:
        super().__init__()
        self.every_n_steps = max(1, every_n_steps)
        self.log_scaler = log_scaler
        # Built on first optimizer step; None means not yet initialised.
        self._groups: Optional[dict[str, list[torch.nn.Parameter]]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_groups(
        pl_module: pl.LightningModule,
    ) -> dict[str, list[torch.nn.Parameter]]:
        """Scan named parameters once and return {group_key: [param, ...]} map.

        Groups produced
        ---------------
        ``processor``
            All parameters belonging to any processor layer (aggregate over
            all N blocks).
        ``proc.{N}``
            All parameters belonging to main processor layer N
            (path contains ``processor.proc.{N}.``).
        ``proc.{N}.{sub}``
            Parameters in sub-module *sub* of processor layer N, for each
            sub in ``_GT_SUBS`` that is present.
        ``encoder``
            All parameters whose path contains ``encoder.`` but not
            ``processor.proc.`` (aggregate).
        ``encoder.{sub}``
            Parameters in sub-module *sub* of the encoder block, for each
            sub in ``_ENC_DEC_SUBS`` that is present.
        ``decoder``
            All parameters whose path contains ``decoder.`` but not
            ``processor.proc.`` (aggregate).
        ``decoder.{sub}``
            Parameters in sub-module *sub* of the decoder block, for each
            sub in ``_ENC_DEC_SUBS`` that is present.
        """
        groups: dict[str, list[torch.nn.Parameter]] = {}
        # Named-parameter iteration gives DDP-unwrapped names in Lightning.
        param_map: dict[str, torch.nn.Parameter] = dict(pl_module.named_parameters())

        for name, param in param_map.items():
            if not param.requires_grad:
                continue

            if "processor.proc." in name:
                # Main processor — parse layer index.
                m = re.search(r"processor\.proc\.(\d+)\.", name)
                if m:
                    layer_key = f"proc.{m.group(1)}"
                    groups.setdefault("processor", []).append(param)
                    groups.setdefault(layer_key, []).append(param)

                    # Sub-module within this processor layer.
                    tail = name[m.end():]          # everything after 'proc.N.'
                    sub = tail.split(".")[0]        # first path segment
                    if sub in _GT_SUBS:
                        groups.setdefault(f"{layer_key}.{sub}", []).append(param)

            elif "encoder." in name:
                groups.setdefault("encoder", []).append(param)
                # Per-sublayer breakdown for the encoder block.
                m = re.search(r"encoder\..*?\.proc\.", name)
                if m:
                    sub = name[m.end():].split(".")[0]
                    if sub in _ENC_DEC_SUBS:
                        groups.setdefault(f"encoder.{sub}", []).append(param)

            elif "decoder." in name:
                groups.setdefault("decoder", []).append(param)
                # Per-sublayer breakdown for the decoder block.
                m = re.search(r"decoder\..*?\.proc\.", name)
                if m:
                    sub = name[m.end():].split(".")[0]
                    if sub in _ENC_DEC_SUBS:
                        groups.setdefault(f"decoder.{sub}", []).append(param)

        n_proc = sum(1 for k in groups if re.fullmatch(r"proc\.\d+", k))
        subs_found = sorted(
            set(k.split(".")[-1] for k in groups if re.match(r"proc\.\d+\.", k))
        )
        LOGGER.info(
            "PerLayerGradientMonitor: built %d groups — %d processor layers, "
            "sub-modules tracked: %s",
            len(groups),
            n_proc,
            subs_found,
        )
        return groups

    @staticmethod
    def _l2_norm(params: list[torch.nn.Parameter]) -> Optional[float]:
        """Return the L2 norm of gradients for *params*, or None if all are zero/missing."""
        grads = [p.grad.detach() for p in params if p.grad is not None]
        if not grads:
            return None
        return torch.norm(
            torch.stack([torch.norm(g, 2.0) for g in grads]),
            2.0,
        ).item()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: object,
    ) -> None:
        step = trainer.global_step

        # Always log the GradScaler scale when active (not gated by every_n_steps).
        if self.log_scaler:
            scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
            if scaler is not None:
                trainer.logger.log_metrics(
                    {"train/grad_scaler_scale": scaler.get_scale()},
                    step=step,
                )

        if step % self.every_n_steps != 0:
            return

        # Build groups once on the first logging step.
        if self._groups is None:
            self._groups = self._build_groups(pl_module)

        metrics: dict[str, float] = {}

        # Global L2 norm (continuity with GradientMonitor).
        all_grads = [p.grad.detach() for p in pl_module.parameters() if p.grad is not None]
        if all_grads:
            global_norm = torch.norm(
                torch.stack([torch.norm(g, 2.0) for g in all_grads]),
                2.0,
            ).item()
            metrics["train/grad_norm"] = global_norm

        # Per-group norms.
        for group_key, params in self._groups.items():
            norm = self._l2_norm(params)
            if norm is not None:
                metrics[f"train/glayer/{group_key}"] = norm

        if metrics:
            trainer.logger.log_metrics(metrics, step=step)
            LOGGER.debug(
                "PerLayerGradientMonitor: %d group norms logged at step %d",
                len(metrics),
                step,
            )


