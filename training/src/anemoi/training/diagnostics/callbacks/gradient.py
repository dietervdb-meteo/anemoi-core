# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

LOGGER = logging.getLogger(__name__)

# Parameter name suffixes tracked by WeightNormMonitor.
_WEIGHT_NORM_SUFFIXES = (
    "lin_query.weight",
    "lin_key.weight",
    "q_norm.weight",
    "k_norm.weight",
)


class GradientMonitor(pl.callbacks.Callback):
    """Monitor gradient norms and (optionally) AMP scaler scale during training.

    Computes the global L2 gradient norm over all model parameters every
    ``every_n_steps`` global optimiser steps and logs it as ``train/grad_norm``
    to the Lightning logger (wandb, MLflow, …).

    If ``log_scaler=True`` and an AMP ``GradScaler`` is active (fp16 training),
    the current scale factor is also logged as ``train/grad_scaler_scale``.

    The hook fires after Lightning's gradient clipping (if any), so the logged
    norm reflects what the optimiser actually sees.

    Add to ``config.diagnostics.callbacks``:

    .. code-block:: yaml

        diagnostics:
          callbacks:
            - _target_: anemoi.training.diagnostics.callbacks.gradient.GradientMonitor
              every_n_steps: 100
              log_scaler: false

    """

    def __init__(
        self,
        config: DictConfig,
        every_n_steps: int = 100,
        log_scaler: bool = False,
    ) -> None:
        """Initialise GradientMonitor.

        Parameters
        ----------
        config : DictConfig
            Job configuration (required by the anemoi callback contract).
        every_n_steps : int
            Log every N global optimiser steps. Must be >= 1.
        log_scaler : bool
            If True, also log the AMP GradScaler scale factor when active.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.log_scaler = log_scaler

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: object,
    ) -> None:
        if self.every_n_steps <= 0 or trainer.global_step % self.every_n_steps != 0:
            return

        # Global L2 gradient norm over all parameters that have a gradient.
        # Gradients are all-reduced by Lightning before this hook fires in DDP,
        # so rank 0's norm equals every other rank's norm.
        grads = [p.grad.detach() for p in pl_module.parameters() if p.grad is not None]
        if grads:
            grad_norm = torch.norm(
                torch.stack([torch.norm(g, 2.0) for g in grads]),
                2.0,
            ).item()
            trainer.logger.log_metrics({"train/grad_norm": grad_norm}, step=trainer.global_step)
            LOGGER.debug("grad_norm=%.4e at step %d", grad_norm, trainer.global_step)

        if self.log_scaler:
            scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
            if scaler is not None:
                trainer.logger.log_metrics(
                    {"train/grad_scaler_scale": scaler.get_scale()},
                    step=trainer.global_step,
                )


class WeightNormMonitor(pl.callbacks.Callback):
    """Monitor L2 norms of selected attention projection weights during training.

    Logs the L2 norm of every parameter whose fully-qualified name ends with one
    of the suffixes in ``_WEIGHT_NORM_SUFFIXES`` (currently ``lin_query.weight``,
    ``lin_key.weight``, ``q_norm.weight``, ``k_norm.weight``).

    These four parameters together tell the story of attention logit magnitude:

    * **Without qk_norm** (``qk_norm=False``): ``lin_query.weight`` and
      ``lin_key.weight`` norms grow proportionally to softmax logit std.
    * **With qk_norm** (``qk_norm=True``): ``lin_query.weight`` norm is
      not informative.  Instead, ``q_norm.weight`` and ``k_norm.weight`` are
      the direct proxy: effective logit std ≈ mean(q_norm.weight) × mean(k_norm.weight) × √d_head.

    Metrics are logged to MLflow (or any Lightning logger) as
    ``weight_norm/<param_path_without_.weight_suffix>``.

    Add to ``config.diagnostics.callbacks``:

    .. code-block:: yaml

        diagnostics:
          callbacks:
            - _target_: anemoi.training.diagnostics.callbacks.gradient.WeightNormMonitor
              every_n_steps: 100

    """

    def __init__(
        self,
        config: DictConfig,
        every_n_steps: int = 100,
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: object,
    ) -> None:
        if self.every_n_steps <= 0 or trainer.global_step % self.every_n_steps != 0:
            return

        metrics: dict[str, float] = {}
        for name, param in pl_module.named_parameters():
            if any(name.endswith(sfx) for sfx in _WEIGHT_NORM_SUFFIXES):
                key = "weight_norm/" + name[: -len(".weight")]
                metrics[key] = param.detach().norm(2.0).item()

        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
            LOGGER.debug("logged %d weight norms at step %d", len(metrics), trainer.global_step)
