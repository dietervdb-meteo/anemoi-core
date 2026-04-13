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
    """Monitor attention weight norms, Adam state, and gradient-radial projections.

    For every parameter whose fully-qualified name ends with one of the suffixes
    in ``_WEIGHT_NORM_SUFFIXES`` (``lin_query.weight``, ``lin_key.weight``,
    ``q_norm.weight``, ``k_norm.weight``), logs:

    **Weight norm** (``weight_norm/<param_path>``)
        L2 norm of the weight tensor.  Primary observable for logit-scale drift.

    **Gradient radial projection** (``train/radial/<param_path>``, optional)
        Signed projection of the gradient onto the weight vector, normalised by
        the weight norm:  ``(g · W) / ‖W‖``.

        * Negative  → gradient is restoring norm (pulling weight back toward origin).
        * Near zero → gradient is tangential; norm is at equilibrium.
        * Positive  → gradient is extending norm (pushing weight further out).

        Persistently positive values in a run indicate the optimiser's restoring
        force is insufficient — the Phase 1 equilibrium-disruption signature.

    **Adam first moment norm** (``train/adam_m1/<param_path>``, optional)
        L2 norm of the exponential moving average of gradients (``exp_avg``).
        Captures momentum accumulated across steps; remains non-zero even after
        the instantaneous gradient collapses (Phase 2 runaway diagnostic).

    **Adam second moment norm** (``train/adam_v2/<param_path>``, optional)
        L2 norm of ``exp_avg_sq``.  Together with ``m1``, allows computing the
        effective per-element step magnitude.

    **Adam effective step norm** (``train/adam_eff/<param_path>``, optional)
        L2 norm of ``m1 / (sqrt(v2) + eps)``.  This is proportional to the
        actual weight update vector; if this is large while ``‖grad‖`` is small,
        Adam momentum is driving norm growth independently of the current gradient.

    The last four metrics require ``log_adam=True`` / ``log_radial=True``
    respectively.  They are off by default to avoid log bloat in production runs.

    Metrics are logged to MLflow (or any Lightning logger).

    Add to ``config.diagnostics.callbacks``:

    .. code-block:: yaml

        diagnostics:
          callbacks:
            - _target_: anemoi.training.diagnostics.callbacks.gradient.WeightNormMonitor
              every_n_steps: 50
              log_radial: true
              log_adam: true

    """

    def __init__(
        self,
        config: DictConfig,
        every_n_steps: int = 100,
        log_radial: bool = False,
        log_adam: bool = False,
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.log_radial = log_radial
        self.log_adam = log_adam

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
            if not any(name.endswith(sfx) for sfx in _WEIGHT_NORM_SUFFIXES):
                continue

            key = name[: -len(".weight")]
            w = param.detach()
            metrics[f"weight_norm/{key}"] = w.norm(2.0).item()

            # Gradient radial projection: (g · W) / ‖W‖
            # Positive = gradient extends norm; negative = restores norm.
            if self.log_radial and param.grad is not None:
                g = param.grad.detach()
                w_norm = w.norm(2.0)
                if w_norm > 0:
                    metrics[f"train/radial/{key}"] = (
                        torch.dot(g.flatten(), w.flatten()) / w_norm
                    ).item()

            # Adam internal state.
            if self.log_adam:
                opt_state = optimizer.state.get(param, {})
                m1 = opt_state.get("exp_avg")
                v2 = opt_state.get("exp_avg_sq")
                if m1 is not None:
                    metrics[f"train/adam_m1/{key}"] = m1.norm(2.0).item()
                if v2 is not None:
                    metrics[f"train/adam_v2/{key}"] = v2.norm(2.0).item()
                if m1 is not None and v2 is not None:
                    eff = (m1 / (v2.sqrt() + 1e-8)).norm(2.0).item()
                    metrics[f"train/adam_eff/{key}"] = eff

        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
            LOGGER.debug("logged %d weight/adam/radial metrics at step %d", len(metrics), trainer.global_step)
