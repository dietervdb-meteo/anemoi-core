# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DtypeAuditCallback — log the dtype of weights, gradients, and Adam state.

Fires once (or on demand) and writes a structured dtype report to the Lightning
logger and to stdout.  Zero overhead after the first audit step.

For each parameter whose name matches ``name_filter`` (a ``re.search()`` pattern,
default: attention projection weights) reports:

  dtype/weight/<param>         dtype of the weight tensor itself
  dtype/grad/<param>           dtype of param.grad (after backward)
  dtype/adam_m1/<param>        dtype of Adam exp_avg  (first moment)
  dtype/adam_v2/<param>        dtype of Adam exp_avg_sq (second moment)

These four fields answer the key diagnostic questions:

* ``dtype/weight`` — are weights stored in bf16/f16 or fp32? (expected: bf16/f16)
* ``dtype/grad``   — are *weight* gradients bf16 or fp32?
  In a plain AMP bf16 run without Fix AU the weight grad is bf16 (7-bit mantissa
  noise flows all the way to the optimizer).  With Fix AU (attn_logit_fp32=True)
  the upstream dQ/dK/dV are fp32, so the weight grad is typically fp32 too.
* ``dtype/adam_m1/v2`` — Adam state is normally kept in fp32 by PyTorch regardless
  of AMP dtype; this confirms (or refutes) that assumption.

Dtypes are encoded as strings ("torch.float32", "torch.bfloat16", etc.).
MLflow does not support string metrics; they are therefore logged as
``AUDIT`` INFO messages and written to a plain-text file
``<output_dir>/dtype_audit_step_{step}.txt`` for offline inspection.

Configuration
-------------
.. code-block:: yaml

    diagnostics:
      callbacks:
        - _target_: anemoi.training.diagnostics.callbacks.dtype_audit.DtypeAuditCallback
          # re.search() pattern matched against full parameter names.
          # Default covers all attention projection weights across all components.
          name_filter: "lin_key\\.weight$|lin_query\\.weight$"
          # Training steps at which to run the audit.  Default: [1] (first step
          # only).  Add e.g. 7750 to catch the state just before collapse.
          audit_steps: [1]
          # Directory for the plain-text report file.  Defaults to the trainer's
          # default_root_dir if null.
          output_dir: null
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities import rank_zero_only

LOGGER = logging.getLogger(__name__)

# Default pattern: all attention projection weights in any component.
_DEFAULT_FILTER = r"lin_key\.weight$|lin_query\.weight$"


class DtypeAuditCallback(pl.callbacks.Callback):
    """Log dtypes of weights, gradients, and Adam state for selected parameters.

    Parameters
    ----------
    config : DictConfig
        Full job configuration (required by the anemoi callback contract).
    name_filter : str
        ``re.search()`` pattern matched against fully-qualified parameter names.
        Default covers ``lin_key.weight`` and ``lin_query.weight`` everywhere.
    audit_steps : list[int]
        Global optimizer steps at which to run the audit.  Default ``[1]``.
    output_dir : str or None
        Directory for the plain-text report.  Falls back to
        ``trainer.default_root_dir`` when ``None``.
    """

    def __init__(
        self,
        config: DictConfig,
        name_filter: str = _DEFAULT_FILTER,
        audit_steps: Optional[list] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.name_filter = re.compile(name_filter)
        if audit_steps is None:
            audit_steps = [1]
        # Accept OmegaConf ListConfig as well as plain list.
        self.audit_steps: set[int] = set(int(s) for s in audit_steps)
        self._output_dir = Path(output_dir) if output_dir else None

    # ------------------------------------------------------------------

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: object,
    ) -> None:
        step = trainer.global_step
        if step not in self.audit_steps:
            return

        out_dir = self._output_dir or Path(trainer.default_root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"dtype_audit_step_{step:06d}.txt"

        lines: list[str] = [
            f"=== DtypeAuditCallback — step {step} ===",
            f"{'parameter':<70}  {'weight':>12}  {'grad':>12}  {'adam_m1':>12}  {'adam_v2':>12}",
            "-" * 120,
        ]

        for name, param in pl_module.named_parameters():
            if not self.name_filter.search(name):
                continue

            w_dtype  = str(param.dtype)
            g_dtype  = str(param.grad.dtype)  if param.grad is not None else "no grad"
            opt_state = optimizer.state.get(param, {})
            m1 = opt_state.get("exp_avg")
            v2 = opt_state.get("exp_avg_sq")
            m1_dtype = str(m1.dtype) if m1 is not None else "not init"
            v2_dtype = str(v2.dtype) if v2 is not None else "not init"

            line = f"{name:<70}  {w_dtype:>12}  {g_dtype:>12}  {m1_dtype:>12}  {v2_dtype:>12}"
            lines.append(line)
            LOGGER.info("AUDIT  %s", line)

        report = "\n".join(lines)
        report_path.write_text(report + "\n", encoding="utf-8")
        LOGGER.info("DtypeAuditCallback: report written to %s", report_path)
