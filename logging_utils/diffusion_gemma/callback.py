"""Diffusers callback coordinating canvas, logits, and MoE logs."""

from __future__ import annotations

import warnings
from typing import Any

import torch

from .canvas import CanvasLogger
from .logits import DiffusionLogitsLogger
from .router import DiffusionGemmaRouterTracer


class DiffusionLoggingCallback:
    def __init__(
        self,
        *,
        context: dict[str, Any],
        canvas_logger: CanvasLogger | None,
        logits_logger: DiffusionLogitsLogger | None,
        router_tracer: DiffusionGemmaRouterTracer | None,
        disabled_components: set[str] | None = None,
    ) -> None:
        self.context = context
        self.canvas_index = -1
        self.previous_canvas: torch.Tensor | None = None
        self.canvas_logger = canvas_logger
        self.logits_logger = logits_logger
        self.router_tracer = router_tracer
        self.disabled_components = (
            disabled_components if disabled_components is not None else set()
        )

    def __call__(
        self,
        pipe: Any,
        global_step: int,
        step_index: int,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        try:
            canvas = callback_kwargs["canvas"].detach()
        except Exception as error:
            if "callback_input" not in self.disabled_components:
                warnings.warn(
                    f"Optional DiffusionGemma step telemetry disabled after "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.disabled_components.update(
                    {"callback_input", "canvas", "logits", "moe"}
                )
            return {}
        if step_index == 0:
            self.canvas_index += 1
            self.previous_canvas = None

        scheduler_trace = pipe.scheduler.last_trace
        if scheduler_trace is None:
            if "scheduler_trace" not in self.disabled_components:
                warnings.warn(
                    "Optional DiffusionGemma step telemetry disabled because the "
                    "scheduler did not expose a step result",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.disabled_components.update(
                    {"scheduler_trace", "canvas", "logits", "moe"}
                )
            return {}

        changed_mask = (
            torch.ones_like(canvas, dtype=torch.bool)
            if self.previous_canvas is None
            else canvas.ne(self.previous_canvas)
        )
        try:
            accepted_mask = scheduler_trace["accepted_index"]
            sampled_tokens = scheduler_trace["sampled_tokens"]
            sampled_probabilities = scheduler_trace["sampled_probs"]
            predicted_logits = scheduler_trace["pred_logits"]
        except Exception as error:
            if "scheduler_trace" not in self.disabled_components:
                warnings.warn(
                    f"Optional DiffusionGemma step telemetry disabled after "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.disabled_components.update(
                    {"scheduler_trace", "canvas", "logits", "moe"}
                )
            return {}
        step_context = {
            "canvas_index": self.canvas_index,
            "global_step": global_step,
            "step_index": step_index,
        }
        context = {**self.context, **step_context}

        if self.canvas_logger is not None and "canvas" not in self.disabled_components:
            try:
                self.canvas_logger.log_step(
                    context=context,
                    canvas=canvas,
                    changed_mask=changed_mask,
                    accepted_mask=accepted_mask,
                    sampled_tokens=sampled_tokens,
                )
            except Exception as error:
                self.disabled_components.add("canvas")
                warnings.warn(
                    f"Optional canvas telemetry disabled after "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self.logits_logger is not None and "logits" not in self.disabled_components:
            try:
                self.logits_logger.log_step(
                    context=context,
                    logits=predicted_logits,
                    sampled_tokens=sampled_tokens,
                    sampled_probabilities=sampled_probabilities,
                    accepted_mask=accepted_mask,
                )
            except Exception as error:
                self.disabled_components.add("logits")
                warnings.warn(
                    f"Optional logits telemetry disabled after "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self.router_tracer is not None and "moe" not in self.disabled_components:
            try:
                self.router_tracer.flush_step(step_context)
            except Exception as error:
                self.disabled_components.add("moe")
                self.router_tracer.disable(error)

        self.previous_canvas = canvas.clone()
        return {}
