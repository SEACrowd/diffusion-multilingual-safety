"""Diffusers callback coordinating canvas, logits, and MoE logs."""

from __future__ import annotations

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
        canvas_logger: CanvasLogger,
        logits_logger: DiffusionLogitsLogger | None,
        router_tracer: DiffusionGemmaRouterTracer | None,
    ) -> None:
        self.context = context
        self.canvas_index = -1
        self.previous_canvas: torch.Tensor | None = None
        self.canvas_logger = canvas_logger
        self.logits_logger = logits_logger
        self.router_tracer = router_tracer

    def __call__(
        self,
        pipe: Any,
        global_step: int,
        step_index: int,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        canvas = callback_kwargs["canvas"].detach()
        if step_index == 0:
            self.canvas_index += 1
            self.previous_canvas = None

        scheduler_trace = pipe.scheduler.last_trace
        if scheduler_trace is None:
            raise RuntimeError("Tracing scheduler did not expose a step result")

        changed_mask = (
            torch.ones_like(canvas, dtype=torch.bool)
            if self.previous_canvas is None
            else canvas.ne(self.previous_canvas)
        )
        accepted_mask = scheduler_trace["accepted_index"]
        step_context = {
            "canvas_index": self.canvas_index,
            "global_step": global_step,
            "step_index": step_index,
        }
        context = {**self.context, **step_context}

        self.canvas_logger.log_step(
            context=context,
            canvas=canvas,
            changed_mask=changed_mask,
            accepted_mask=accepted_mask,
            sampled_tokens=scheduler_trace["sampled_tokens"],
        )
        if self.logits_logger is not None:
            self.logits_logger.log_step(
                context=context,
                logits=scheduler_trace["pred_logits"],
                sampled_tokens=scheduler_trace["sampled_tokens"],
                sampled_probabilities=scheduler_trace["sampled_probs"],
                accepted_mask=accepted_mask,
            )
        if self.router_tracer is not None:
            self.router_tracer.flush_step(step_context)

        self.previous_canvas = canvas.clone()
        return {}
