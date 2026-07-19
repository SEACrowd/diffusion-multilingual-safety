"""DiffusionGemma encoder and denoiser MoE routing traces."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from logging_utils.moe import (
    route_change_rate,
    router_layer_index,
    summarize_router_output,
)
from logging_utils.writer import JsonlWriter


class DiffusionGemmaRouterTracer:
    def __init__(self, path: str | Path, *, selected_experts: int) -> None:
        self.writer = JsonlWriter(path)
        self.selected_experts = selected_experts
        self.context: dict[str, Any] | None = None
        self.pending: list[tuple[str, int, dict[str, Any], torch.Tensor]] = []
        self.previous_decoder_routes: dict[tuple[int, int], torch.Tensor] = {}
        self.handles: list[Any] = []
        self.disabled = False

    def attach(self, model: torch.nn.Module) -> None:
        for module_name, module in model.named_modules():
            if not module_name.endswith(".router"):
                continue
            phase = "encoder_prefill" if ".encoder." in module_name else "denoising"
            layer_index = router_layer_index(module_name)
            self.handles.append(
                module.register_forward_hook(self._router_hook(phase, layer_index))
            )

    def begin_example(self, context: dict[str, Any]) -> None:
        if self.disabled:
            return
        self.context = context
        self.pending.clear()
        self.previous_decoder_routes.clear()

    def flush_step(self, step_context: dict[str, Any]) -> None:
        if self.context is None:
            return
        try:
            canvas_index = int(step_context["canvas_index"])
            for phase, layer_index, summary, routes in self.pending:
                record = {
                    "event": "moe_route",
                    **self.context,
                    **step_context,
                    "phase": phase,
                    "layer_index": layer_index,
                    **summary,
                }
                if phase == "denoising":
                    route_key = (canvas_index, layer_index)
                    record["expert_set_change_rate"] = route_change_rate(
                        routes,
                        self.previous_decoder_routes.get(route_key),
                    )
                    self.previous_decoder_routes[route_key] = routes
                self.writer.write(record)
            self.pending.clear()
        except Exception as error:
            self.disable(error)

    def end_example(self) -> None:
        if self.pending:
            self.disable(
                RuntimeError("Unflushed DiffusionGemma router events remain after inference")
            )
            return
        self.context = None
        self.previous_decoder_routes.clear()

    def abort_example(self) -> None:
        self.context = None
        self.pending.clear()
        self.previous_decoder_routes.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.writer.close()

    def disable(self, error: Exception) -> None:
        if not self.disabled:
            warnings.warn(
                f"Optional DiffusionGemma MoE telemetry disabled after "
                f"{type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.disabled = True
        self.abort_example()

    def _router_hook(self, phase: str, layer_index: int):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if self.context is None:
                return
            try:
                summary, routes = summarize_router_output(
                    output,
                    selected_experts=self.selected_experts,
                )
                self.pending.append((phase, layer_index, summary, routes))
            except Exception as error:
                self.disable(error)

        return hook
