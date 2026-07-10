"""Gemma autoregressive MoE router tracing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from logging_utils.moe import router_layer_index, summarize_router_output
from logging_utils.writer import JsonlWriter


class GemmaRouterTracer:
    def __init__(self, path: str | Path, *, selected_experts: int) -> None:
        self.writer = JsonlWriter(path)
        self.selected_experts = selected_experts
        self.context: dict[str, Any] | None = None
        self.forward_index = -1
        self.pending: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def attach(self, model: torch.nn.Module) -> None:
        self.handles.append(
            model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        )
        self.handles.append(
            model.register_forward_hook(self._after_forward, with_kwargs=True)
        )
        for module_name, module in model.named_modules():
            if not module_name.endswith(".router"):
                continue
            layer_index = router_layer_index(module_name)
            self.handles.append(
                module.register_forward_hook(self._router_hook(layer_index))
            )

    def begin_example(self, context: dict[str, Any]) -> None:
        self.context = context
        self.forward_index = -1
        self.pending.clear()

    def end_example(self) -> None:
        self.context = None
        self.pending.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.writer.close()

    def _before_forward(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self.context is None:
            return
        self.forward_index += 1
        self.pending.clear()

    def _after_forward(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if self.context is None:
            return
        phase = "prefill" if self.forward_index == 0 else "autoregressive_decode"
        for summary in self.pending:
            self.writer.write(
                {
                    "event": "moe_route",
                    **self.context,
                    "phase": phase,
                    "forward_index": self.forward_index,
                    "input_generated_token_step": (
                        None if self.forward_index == 0 else self.forward_index - 1
                    ),
                    "predicted_token_step": self.forward_index,
                    **summary,
                }
            )
        self.pending.clear()

    def _router_hook(self, layer_index: int):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            if self.context is None:
                return
            summary, _ = summarize_router_output(
                output,
                selected_experts=self.selected_experts,
            )
            self.pending.append({"layer_index": layer_index, **summary})

        return hook
