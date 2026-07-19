"""Entropy-bound scheduler state capture for DiffusionGemma logging."""

from __future__ import annotations

from typing import Any

from diffusers import EntropyBoundScheduler


class TracingEntropyBoundScheduler(EntropyBoundScheduler):
    last_trace: dict[str, Any] | None = None

    def step(self, *args: Any, **kwargs: Any) -> Any:
        output = super().step(*args, **kwargs)
        self.last_trace = {
            "accepted_index": output.accepted_index.detach(),
            "sampled_tokens": output.sampled_tokens.detach(),
            "sampled_probs": output.sampled_probs.detach(),
            "pred_logits": output.pred_logits.detach(),
        }
        return output
