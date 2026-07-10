"""Entropy-bound scheduler instrumentation."""

from __future__ import annotations

import time
from typing import Any

from diffusers import EntropyBoundScheduler


class TracingEntropyBoundScheduler(EntropyBoundScheduler):
    last_trace: dict[str, Any] | None = None

    def step(self, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        output = super().step(*args, **kwargs)
        self.last_trace = {
            "accepted_index": output.accepted_index.detach(),
            "sampled_tokens": output.sampled_tokens.detach(),
            "sampled_probs": output.sampled_probs.detach(),
            "pred_logits": output.pred_logits.detach(),
            "scheduler_latency": time.perf_counter() - started_at,
        }
        return output
