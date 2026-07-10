"""Canvas-state JSONL logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from logging_utils.writer import JsonlWriter


class CanvasLogger:
    def __init__(self, path: str | Path, processor: Any) -> None:
        self.writer = JsonlWriter(path)
        self.processor = processor

    def log_step(
        self,
        *,
        context: dict[str, Any],
        canvas: torch.Tensor,
        changed_mask: torch.Tensor,
        accepted_mask: torch.Tensor,
        sampled_tokens: torch.Tensor,
    ) -> None:
        self.writer.write(
            {
                "event": "canvas_step",
                **context,
                "canvas_token_ids": canvas[0].detach().cpu().tolist(),
                "canvas_text": self.processor.decode(
                    canvas[0].detach().cpu(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "sampled_token_ids": sampled_tokens[0].detach().cpu().tolist(),
                "changed_positions": positions(changed_mask[0]),
                "accepted_positions": positions(accepted_mask[0]),
            }
        )

    def close(self) -> None:
        self.writer.close()


def positions(mask: torch.Tensor) -> list[int]:
    return torch.where(mask.bool())[0].detach().cpu().tolist()
