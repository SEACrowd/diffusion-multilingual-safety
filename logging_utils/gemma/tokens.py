"""Autoregressive token-event logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from logging_utils.writer import JsonlWriter


class GemmaTokenLogger:
    def __init__(self, path: str | Path, processor: Any) -> None:
        self.writer = JsonlWriter(path)
        self.processor = processor

    def log_generation(self, context: dict[str, Any], token_ids: list[int]) -> None:
        for token_step, token_id in enumerate(token_ids):
            self.writer.write(
                {
                    "event": "token_step",
                    **context,
                    "token_step": token_step,
                    "token_id": token_id,
                    "token_text": self.processor.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )

    def close(self) -> None:
        self.writer.close()
