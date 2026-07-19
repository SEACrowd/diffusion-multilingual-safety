"""Final model-output logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .writer import JsonlWriter


class OutputLogger:
    def __init__(self, path: str | Path) -> None:
        self.writer = JsonlWriter(path)

    def log(
        self,
        *,
        context: dict[str, Any],
        final_text: str,
        final_token_ids: list[int],
        input_token_count: int,
        rendered_prompt: str | None = None,
        raw_response: str | None = None,
        response_error: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "event": "final_output",
            **context,
            "input_token_count": input_token_count,
            "output_token_count": len(final_token_ids),
            "response": final_text,
            "final_text": final_text,
            "final_token_ids": final_token_ids,
        }
        if rendered_prompt is not None:
            record["rendered_prompt"] = rendered_prompt
        if raw_response is not None:
            record["raw_response"] = raw_response
        if response_error is not None:
            record["response_error"] = response_error
        self.writer.write(record)
        return record

    def close(self) -> None:
        self.writer.close()
