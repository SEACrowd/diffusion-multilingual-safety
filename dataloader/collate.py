"""Model-neutral collation for multilingual safety examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .data import MultilingualSafetyExample, clean_text, metadata_from_row, row_value


def collate_multilingual_safety_batch(
    examples: list[Mapping[str, Any] | MultilingualSafetyExample],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")

    metadata = [metadata_from_row(example) for example in examples]
    for item, example in zip(metadata, examples, strict=True):
        item["row_index"] = row_value(example, "row_index")
        item["prompt_sha256"] = clean_text(row_value(example, "prompt_sha256", ""))

    return {
        "prompts": [clean_text(row_value(example, "prompt", "")) for example in examples],
        "reference_responses": [
            clean_text(row_value(example, "response", "")) for example in examples
        ],
        "metadata": metadata,
    }
