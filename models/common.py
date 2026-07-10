"""Small helpers shared by model-specific inference runners."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader


def iter_examples(
    dataloader: DataLoader,
    max_batches: int | None,
) -> Iterator[tuple[str, dict[str, Any], str]]:
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        prompts = batch.get("prompts")
        metadata_rows = batch.get("metadata")
        references = batch.get("reference_responses")
        if not prompts or metadata_rows is None or references is None:
            raise ValueError("Dataloader batch is missing prompts, metadata, or references")
        yield from zip(prompts, metadata_rows, references, strict=True)


def example_seed(base_seed: int, example_id: str) -> int:
    digest = sha256(f"{base_seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def example_context(
    *,
    experiment_id: str,
    model_kind: str,
    model_id: str,
    model_revision: str,
    run_id: str,
    metadata: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "model_kind": model_kind,
        "model_id": model_id,
        "model_revision": model_revision,
        "run_id": run_id,
        "example_id": metadata.get("id", ""),
        "row_index": metadata.get("row_index"),
        "prompt_sha256": metadata.get("prompt_sha256", ""),
        "source": metadata.get("source", ""),
        "language": metadata.get("language", ""),
        "prompt_label": metadata.get("prompt_label", ""),
        "response_label": metadata.get("response_label", ""),
        "violated_categories": metadata.get("violated_categories", ""),
        "tag": metadata.get("tag", ""),
        "seed": seed,
    }


def text_config(model: Any) -> Any:
    return getattr(model.config, "text_config", model.config)


def selected_expert_count(model: Any) -> int:
    return int(getattr(text_config(model), "top_k_experts", 8))


def resolve_dtype(value: str | torch.dtype) -> str | torch.dtype:
    if not isinstance(value, str) or value == "auto":
        return value
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype {value!r}")
    return dtype


def parse_response_text(processor: Any, raw_text: str) -> str:
    parser = getattr(processor, "parse_response", None)
    if not callable(parser):
        return raw_text
    parsed = parser(raw_text)
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("response", "content", "final_answer", "text"):
            if isinstance(parsed.get(key), str):
                return parsed[key]
    for attribute in ("response", "content", "final_answer", "text"):
        value = getattr(parsed, attribute, None)
        if isinstance(value, str):
            return value
    return raw_text
