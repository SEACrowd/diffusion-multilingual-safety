"""Small helpers shared by model-specific inference runners."""

from __future__ import annotations

import warnings
from hashlib import sha256
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader


def iter_batches(
    dataloader: DataLoader,
    max_batches: int | None,
) -> Iterator[tuple[list[str], list[dict[str, Any]], list[str]]]:
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        prompts = batch.get("prompts")
        metadata_rows = batch.get("metadata")
        references = batch.get("reference_responses")
        if not prompts or metadata_rows is None or references is None:
            raise ValueError("Dataloader batch is missing prompts, metadata, or references")
        yield list(prompts), list(metadata_rows), list(references)


def iter_examples(
    dataloader: DataLoader,
    max_batches: int | None,
) -> Iterator[tuple[str, dict[str, Any], str]]:
    for prompts, metadata_rows, references in iter_batches(dataloader, max_batches):
        yield from zip(prompts, metadata_rows, references, strict=True)


def batch_seed(base_seed: int, example_ids: list[str]) -> int:
    return example_seed(base_seed, "|".join(example_ids))


def ensure_left_padding(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("Tokenizer needs a pad_token or eos_token for batched generation")
        tokenizer.pad_token = eos_token
    tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"


def trim_at_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    try:
        end = token_ids.index(eos_token_id) + 1
    except ValueError:
        return token_ids
    return token_ids[:end]


def resolve_response_boundary_token_ids(
    processor: Any,
) -> tuple[int | None, int | None, int | None]:
    """Return ``(eos_token_id, pad_token_id, turn_end_token_id)``."""
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    turn_end_token_id: int | None = None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        candidate = convert("<turn|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(candidate, int) and candidate != unk_token_id:
            turn_end_token_id = candidate
    if turn_end_token_id is None:
        turn_end_token_id = 106
    if pad_token_id is None:
        pad_token_id = 0
    return eos_token_id, pad_token_id, turn_end_token_id


def sanitize_generated_token_ids(
    token_ids: list[int],
    *,
    eos_token_id: int | None = None,
    pad_token_id: int | None = 0,
    turn_end_token_id: int | None = 106,
) -> list[int]:
    """Cut at EOS / ``<turn|>``, then drop pad and turn-end tokens.

    Batched decoding often continues with ``<pad>`` after the real response, and
    Gemma ends assistant turns with ``<turn|>`` instead of ``<eos>``.
    """
    stop_ids = {
        token_id
        for token_id in (eos_token_id, turn_end_token_id)
        if token_id is not None
    }
    end = len(token_ids)
    for index, token_id in enumerate(token_ids):
        if token_id not in stop_ids:
            continue
        # Keep EOS in the kept span; drop ``<turn|>`` by cutting before it.
        end = index + 1 if token_id == eos_token_id else index
        break

    drop_ids = {
        token_id
        for token_id in (pad_token_id, turn_end_token_id)
        if token_id is not None
    }
    return [token_id for token_id in token_ids[:end] if token_id not in drop_ids]


def strip_response_marker_text(text: str) -> str:
    cleaned = text
    for marker in ("<pad>", "<turn|>"):
        cleaned = cleaned.replace(marker, "")
    return cleaned.rstrip()


def model_primary_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters to resolve a device from") from exc


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
    enable_thinking: bool,
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
        "enable_thinking": enable_thinking,
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
    try:
        parsed = parser(raw_text)
    except Exception as exc:
        warnings.warn(
            f"Response parsing failed; preserving raw decoded output: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return raw_text
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
