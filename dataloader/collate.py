"""Stateless collation helpers for DiffusionGemma text batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any


from .data import MultilingualSafetyExample, clean_text, metadata_from_row, row_value

IGNORE_INDEX = -100


def build_diffusion_gemma_collate_fn(
    processor: Any | None = None,
    tokenizer: Any | None = None,
    max_length: int = 2048,
    prompt_column: str = "prompt",
    response_column: str = "response",
    mask_prompt_labels: bool = True,
    return_metadata: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> Callable[[list[Mapping[str, Any] | MultilingualSafetyExample]], dict[str, Any]]:
    resolved_tokenizer = resolve_tokenizer(processor=processor, tokenizer=tokenizer)
    ensure_pad_token(resolved_tokenizer)
    template_source = processor if has_chat_template(processor) else resolved_tokenizer

    return partial(
        collate_diffusion_gemma_batch,
        tokenizer=resolved_tokenizer,
        template_source=template_source,
        max_length=max_length,
        prompt_column=prompt_column,
        response_column=response_column,
        mask_prompt_labels=mask_prompt_labels,
        return_metadata=return_metadata,
        ignore_index=ignore_index,
    )


def collate_diffusion_gemma_batch(
    examples: list[Mapping[str, Any] | MultilingualSafetyExample],
    *,
    tokenizer: Any,
    template_source: Any,
    max_length: int,
    prompt_column: str,
    response_column: str,
    mask_prompt_labels: bool,
    return_metadata: bool,
    ignore_index: int,
) -> dict[str, Any]:
    full_texts = [
        format_full_text(
            example,
            tokenizer=tokenizer,
            template_source=template_source,
            prompt_column=prompt_column,
            response_column=response_column,
        )
        for example in examples
    ]
    uses_chat_template = has_chat_template(template_source)
    encoded = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=not uses_chat_template,
        return_tensors="pt",
    )

    batch = dict(encoded)
    labels = encoded["input_ids"].clone()

    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, ignore_index)

    if mask_prompt_labels:
        prompt_lengths = [
            prompt_token_length(
                example,
                tokenizer=tokenizer,
                template_source=template_source,
                max_length=max_length,
                prompt_column=prompt_column,
            )
            for example in examples
        ]
        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, : min(prompt_length, labels.shape[1])] = ignore_index

    batch["labels"] = labels

    if return_metadata:
        batch["metadata"] = [metadata_from_row(example) for example in examples]

    return batch


def format_full_text(
    example: Mapping[str, Any] | MultilingualSafetyExample,
    *,
    tokenizer: Any,
    template_source: Any,
    prompt_column: str = "prompt",
    response_column: str = "response",
) -> str:
    prompt = clean_text(row_value(example, prompt_column, ""))
    response = clean_text(row_value(example, response_column, ""))

    if has_chat_template(template_source):
        return template_source.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    eos_token = getattr(tokenizer, "eos_token", "") or ""
    return f"User: {prompt}\nAssistant: {response}{eos_token}"


def format_prompt_prefix(
    example: Mapping[str, Any] | MultilingualSafetyExample,
    *,
    template_source: Any,
    prompt_column: str = "prompt",
) -> str:
    prompt = clean_text(row_value(example, prompt_column, ""))

    if has_chat_template(template_source):
        return template_source.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"User: {prompt}\nAssistant:"


def prompt_token_length(
    example: Mapping[str, Any] | MultilingualSafetyExample,
    *,
    tokenizer: Any,
    template_source: Any,
    max_length: int,
    prompt_column: str = "prompt",
) -> int:
    uses_chat_template = has_chat_template(template_source)
    tokens = tokenizer(
        format_prompt_prefix(
            example,
            template_source=template_source,
            prompt_column=prompt_column,
        ),
        truncation=True,
        max_length=max_length,
        add_special_tokens=not uses_chat_template,
    )["input_ids"]
    return len(tokens)


def resolve_tokenizer(processor: Any | None = None, tokenizer: Any | None = None) -> Any:
    resolved = tokenizer or getattr(processor, "tokenizer", None) or processor
    if resolved is None:
        raise ValueError("Pass either processor or tokenizer")
    return resolved


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is not None:
        tokenizer.pad_token = eos_token


def has_chat_template(value: Any) -> bool:
    return callable(getattr(value, "apply_chat_template", None))

