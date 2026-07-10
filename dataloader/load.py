"""Hugging Face dataset loading and PyTorch DataLoader construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from pydantic import BaseModel
from torch.utils.data import DataLoader

from .collate import build_diffusion_gemma_collate_fn
from .data import DATASET_COLUMNS, DATASET_NAME, normalize_multilingual_safety_row


class MultilingualSafetyDataConfig(BaseModel):
    dataset_name: str = DATASET_NAME
    split: str = "train"
    streaming: bool = False
    cache_dir: str | None = None
    token: str | bool | None = None
    max_samples: int | None = None
    languages: Sequence[str] | str | None = None
    prompt_labels: Sequence[str] | str | None = None
    response_labels: Sequence[str] | str | None = None
    tags: Sequence[str] | str | None = None
    sources: Sequence[str] | str | None = None
    seed: int = 42
    filter_num_proc: int | None = None
    map_num_proc: int | None = None
    normalize: bool = True


def load_multilingual_safety_dataset(
    config: MultilingualSafetyDataConfig | None = None,
    **overrides: Any,
) -> Dataset | IterableDataset:
    cfg = merge_config(config, overrides)
    dataset = load_dataset(
        cfg.dataset_name,
        split=cfg.split,
        streaming=cfg.streaming,
        cache_dir=cfg.cache_dir,
        token=cfg.token,
    )

    validate_columns(dataset)
    dataset = filter_dataset(
        dataset,
        languages=cfg.languages,
        prompt_labels=cfg.prompt_labels,
        response_labels=cfg.response_labels,
        tags=cfg.tags,
        sources=cfg.sources,
        num_proc=None if cfg.streaming else cfg.filter_num_proc,
    )

    if cfg.normalize:
        dataset = map_dataset(
            dataset,
            normalize_multilingual_safety_row,
            num_proc=None if cfg.streaming else cfg.map_num_proc,
        )

    if cfg.max_samples is not None:
        if cfg.max_samples < 1:
            raise ValueError("max_samples must be positive when provided")
        if cfg.streaming:
            dataset = dataset.take(cfg.max_samples)
        else:
            dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

    return dataset


def create_multilingual_safety_dataloader(
    processor: Any | None = None,
    tokenizer: Any | None = None,
    config: MultilingualSafetyDataConfig | None = None,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_length: int = 2048,
    mask_prompt_labels: bool = True,
    return_metadata: bool = True,
    shuffle_buffer_size: int = 10_000,
    **dataset_overrides: Any,
) -> DataLoader:
    cfg = merge_config(config, dataset_overrides)
    dataset = load_multilingual_safety_dataset(cfg)

    dataloader_shuffle = shuffle
    if cfg.streaming:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=cfg.seed)
        dataloader_shuffle = False

    collate_fn = build_diffusion_gemma_collate_fn(
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_prompt_labels=mask_prompt_labels,
        return_metadata=return_metadata,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def merge_config(
    config: MultilingualSafetyDataConfig | None,
    overrides: dict[str, Any],
) -> MultilingualSafetyDataConfig:
    base = config or MultilingualSafetyDataConfig()
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if hasattr(base, "model_copy"):
        return base.model_copy(update=clean_overrides)
    return base.copy(update=clean_overrides)


def validate_columns(dataset: Dataset | IterableDataset) -> None:
    column_names = getattr(dataset, "column_names", None)
    if column_names is None and getattr(dataset, "features", None) is not None:
        column_names = list(dataset.features.keys())
    if column_names is None:
        return

    missing = sorted(set(DATASET_COLUMNS) - set(column_names))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {', '.join(missing)}")


def filter_dataset(
    dataset: Dataset | IterableDataset,
    languages: Sequence[str] | str | None = None,
    prompt_labels: Sequence[str] | str | None = None,
    response_labels: Sequence[str] | str | None = None,
    tags: Sequence[str] | str | None = None,
    sources: Sequence[str] | str | None = None,
    num_proc: int | None = None,
) -> Dataset | IterableDataset:
    filters = {
        "language": as_filter_set(languages),
        "prompt_label": as_filter_set(prompt_labels),
        "response_label": as_filter_set(response_labels),
        "tag": as_filter_set(tags),
        "source": as_filter_set(sources),
    }
    active_filters = {key: value for key, value in filters.items() if value is not None}
    if not active_filters:
        return dataset

    def keep(example: dict[str, Any]) -> bool:
        return all(
            str(example.get(column, "")) in allowed
            for column, allowed in active_filters.items()
        )

    if isinstance(dataset, IterableDataset):
        return dataset.filter(keep)
    return dataset.filter(keep, num_proc=num_proc)


def map_dataset(
    dataset: Dataset | IterableDataset,
    function: Any,
    num_proc: int | None = None,
) -> Dataset | IterableDataset:
    if isinstance(dataset, IterableDataset):
        return dataset.map(function)
    return dataset.map(function, num_proc=num_proc)


def as_filter_set(values: Sequence[str] | str | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}



