"""Hugging Face dataset loading and fixed-manifest construction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from logging_utils.writer import write_json, write_jsonl

from .collate import collate_multilingual_safety_batch
from .data import DATASET_COLUMNS, DATASET_NAME, normalize_multilingual_safety_row


class MultilingualSafetyDataConfig(BaseModel):
    dataset_name: str = DATASET_NAME
    revision: str = "main"
    split: str = "train"
    streaming: bool = False
    cache_dir: str | None = None
    max_samples: int | None = Field(default=None, gt=0)
    languages: Sequence[str] | str | None = None
    prompt_labels: Sequence[str] | str | None = None
    response_labels: Sequence[str] | str | None = None
    tags: Sequence[str] | str | None = None
    sources: Sequence[str] | str | None = None
    seed: int = 42
    shuffle: bool = False
    shuffle_buffer_size: int = Field(default=10_000, gt=0)
    filter_num_proc: int | None = None
    map_num_proc: int | None = None
    normalize: bool = True


def resolve_dataset_revision(
    dataset_name: str,
    revision: str,
    token: str | None = None,
) -> str:
    return HfApi(token=token).dataset_info(dataset_name, revision=revision).sha


def load_multilingual_safety_dataset(
    config: MultilingualSafetyDataConfig | None = None,
    *,
    token: str | None = None,
    apply_limit: bool = True,
    **overrides: Any,
) -> Dataset | IterableDataset:
    cfg = merge_config(config, overrides)
    dataset = load_dataset(
        cfg.dataset_name,
        split=cfg.split,
        revision=cfg.revision,
        streaming=cfg.streaming,
        cache_dir=cfg.cache_dir,
        token=token,
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

    if apply_limit:
        dataset = limit_dataset(dataset, cfg.max_samples)
    return dataset


def materialize_input_manifest(
    config: MultilingualSafetyDataConfig,
    *,
    manifest_path: str | Path,
    metadata_path: str | Path,
    token: str | None = None,
) -> dict[str, Any]:
    resolved_revision = resolve_dataset_revision(
        config.dataset_name,
        config.revision,
        token,
    )
    resolved_config = model_copy(config, revision=resolved_revision, max_samples=None)
    dataset = load_multilingual_safety_dataset(
        resolved_config,
        token=token,
        apply_limit=False,
    )
    dataset = shuffle_dataset(dataset, config)
    dataset = limit_dataset(dataset, config.max_samples)
    fingerprint = getattr(dataset, "_fingerprint", None)

    seen_ids: set[str] = set()

    def rows() -> Iterator[dict[str, Any]]:
        for row_index, row in enumerate(dataset):
            example = normalize_multilingual_safety_row(row)
            example_id = example["id"]
            prompt = example["prompt"]
            if not example_id:
                raise ValueError(f"Dataset row {row_index} has an empty id")
            if example_id in seen_ids:
                # raise ValueError(f"Dataset contains duplicate id {example_id!r}")
                print(f"Dataset contains duplicate id {example_id!r}")
                continue
            if not prompt:
                # raise ValueError(f"Dataset row {example_id!r} has an empty prompt")
                print(f"Dataset row {example_id!r} has an empty prompt")
                continue
            seen_ids.add(example_id)
            yield {
                **example,
                "row_index": row_index,
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            }

    count = write_jsonl(manifest_path, rows())
    metadata = {
        "dataset_name": config.dataset_name,
        "requested_revision": config.revision,
        "resolved_revision": resolved_revision,
        "split": config.split,
        "streaming": config.streaming,
        "fingerprint": fingerprint,
        "examples": count,
        "manifest_path": str(manifest_path),
        "filters": {
            "languages": as_list(config.languages),
            "prompt_labels": as_list(config.prompt_labels),
            "response_labels": as_list(config.response_labels),
            "tags": as_list(config.tags),
            "sources": as_list(config.sources),
        },
        "shuffle": config.shuffle,
        "seed": config.seed,
    }
    write_json(metadata_path, metadata)
    return metadata


def create_manifest_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = load_dataset(
        "json",
        data_files=str(manifest_path),
        split="train",
        streaming=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_multilingual_safety_batch,
    )


def merge_config(
    config: MultilingualSafetyDataConfig | None,
    overrides: dict[str, Any],
) -> MultilingualSafetyDataConfig:
    base = config or MultilingualSafetyDataConfig()
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return model_copy(base, **clean_overrides)


def model_copy(config: MultilingualSafetyDataConfig, **updates: Any) -> MultilingualSafetyDataConfig:
    if hasattr(config, "model_copy"):
        return config.model_copy(update=updates)
    return config.copy(update=updates)


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


def shuffle_dataset(
    dataset: Dataset | IterableDataset,
    config: MultilingualSafetyDataConfig,
) -> Dataset | IterableDataset:
    if not config.shuffle:
        return dataset
    if isinstance(dataset, IterableDataset):
        return dataset.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    return dataset.shuffle(seed=config.seed)


def limit_dataset(
    dataset: Dataset | IterableDataset,
    max_samples: int | None,
) -> Dataset | IterableDataset:
    if max_samples is None:
        return dataset
    if isinstance(dataset, IterableDataset):
        return dataset.take(max_samples)
    return dataset.select(range(min(max_samples, len(dataset))))


def as_filter_set(values: Sequence[str] | str | None) -> set[str] | None:
    items = as_list(values)
    return set(items) if items else None


def as_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]
