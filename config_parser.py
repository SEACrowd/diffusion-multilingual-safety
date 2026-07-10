"""Pydantic parser for environment-backed application config."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel

import config
from dataloader import MultilingualSafetyDataConfig


class ModelConfig(BaseModel):
    model_name: str
    processor_name: str
    dtype: str
    device_map: str | None
    inference_max_batches: int | None


class DataLoaderConfig(BaseModel):
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    max_length: int
    mask_prompt_labels: bool
    return_metadata: bool
    shuffle_buffer_size: int


class AppConfig(BaseModel):
    model: ModelConfig
    data: MultilingualSafetyDataConfig
    dataloader: DataLoaderConfig


def parse_app_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    env = environ or os.environ

    return AppConfig(
        model=ModelConfig(
            model_name=parse_str(env, "HF_MODEL_NAME", config.HF_MODEL_NAME),
            processor_name=parse_str(
                env,
                "HF_PROCESSOR_NAME",
                config.HF_PROCESSOR_NAME,
            ),
            dtype=parse_str(env, "MODEL_DTYPE", config.MODEL_DTYPE),
            device_map=parse_optional_str(env, "MODEL_DEVICE_MAP", config.MODEL_DEVICE_MAP),
            inference_max_batches=parse_optional_int(
                env,
                "INFERENCE_MAX_BATCHES",
                config.INFERENCE_MAX_BATCHES,
            ),
        ),
        data=MultilingualSafetyDataConfig(
            dataset_name=parse_str(env, "DATASET_NAME", config.DATASET_NAME),
            split=parse_str(env, "DATASET_SPLIT", config.DATASET_SPLIT),
            streaming=parse_bool(env, "DATASET_STREAMING", config.DATASET_STREAMING),
            cache_dir=parse_optional_str(env, "HF_DATASETS_CACHE", config.HF_DATASETS_CACHE),
            max_samples=parse_optional_int(
                env,
                "DATASET_MAX_SAMPLES",
                config.DATASET_MAX_SAMPLES,
            ),
            languages=parse_list(env, "DATASET_LANGUAGES", config.DATASET_LANGUAGES),
            prompt_labels=parse_list(
                env,
                "DATASET_PROMPT_LABELS",
                config.DATASET_PROMPT_LABELS,
            ),
            response_labels=parse_list(
                env,
                "DATASET_RESPONSE_LABELS",
                config.DATASET_RESPONSE_LABELS,
            ),
            tags=parse_list(env, "DATASET_TAGS", config.DATASET_TAGS),
            sources=parse_list(env, "DATASET_SOURCES", config.DATASET_SOURCES),
            seed=parse_int(env, "DATASET_SEED", config.DATASET_SEED),
            filter_num_proc=parse_optional_int(
                env,
                "DATASET_FILTER_NUM_PROC",
                config.DATASET_FILTER_NUM_PROC,
            ),
            map_num_proc=parse_optional_int(
                env,
                "DATASET_MAP_NUM_PROC",
                config.DATASET_MAP_NUM_PROC,
            ),
            normalize=parse_bool(env, "DATASET_NORMALIZE", config.DATASET_NORMALIZE),
        ),
        dataloader=DataLoaderConfig(
            batch_size=parse_int(env, "BATCH_SIZE", config.BATCH_SIZE),
            shuffle=parse_bool(env, "SHUFFLE_DATASET", config.SHUFFLE_DATASET),
            num_workers=parse_int(env, "NUM_WORKERS", config.NUM_WORKERS),
            pin_memory=parse_bool(env, "PIN_MEMORY", config.PIN_MEMORY),
            max_length=parse_int(env, "MAX_LENGTH", config.MAX_LENGTH),
            mask_prompt_labels=parse_bool(
                env,
                "MASK_PROMPT_LABELS",
                config.MASK_PROMPT_LABELS,
            ),
            return_metadata=parse_bool(env, "RETURN_METADATA", config.RETURN_METADATA),
            shuffle_buffer_size=parse_int(
                env,
                "SHUFFLE_BUFFER_SIZE",
                config.SHUFFLE_BUFFER_SIZE,
            ),
        ),
    )


def parse_str(env: Mapping[str, str], name: str, default: str) -> str:
    value = parse_optional_str(env, name, default)
    if value is None:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def parse_optional_str(
    env: Mapping[str, str],
    name: str,
    default: str | None = None,
) -> str | None:
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = parse_optional_int(env, name, default)
    if value is None:
        raise ValueError(f"{name} must be an integer")
    return value


def parse_optional_int(
    env: Mapping[str, str],
    name: str,
    default: int | None = None,
) -> int | None:
    value = parse_optional_str(env, name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = parse_optional_str(env, name)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def parse_list(
    env: Mapping[str, str],
    name: str,
    default: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...] | None:
    value = parse_optional_str(env, name)
    if value is None:
        items = tuple(default or ())
    else:
        items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None
