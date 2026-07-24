"""Typed parsing for application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field

import config
from dataloader import MultilingualSafetyDataConfig


class ModelConfig(BaseModel):
    model_name: str
    processor_name: str
    revision: str
    dtype: str
    device_map: str | None


class GemmaGenerationConfig(BaseModel):
    max_new_tokens: int = Field(gt=0)
    enable_thinking: bool
    do_sample: bool
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)


class DiffusionGemmaGenerationConfig(BaseModel):
    gen_length: int = Field(gt=0)
    max_denoising_steps: int = Field(gt=0)
    t_min: float = Field(gt=0)
    t_max: float = Field(gt=0)
    entropy_bound: float = Field(gt=0)
    stability_threshold: int = Field(gt=0)
    confidence_threshold: float | None = Field(default=None, gt=0)


class LoggingConfig(BaseModel):
    root: str
    experiment_id: str | None
    top_k: int = Field(gt=0)
    log_moe: bool
    log_tokens: bool
    log_logits: bool
    save_full_logits: bool
    seed: int


class DataLoaderConfig(BaseModel):
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    pin_memory: bool


class AppConfig(BaseModel):
    models_to_run: tuple[Literal["gemma", "diffusion_gemma"], ...]
    inference_max_batches: int | None = Field(default=None, gt=0)
    gemma_model: ModelConfig
    diffusion_gemma_model: ModelConfig
    gemma_generation: GemmaGenerationConfig
    diffusion_gemma_generation: DiffusionGemmaGenerationConfig
    logging: LoggingConfig
    data: MultilingualSafetyDataConfig
    dataloader: DataLoaderConfig


def parse_app_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    env = environ or os.environ
    models_to_run = parse_list(env, "MODELS_TO_RUN", config.MODELS_TO_RUN)
    if not models_to_run:
        raise ValueError("MODELS_TO_RUN must contain at least one model")

    return AppConfig(
        models_to_run=tuple(dict.fromkeys(models_to_run)),
        inference_max_batches=parse_optional_int(
            env,
            "INFERENCE_MAX_BATCHES",
            config.INFERENCE_MAX_BATCHES,
        ),
        gemma_model=parse_model_config(env, "GEMMA", config.GEMMA_MODEL_NAME),
        diffusion_gemma_model=parse_model_config(
            env,
            "DIFFUSION_GEMMA",
            config.DIFFUSION_GEMMA_MODEL_NAME,
        ),
        gemma_generation=GemmaGenerationConfig(
            max_new_tokens=parse_int(env, "GEMMA_MAX_NEW_TOKENS", config.GEMMA_MAX_NEW_TOKENS),
            enable_thinking=parse_bool(
                env,
                "GEMMA_ENABLE_THINKING",
                config.GEMMA_ENABLE_THINKING,
            ),
            do_sample=parse_bool(env, "GEMMA_DO_SAMPLE", config.GEMMA_DO_SAMPLE),
            temperature=parse_float(env, "GEMMA_TEMPERATURE", config.GEMMA_TEMPERATURE),
            top_p=parse_float(env, "GEMMA_TOP_P", config.GEMMA_TOP_P),
            top_k=parse_int(env, "GEMMA_TOP_K", config.GEMMA_TOP_K),
        ),
        diffusion_gemma_generation=DiffusionGemmaGenerationConfig(
            gen_length=parse_int(
                env,
                "DIFFUSION_GEMMA_GEN_LENGTH",
                config.DIFFUSION_GEMMA_GEN_LENGTH,
            ),
            max_denoising_steps=parse_int(
                env,
                "DIFFUSION_GEMMA_MAX_DENOISING_STEPS",
                config.DIFFUSION_GEMMA_MAX_DENOISING_STEPS,
            ),
            t_min=parse_float(env, "DIFFUSION_GEMMA_T_MIN", config.DIFFUSION_GEMMA_T_MIN),
            t_max=parse_float(env, "DIFFUSION_GEMMA_T_MAX", config.DIFFUSION_GEMMA_T_MAX),
            entropy_bound=parse_float(
                env,
                "DIFFUSION_GEMMA_ENTROPY_BOUND",
                config.DIFFUSION_GEMMA_ENTROPY_BOUND,
            ),
            stability_threshold=parse_int(
                env,
                "DIFFUSION_GEMMA_STABILITY_THRESHOLD",
                config.DIFFUSION_GEMMA_STABILITY_THRESHOLD,
            ),
            confidence_threshold=parse_optional_float(
                env,
                "DIFFUSION_GEMMA_CONFIDENCE_THRESHOLD",
                config.DIFFUSION_GEMMA_CONFIDENCE_THRESHOLD,
            ),
        ),
        logging=LoggingConfig(
            root=parse_str(env, "LOGGING_ROOT", config.LOGGING_ROOT),
            experiment_id=parse_optional_str(
                env,
                "LOGGING_EXPERIMENT_ID",
                config.LOGGING_EXPERIMENT_ID,
            ),
            top_k=parse_int(env, "LOG_TOP_K", config.LOG_TOP_K),
            log_moe=parse_bool(env, "LOG_MOE", config.LOG_MOE),
            log_tokens=parse_bool(env, "LOG_TOKENS", config.LOG_TOKENS),
            log_logits=parse_bool(env, "LOG_LOGITS", config.LOG_LOGITS),
            save_full_logits=parse_bool(
                env,
                "LOG_SAVE_FULL_LOGITS",
                config.LOG_SAVE_FULL_LOGITS,
            ),
            seed=parse_int(env, "LOG_SEED", config.LOG_SEED),
        ),
        data=MultilingualSafetyDataConfig(
            dataset_name=parse_str(env, "DATASET_NAME", config.DATASET_NAME),
            revision=parse_str(env, "DATASET_REVISION", config.DATASET_REVISION),
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
            shuffle=parse_bool(env, "DATASET_SHUFFLE", config.DATASET_SHUFFLE),
            shuffle_buffer_size=parse_int(
                env,
                "DATASET_SHUFFLE_BUFFER_SIZE",
                config.DATASET_SHUFFLE_BUFFER_SIZE,
            ),
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
            num_workers=parse_int(env, "NUM_WORKERS", config.NUM_WORKERS),
            pin_memory=parse_bool(env, "PIN_MEMORY", config.PIN_MEMORY),
        ),
    )


def parse_model_config(env: Mapping[str, str], prefix: str, default_name: str) -> ModelConfig:
    default_processor = getattr(config, f"{prefix}_PROCESSOR_NAME")
    default_revision = getattr(config, f"{prefix}_MODEL_REVISION")
    default_dtype = getattr(config, f"{prefix}_MODEL_DTYPE")
    default_device_map = getattr(config, f"{prefix}_MODEL_DEVICE_MAP")
    return ModelConfig(
        model_name=parse_str(env, f"{prefix}_MODEL_NAME", default_name),
        processor_name=parse_str(env, f"{prefix}_PROCESSOR_NAME", default_processor),
        revision=parse_str(env, f"{prefix}_MODEL_REVISION", default_revision),
        dtype=parse_str(env, f"{prefix}_MODEL_DTYPE", default_dtype),
        device_map=parse_optional_str(env, f"{prefix}_MODEL_DEVICE_MAP", default_device_map),
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


def parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = parse_optional_float(env, name, default)
    if value is None:
        raise ValueError(f"{name} must be a float")
    return value


def parse_optional_float(
    env: Mapping[str, str],
    name: str,
    default: float | None = None,
) -> float | None:
    value = parse_optional_str(env, name)
    if value is None:
        return default
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float or None, got {value!r}") from exc


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
