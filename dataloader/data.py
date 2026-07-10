"""Schema and normalization for the multilingual safety dataset."""

from __future__ import annotations

from collections.abc import Mapping
from math import isnan
from typing import Any

from pydantic import BaseModel

DATASET_NAME = "tackhwa/multilingual_safety"
DATASET_COLUMNS = (
    "id",
    "prompt",
    "response",
    "prompt_label",
    "response_label",
    "violated_categories",
    "prompt_label_source",
    "response_label_source",
    "tag",
    "language",
    "reconstruction_id_if_redacted",
    "source",
)
EXPECTED_COLUMNS = DATASET_COLUMNS
METADATA_COLUMNS = (
    "id",
    "prompt_label",
    "response_label",
    "violated_categories",
    "prompt_label_source",
    "response_label_source",
    "tag",
    "language",
    "reconstruction_id_if_redacted",
    "source",
)


class MultilingualSafetyExample(BaseModel):
    id: str
    prompt: str
    response: str
    prompt_label: str
    response_label: str
    violated_categories: str
    prompt_label_source: str
    response_label_source: str
    tag: str
    language: str
    reconstruction_id_if_redacted: float | None
    source: str


def parse_multilingual_safety_row(row: Mapping[str, Any]) -> MultilingualSafetyExample:
    return MultilingualSafetyExample(
        id=clean_text(row.get("id")),
        prompt=clean_text(row.get("prompt")),
        response=clean_text(row.get("response")),
        prompt_label=clean_text(row.get("prompt_label")),
        response_label=clean_text(row.get("response_label")),
        violated_categories=clean_text(row.get("violated_categories")),
        prompt_label_source=clean_text(row.get("prompt_label_source")),
        response_label_source=clean_text(row.get("response_label_source")),
        tag=clean_text(row.get("tag")),
        language=clean_text(row.get("language")),
        reconstruction_id_if_redacted=clean_optional_float(
            row.get("reconstruction_id_if_redacted")
        ),
        source=clean_text(row.get("source")),
    )


def normalize_multilingual_safety_row(row: Mapping[str, Any]) -> dict[str, Any]:
    example = parse_multilingual_safety_row(row)
    if hasattr(example, "model_dump"):
        return example.model_dump()
    return example.dict()


def metadata_from_row(row: Mapping[str, Any] | MultilingualSafetyExample) -> dict[str, Any]:
    return {column: row_value(row, column) for column in METADATA_COLUMNS}


def row_value(
    row: Mapping[str, Any] | MultilingualSafetyExample,
    column: str,
    default: Any = None,
) -> Any:
    if isinstance(row, MultilingualSafetyExample):
        return getattr(row, column, default)
    return row.get(column, default)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and isnan(value):
        return ""
    return str(value).strip()


def clean_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if isnan(number):
        return None
    return number

