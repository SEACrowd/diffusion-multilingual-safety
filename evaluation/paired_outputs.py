"""Strict streaming join of Gemma and DiffusionGemma result streams."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Any

from logging_utils.writer import JsonlWriter, read_jsonl


def create_paired_outputs(
    *,
    gemma_outputs_path: str | Path,
    diffusion_outputs_path: str | Path,
    pairs_path: str | Path,
) -> int:
    streams = (
        read_events(gemma_outputs_path, "final_output"),
        read_events(diffusion_outputs_path, "final_output"),
    )
    missing = object()
    count = 0
    with JsonlWriter(pairs_path) as writer:
        for rows in zip_longest(*streams, fillvalue=missing):
            if missing in rows:
                raise ValueError("Model output streams have different lengths")
            gemma_output, diffusion_output = rows
            example_ids = {str(row.get("example_id", "")) for row in rows}
            if len(example_ids) != 1:
                raise ValueError(f"Cannot pair mismatched example IDs: {sorted(example_ids)}")
            prompt_hashes = {
                str(gemma_output.get("prompt_sha256", "")),
                str(diffusion_output.get("prompt_sha256", "")),
            }
            if len(prompt_hashes) != 1:
                raise ValueError(f"Prompt mismatch for example {next(iter(example_ids))}")

            gemma_result = model_result(gemma_output)
            diffusion_result = model_result(diffusion_output)
            record = {
                "event": "paired_output",
                "schema_version": 2,
                "experiment_id": gemma_output["experiment_id"],
                "example_id": gemma_output["example_id"],
                "row_index": gemma_output.get("row_index"),
                "prompt_sha256": gemma_output.get("prompt_sha256", ""),
                "source": gemma_output.get("source", ""),
                "language": gemma_output.get("language", ""),
                "prompt_label": gemma_output.get("prompt_label", ""),
                "response_label": gemma_output.get("response_label", ""),
                "violated_categories": gemma_output.get("violated_categories", ""),
                "tag": gemma_output.get("tag", ""),
                "prompt": gemma_output["prompt"],
                "reference_response": gemma_output.get("reference_response", ""),
                "gemma": gemma_result,
                "diffusion_gemma": diffusion_result,
                "output_token_difference": (
                    diffusion_result["output_token_count"]
                    - gemma_result["output_token_count"]
                ),
            }
            writer.write(record)
            count += 1
    return count


def model_result(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": output["model_id"],
        "model_revision": output["model_revision"],
        "run_id": output["run_id"],
        "seed": output["seed"],
        "generation_configuration": output["generation_configuration"],
        "final_text": output["final_text"],
        "final_token_ids": output["final_token_ids"],
        "input_token_count": output["input_token_count"],
        "output_token_count": output["output_token_count"],
    }


def read_events(path: str | Path, event: str):
    for record in read_jsonl(path):
        if record.get("event") == event:
            yield record
