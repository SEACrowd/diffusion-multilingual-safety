"""Strict streaming join of Gemma and DiffusionGemma result streams."""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from logging_utils.writer import JsonlWriter, read_jsonl, write_json


def create_paired_outputs(
    *,
    gemma_outputs_path: str | Path,
    gemma_performance_path: str | Path,
    diffusion_outputs_path: str | Path,
    diffusion_performance_path: str | Path,
    pairs_path: str | Path,
    source_summary_path: str | Path,
) -> int:
    streams = (
        read_events(gemma_outputs_path, "final_output"),
        read_events(gemma_performance_path, "inference_performance"),
        read_events(diffusion_outputs_path, "final_output"),
        read_events(diffusion_performance_path, "inference_performance"),
    )
    missing = object()
    source_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "gemma_latency_s": 0.0,
            "diffusion_gemma_latency_s": 0.0,
            "gemma_output_tokens": 0,
            "diffusion_gemma_output_tokens": 0,
        }
    )
    count = 0
    with JsonlWriter(pairs_path) as writer:
        for rows in zip_longest(*streams, fillvalue=missing):
            if missing in rows:
                raise ValueError("Model output and performance streams have different lengths")
            gemma_output, gemma_performance, diffusion_output, diffusion_performance = rows
            example_ids = {
                str(row.get("example_id", ""))
                for row in rows
            }
            if len(example_ids) != 1:
                raise ValueError(f"Cannot pair mismatched example IDs: {sorted(example_ids)}")
            prompt_hashes = {
                str(gemma_output.get("prompt_sha256", "")),
                str(diffusion_output.get("prompt_sha256", "")),
            }
            if len(prompt_hashes) != 1:
                raise ValueError(f"Prompt mismatch for example {next(iter(example_ids))}")

            gemma_result = model_result(gemma_output, gemma_performance)
            diffusion_result = model_result(diffusion_output, diffusion_performance)
            gemma_latency = float(gemma_performance["total_latency_s"])
            diffusion_latency = float(diffusion_performance["total_latency_s"])
            record = {
                "event": "paired_output",
                "schema_version": 1,
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
                "latency_ratio_diffusion_over_gemma": (
                    diffusion_latency / gemma_latency if gemma_latency > 0 else None
                ),
                "output_token_difference": (
                    diffusion_result["output_token_count"]
                    - gemma_result["output_token_count"]
                ),
            }
            writer.write(record)
            update_source_totals(
                source_totals[str(record["source"] or "unknown")],
                gemma_result,
                diffusion_result,
            )
            count += 1

    write_json(
        source_summary_path,
        {
            "examples": count,
            "by_source": {
                source: finalize_source_totals(totals)
                for source, totals in sorted(source_totals.items())
            },
        },
    )
    return count


def model_result(output: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    if output.get("example_id") != performance.get("example_id"):
        raise ValueError("A model output does not match its performance record")
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
        "total_latency_s": performance["total_latency_s"],
        "tokens_per_second": performance["tokens_per_second"],
        "cuda_memory": performance["cuda_memory"],
        "instrumented": performance["instrumented"],
    }


def read_events(path: str | Path, event: str):
    for record in read_jsonl(path):
        if record.get("event") == event:
            yield record


def update_source_totals(
    totals: dict[str, float],
    gemma: dict[str, Any],
    diffusion: dict[str, Any],
) -> None:
    totals["count"] += 1
    totals["gemma_latency_s"] += float(gemma["total_latency_s"])
    totals["diffusion_gemma_latency_s"] += float(diffusion["total_latency_s"])
    totals["gemma_output_tokens"] += int(gemma["output_token_count"])
    totals["diffusion_gemma_output_tokens"] += int(diffusion["output_token_count"])


def finalize_source_totals(totals: dict[str, float]) -> dict[str, float]:
    count = totals["count"]
    return {
        "count": int(count),
        "gemma_mean_latency_s": totals["gemma_latency_s"] / count,
        "diffusion_gemma_mean_latency_s": (
            totals["diffusion_gemma_latency_s"] / count
        ),
        "gemma_aggregate_tokens_per_second": safe_ratio(
            totals["gemma_output_tokens"], totals["gemma_latency_s"]
        ),
        "diffusion_gemma_aggregate_tokens_per_second": (
            safe_ratio(
                totals["diffusion_gemma_output_tokens"],
                totals["diffusion_gemma_latency_s"],
            )
        ),
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None
