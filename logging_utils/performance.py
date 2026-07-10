"""Inference timing, CUDA memory metrics, and source aggregation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .writer import JsonlWriter, read_jsonl, write_json


class PerformanceLogger:
    def __init__(self, path: str | Path) -> None:
        self.writer = JsonlWriter(path)

    def log_step(
        self,
        *,
        context: dict[str, Any],
        step_latency: float,
        scheduler_latency: float,
        changed_count: int,
        accepted_count: int,
    ) -> None:
        self.writer.write(
            {
                "event": "performance_step",
                **context,
                "step_latency_s": step_latency,
                "scheduler_latency_s": scheduler_latency,
                "changed_count": changed_count,
                "accepted_count": accepted_count,
            }
        )

    def log_inference(
        self,
        *,
        context: dict[str, Any],
        total_latency: float,
        input_token_count: int,
        output_token_count: int,
        instrumented: bool,
        cuda_memory: list[dict[str, int]],
    ) -> dict[str, Any]:
        record = {
            "event": "inference_performance",
            **context,
            "total_latency_s": total_latency,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "tokens_per_second": (
                output_token_count / total_latency if total_latency > 0 else None
            ),
            "instrumented": instrumented,
            "cuda_memory": cuda_memory,
        }
        self.writer.write(record)
        return record

    def close(self) -> None:
        self.writer.close()


def synchronize_cuda() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def reset_cuda_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def cuda_peak_memory() -> list[dict[str, int]]:
    if not torch.cuda.is_available():
        return []
    return [
        {
            "device_index": device_index,
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
        }
        for device_index in range(torch.cuda.device_count())
    ]


def write_source_summary_from_jsonl(
    performance_path: str | Path,
    summary_path: str | Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    for record in read_jsonl(performance_path):
        if record.get("event") != "inference_performance":
            continue
        grouped[str(record.get("source") or "unknown")].append(record)
        total += 1

    write_json(
        summary_path,
        {
            "examples": total,
            "by_source": {
                source: summarize_performance(items)
                for source, items in sorted(grouped.items())
            },
        },
    )


def summarize_performance(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["total_latency_s"]) for record in records]
    output_tokens = [int(record["output_token_count"]) for record in records]
    return {
        "count": len(records),
        "mean_total_latency_s": sum(latencies) / len(latencies),
        "mean_output_tokens": sum(output_tokens) / len(output_tokens),
        "total_output_tokens": sum(output_tokens),
        "aggregate_tokens_per_second": (
            sum(output_tokens) / sum(latencies) if sum(latencies) > 0 else None
        ),
    }
