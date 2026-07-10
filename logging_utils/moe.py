"""Shared Mixture-of-Experts router reductions."""

from __future__ import annotations

import re
from typing import Any

import torch


LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.router$")


def router_layer_index(module_name: str) -> int:
    match = LAYER_PATTERN.search(module_name)
    if match is None:
        raise ValueError(f"Cannot determine router layer from {module_name!r}")
    return int(match.group(1))


def router_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Router returned unsupported output type {type(output).__name__}")


def summarize_router_output(
    output: Any,
    *,
    selected_experts: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    scores = router_tensor(output).detach().float()
    scores = scores.reshape(-1, scores.shape[-1])
    if scores.numel() == 0:
        raise ValueError("Router returned an empty tensor")

    probabilities = normalize_router_scores(scores)
    k = min(selected_experts, probabilities.shape[-1])
    top_probabilities, top_expert_ids = torch.topk(probabilities, k=k, dim=-1)
    counts = torch.bincount(
        top_expert_ids.reshape(-1),
        minlength=probabilities.shape[-1],
    )
    load = counts.float() / max(1, top_expert_ids.numel())
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
    margin = (
        top_probabilities[:, 0] - top_probabilities[:, 1]
        if k > 1
        else top_probabilities[:, 0]
    )
    load_mean = load.mean()
    load_cv = load.std(unbiased=False) / load_mean if load_mean > 0 else torch.tensor(0.0)
    busiest_count = min(8, counts.numel())
    busiest_loads, busiest_ids = torch.topk(load, k=busiest_count)

    summary = {
        "tokens_routed": scores.shape[0],
        "routed_expert_count": scores.shape[-1],
        "selected_experts_per_token": k,
        "shared_expert_active": True,
        "expert_token_counts": counts.cpu().tolist(),
        "expert_load_fraction": load.cpu().tolist(),
        "router_entropy_mean": entropy.mean().item(),
        "router_entropy_p95": torch.quantile(entropy, 0.95).item(),
        "top1_router_margin_mean": margin.mean().item(),
        "expert_load_cv": load_cv.item(),
        "most_used_experts": [
            {"expert_id": int(expert_id), "load_fraction": float(expert_load)}
            for expert_id, expert_load in zip(
                busiest_ids.cpu().tolist(),
                busiest_loads.cpu().tolist(),
                strict=True,
            )
        ],
    }
    return summary, top_expert_ids.cpu()


def normalize_router_scores(scores: torch.Tensor) -> torch.Tensor:
    row_sums = scores.sum(dim=-1)
    looks_normalized = bool(
        scores.min().item() >= 0
        and torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-3,
            rtol=1e-3,
        )
    )
    return scores if looks_normalized else scores.softmax(dim=-1)


def route_change_rate(current: torch.Tensor, previous: torch.Tensor | None) -> float | None:
    if previous is None or current.shape != previous.shape:
        return None
    intersection = (
        current.unsqueeze(-1).eq(previous.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
    )
    union = current.shape[-1] * 2 - intersection
    jaccard = intersection / union.clamp_min(1)
    return float(1 - jaccard.mean().item())
