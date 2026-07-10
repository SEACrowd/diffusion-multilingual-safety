"""Diffusion canvas vocabulary-logit summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from logging_utils.writer import JsonlWriter


class DiffusionLogitsLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        top_k: int,
        save_full_logits: bool,
        full_logits_directory: str | Path,
    ) -> None:
        self.writer = JsonlWriter(path)
        self.top_k = top_k
        self.save_full_logits = save_full_logits
        self.full_logits_directory = Path(full_logits_directory)

    def log_step(
        self,
        *,
        context: dict[str, Any],
        logits: torch.Tensor,
        sampled_tokens: torch.Tensor,
        sampled_probabilities: torch.Tensor,
        accepted_mask: torch.Tensor,
    ) -> None:
        row_logits = logits[0].detach().float()
        k = min(self.top_k, row_logits.shape[-1])
        log_normalizer = torch.logsumexp(row_logits, dim=-1, keepdim=True)
        top_values, top_tokens = torch.topk(row_logits, k=k, dim=-1)
        log_probabilities = row_logits - log_normalizer
        record: dict[str, Any] = {
            "event": "logits_step",
            **context,
            "top_k_token_ids": top_tokens.cpu().tolist(),
            "top_k_probabilities": torch.exp(top_values - log_normalizer).cpu().tolist(),
            "entropy_per_position": (
                -(log_probabilities.exp() * log_probabilities).sum(dim=-1).cpu().tolist()
            ),
            "sampled_token_ids": sampled_tokens[0].detach().cpu().tolist(),
            "sampled_token_probabilities": (
                sampled_probabilities[0].detach().float().cpu().tolist()
            ),
            "accepted_token_probabilities": (
                sampled_probabilities[0][accepted_mask[0]].detach().float().cpu().tolist()
            ),
        }
        if self.save_full_logits:
            target = self.full_logits_directory / context["run_id"]
            target.mkdir(parents=True, exist_ok=True)
            tensor_path = target / (
                f"canvas-{context['canvas_index']:03d}-step-{context['step_index']:03d}.pt"
            )
            torch.save(logits.detach().cpu(), tensor_path)
            record["full_logits_path"] = str(tensor_path)
        self.writer.write(record)

    def close(self) -> None:
        self.writer.close()
