"""Autoregressive vocabulary-logit summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from logging_utils.writer import JsonlWriter


class GemmaLogitsLogger:
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

    def log_generation(
        self,
        *,
        context: dict[str, Any],
        scores: tuple[torch.Tensor, ...],
        generated_token_ids: list[int],
    ) -> None:
        if len(scores) != len(generated_token_ids):
            raise ValueError("Generation score count does not match generated token count")

        for token_step, (step_scores, token_id) in enumerate(
            zip(scores, generated_token_ids, strict=True)
        ):
            row_scores = step_scores[0].detach().float()
            log_normalizer = torch.logsumexp(row_scores, dim=-1)
            log_probabilities = row_scores - log_normalizer
            probabilities = log_probabilities.exp()
            k = min(self.top_k, row_scores.shape[-1])
            top_values, top_ids = torch.topk(row_scores, k=k)
            record: dict[str, Any] = {
                "event": "logits_step",
                **context,
                "token_step": token_step,
                "selected_token_id": token_id,
                "selected_token_probability": probabilities[token_id].item(),
                "top_k_token_ids": top_ids.cpu().tolist(),
                "top_k_probabilities": torch.exp(top_values - log_normalizer).cpu().tolist(),
                "entropy": torch.special.entr(probabilities).sum().item(),
            }
            if self.save_full_logits:
                target = self.full_logits_directory / context["run_id"]
                target.mkdir(parents=True, exist_ok=True)
                tensor_path = target / f"token-{token_step:04d}.pt"
                torch.save(step_scores.detach().cpu(), tensor_path)
                record["full_logits_path"] = str(tensor_path)
            self.writer.write(record)

    def close(self) -> None:
        self.writer.close()
