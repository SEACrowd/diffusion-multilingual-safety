from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.paired_outputs import create_paired_outputs
from logging_utils.moe import route_change_rate, summarize_router_output
from logging_utils.writer import read_jsonl, write_jsonl

try:
    from config_parser import parse_app_config
    from dataloader.collate import collate_multilingual_safety_batch
except ModuleNotFoundError:
    parse_app_config = None
    collate_multilingual_safety_batch = None


class ConfigurationAndDataTests(unittest.TestCase):
    @unittest.skipIf(parse_app_config is None, "project dependencies are not installed")
    def test_default_config_runs_both_models(self) -> None:
        assert parse_app_config is not None
        parsed = parse_app_config({"UNRELATED": "value"})
        self.assertEqual(parsed.models_to_run, ("gemma", "diffusion_gemma"))
        self.assertEqual(parsed.dataloader.batch_size, 1)

    @unittest.skipIf(
        collate_multilingual_safety_batch is None,
        "project dependencies are not installed",
    )
    def test_collate_keeps_raw_text_and_manifest_identity(self) -> None:
        assert collate_multilingual_safety_batch is not None
        batch = collate_multilingual_safety_batch(
            [
                {
                    "id": "example-1",
                    "prompt": "hello",
                    "response": "world",
                    "source": "source-a",
                    "language": "en",
                    "row_index": 4,
                    "prompt_sha256": "abc",
                }
            ]
        )
        self.assertEqual(batch["prompts"], ["hello"])
        self.assertEqual(batch["metadata"][0]["row_index"], 4)
        self.assertEqual(batch["metadata"][0]["prompt_sha256"], "abc")


class MoeSummaryTests(unittest.TestCase):
    def test_router_summary_counts_top_k_assignments(self) -> None:
        scores = torch.tensor(
            [
                [[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]],
            ]
        )

        summary, routes = summarize_router_output(scores, selected_experts=2)

        self.assertEqual(summary["tokens_routed"], 2)
        self.assertEqual(summary["routed_expert_count"], 4)
        self.assertEqual(sum(summary["expert_token_counts"]), 4)
        self.assertEqual(tuple(routes.shape), (2, 2))

    def test_route_change_rate_is_zero_for_identical_routes(self) -> None:
        routes = torch.tensor([[1, 2], [2, 3]])
        self.assertEqual(route_change_rate(routes, routes.clone()), 0.0)
        self.assertIsNone(route_change_rate(routes, None))


class PairedOutputTests(unittest.TestCase):
    def test_pairing_filters_diffusion_step_performance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = {
                "event": "final_output",
                "experiment_id": "test",
                "example_id": "example-1",
                "row_index": 0,
                "prompt_sha256": "abc",
                "source": "source-a",
                "language": "en",
                "prompt_label": "safe",
                "response_label": "",
                "violated_categories": "",
                "tag": "",
                "prompt": "hello",
                "reference_response": "",
                "model_id": "model",
                "model_revision": "revision",
                "run_id": "run",
                "seed": 1,
                "generation_configuration": {},
                "final_text": "world",
                "final_token_ids": [1],
                "input_token_count": 1,
                "output_token_count": 1,
            }
            performance = {
                "event": "inference_performance",
                "example_id": "example-1",
                "total_latency_s": 1.0,
                "tokens_per_second": 1.0,
                "cuda_memory": [],
                "instrumented": True,
            }
            write_jsonl(root / "gemma-output.jsonl", [output])
            write_jsonl(root / "gemma-performance.jsonl", [performance])
            write_jsonl(root / "diffusion-output.jsonl", [output])
            write_jsonl(
                root / "diffusion-performance.jsonl",
                [{"event": "performance_step"}, performance],
            )

            count = create_paired_outputs(
                gemma_outputs_path=root / "gemma-output.jsonl",
                gemma_performance_path=root / "gemma-performance.jsonl",
                diffusion_outputs_path=root / "diffusion-output.jsonl",
                diffusion_performance_path=root / "diffusion-performance.jsonl",
                pairs_path=root / "pairs.jsonl",
                source_summary_path=root / "summary.json",
            )

            self.assertEqual(count, 1)
            pairs = list(read_jsonl(root / "pairs.jsonl"))
            self.assertEqual(pairs[0]["example_id"], "example-1")


if __name__ == "__main__":
    unittest.main()
