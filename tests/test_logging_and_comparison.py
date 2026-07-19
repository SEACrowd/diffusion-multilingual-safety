from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.paired_outputs import create_paired_outputs
from logging_utils.diffusion_gemma.logits import DiffusionLogitsLogger
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
    def test_pairing_uses_model_outputs_without_performance_streams(self) -> None:
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
            write_jsonl(root / "gemma-output.jsonl", [output])
            write_jsonl(
                root / "diffusion-output.jsonl",
                [{**output, "model_id": "diffusion-model", "run_id": "diffusion-run"}],
            )

            count = create_paired_outputs(
                gemma_outputs_path=root / "gemma-output.jsonl",
                diffusion_outputs_path=root / "diffusion-output.jsonl",
                pairs_path=root / "pairs.jsonl",
            )

            self.assertEqual(count, 1)
            pairs = list(read_jsonl(root / "pairs.jsonl"))
            self.assertEqual(pairs[0]["example_id"], "example-1")
            self.assertEqual(pairs[0]["schema_version"], 2)
            self.assertNotIn("total_latency_s", pairs[0]["gemma"])


class DiffusionLogitsTests(unittest.TestCase):
    def test_entropy_is_serialized_as_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "logits.jsonl"
            logger = DiffusionLogitsLogger(
                path,
                top_k=2,
                save_full_logits=False,
                full_logits_directory=root / "full-logits",
            )
            try:
                logger.log_step(
                    context={"run_id": "run", "canvas_index": 0, "step_index": 0},
                    logits=torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]]),
                    sampled_tokens=torch.tensor([[0, 2]]),
                    sampled_probabilities=torch.tensor([[0.6, 0.7]]),
                    accepted_mask=torch.tensor([[True, False]]),
                )
            finally:
                logger.close()

            records = list(read_jsonl(path))
            entropy = records[0]["entropy_per_position"]
            self.assertEqual(len(entropy), 2)
            self.assertTrue(all(isinstance(value, float) for value in entropy))
            self.assertTrue(all(value >= 0 for value in entropy))


if __name__ == "__main__":
    unittest.main()
