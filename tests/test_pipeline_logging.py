from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from logging_utils.diffusion_gemma.callback import DiffusionLoggingCallback
from logging_utils.diffusion_gemma.logits import DiffusionLogitsLogger
from logging_utils.moe import route_change_rate, summarize_router_output
from logging_utils.writer import read_jsonl
from models.common import parse_response_text

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
        self.assertTrue(parsed.logging.log_logits)
        self.assertTrue(parsed.logging.log_moe)

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


class ResponseParsingTests(unittest.TestCase):
    def test_parser_failure_preserves_raw_response(self) -> None:
        class BrokenProcessor:
            @staticmethod
            def parse_response(raw_text: str) -> str:
                raise ValueError("bad response template")

        with self.assertWarns(RuntimeWarning):
            response = parse_response_text(BrokenProcessor(), "raw answer")
        self.assertEqual(response, "raw answer")


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


class DiffusionTelemetryTests(unittest.TestCase):
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

    def test_broken_logits_logger_does_not_abort_callback(self) -> None:
        class BrokenLogitsLogger:
            calls = 0

            def log_step(self, **kwargs) -> None:
                self.calls += 1
                raise TypeError("broken optional logger")

        broken_logger = BrokenLogitsLogger()
        disabled: set[str] = set()
        callback = DiffusionLoggingCallback(
            context={"run_id": "run"},
            canvas_logger=None,
            logits_logger=broken_logger,  # type: ignore[arg-type]
            router_tracer=None,
            disabled_components=disabled,
        )
        scheduler = SimpleNamespace(
            last_trace={
                "accepted_index": torch.tensor([[True, False]]),
                "sampled_tokens": torch.tensor([[1, 2]]),
                "sampled_probs": torch.tensor([[0.8, 0.4]]),
                "pred_logits": torch.tensor([[[2.0, 1.0], [1.0, 2.0]]]),
            }
        )
        pipe = SimpleNamespace(scheduler=scheduler)

        with self.assertWarns(RuntimeWarning):
            result = callback(
                pipe,
                global_step=0,
                step_index=0,
                callback_kwargs={"canvas": torch.tensor([[1, 2]])},
            )

        self.assertEqual(result, {})
        self.assertIn("logits", disabled)
        self.assertEqual(broken_logger.calls, 1)

        callback(
            pipe,
            global_step=1,
            step_index=1,
            callback_kwargs={"canvas": torch.tensor([[1, 2]])},
        )
        self.assertEqual(broken_logger.calls, 1)


if __name__ == "__main__":
    unittest.main()
