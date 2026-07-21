from __future__ import annotations

import unittest

try:
    from smoke_test_pipeline import build_smoke_config
except ModuleNotFoundError:
    build_smoke_config = None


class SmokePipelineConfigurationTests(unittest.TestCase):
    @unittest.skipIf(build_smoke_config is None, "project dependencies are not installed")
    def test_smoke_config_processes_all_ten_unshuffled_examples(self) -> None:
        assert build_smoke_config is not None
        config = build_smoke_config(
            samples=10,
            experiment_id="smoke-test",
            environ={"UNRELATED": "value", "INFERENCE_MAX_BATCHES": "1"},
        )

        self.assertEqual(config.data.max_samples, 10)
        self.assertFalse(config.data.shuffle)
        self.assertIsNone(config.inference_max_batches)
        self.assertEqual(config.models_to_run, ("gemma", "diffusion_gemma"))
        self.assertEqual(config.thinking_variants, ("non_thinking", "thinking"))
        self.assertEqual(config.logging.experiment_id, "smoke-test")

    @unittest.skipIf(build_smoke_config is None, "project dependencies are not installed")
    def test_smoke_config_can_select_one_model(self) -> None:
        assert build_smoke_config is not None
        config = build_smoke_config(
            samples=2,
            models=("gemma",),
            experiment_id="smoke-gemma",
            output_root="smoke-logs",
            environ={"UNRELATED": "value"},
        )

        self.assertEqual(config.models_to_run, ("gemma",))
        self.assertEqual(config.logging.root, "smoke-logs")


if __name__ == "__main__":
    unittest.main()
