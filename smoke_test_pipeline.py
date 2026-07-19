"""Run and validate a small, real end-to-end pipeline experiment.

This runner intentionally uses the normal model and dataset implementations.  It
only caps the input manifest and removes the independent batch limit so every
selected smoke-test example reaches every selected model.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from config_parser import AppConfig, parse_app_config


MODEL_CHOICES = ("gemma", "diffusion_gemma")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real pipeline on the first few dataset examples.",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=10,
        help="Number of input examples to process (default: 10).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="Models to run (default: both models).",
    )
    parser.add_argument(
        "--experiment-id",
        help="Optional fixed experiment ID; by default a unique smoke-test ID is used.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional logging directory override (default: LOGGING_ROOT from config.py).",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_smoke_config(
    *,
    samples: int = 10,
    models: Sequence[str] = MODEL_CHOICES,
    experiment_id: str | None = None,
    output_root: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Build a normal application config with only smoke-run limits changed."""
    if samples <= 0:
        raise ValueError("samples must be greater than zero")
    unknown_models = sorted(set(models) - set(MODEL_CHOICES))
    if unknown_models:
        raise ValueError(f"Unsupported models: {', '.join(unknown_models)}")
    if not models:
        raise ValueError("At least one model must be selected")

    parsed = parse_app_config(environ)
    smoke_id = experiment_id or make_experiment_id(samples)
    data_config = validated_copy(
        parsed.data,
        max_samples=samples,
        shuffle=False,
    )
    logging_updates: dict[str, Any] = {"experiment_id": smoke_id}
    if output_root is not None:
        logging_updates["root"] = output_root
    logging_config = validated_copy(parsed.logging, **logging_updates)

    return validated_copy(
        parsed,
        models_to_run=tuple(dict.fromkeys(models)),
        # The manifest is already capped.  Disabling the batch cap guarantees
        # that batch size cannot make a 10-example smoke test stop early.
        inference_max_batches=None,
        data=data_config,
        logging=logging_config,
    )


def validated_copy(model: Any, **updates: Any) -> Any:
    values = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    values.update(updates)
    return type(model)(**values)


def make_experiment_id(samples: int) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"smoke-{samples}-{timestamp}-{uuid4().hex[:8]}"


def validate_completed_run(config: AppConfig, completed: Mapping[str, int]) -> Path:
    """Check that the manifest, model runs, pairing, and core artifacts completed."""
    expected_count = config.data.max_samples
    if expected_count is None:
        raise ValueError("A smoke test must have a dataset sample limit")
    experiment_id = config.logging.experiment_id
    if experiment_id is None:
        raise ValueError("A smoke test must have an experiment ID")

    experiment_root = Path(config.logging.root) / experiment_id
    dataset_path = experiment_root / "dataset.json"
    if not dataset_path.is_file():
        raise RuntimeError(f"Missing pipeline artifact: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as source:
        dataset_metadata = json.load(source)
    actual_inputs = int(dataset_metadata.get("examples", -1))
    if actual_inputs != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} manifest examples, but found {actual_inputs}"
        )

    expected_counts = {model: expected_count for model in config.models_to_run}
    if set(MODEL_CHOICES).issubset(config.models_to_run):
        expected_counts["paired"] = expected_count
    mismatches = {
        name: {"expected": expected, "actual": completed.get(name)}
        for name, expected in expected_counts.items()
        if completed.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Pipeline output count mismatch: {mismatches}")

    required_paths = [
        experiment_root / "manifest.json",
        experiment_root / "dataset.json",
        experiment_root / "inputs.jsonl",
    ]
    for model in config.models_to_run:
        required_paths.extend(
            [
                experiment_root / model / "outputs.jsonl",
            ]
        )
    if "gemma" in config.models_to_run:
        required_paths.append(experiment_root / "gemma" / "tokens.jsonl")
    if "diffusion_gemma" in config.models_to_run:
        required_paths.append(experiment_root / "diffusion_gemma" / "canvas.jsonl")
    if set(MODEL_CHOICES).issubset(config.models_to_run):
        required_paths.append(experiment_root / "comparison" / "pairs.jsonl")

    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(f"Missing pipeline artifacts: {missing_paths}")
    return experiment_root


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    config = build_smoke_config(
        samples=args.samples,
        models=args.models,
        experiment_id=args.experiment_id,
        output_root=args.output_root,
        environ=os.environ,
    )

    print(
        "Starting pipeline smoke test: "
        f"samples={config.data.max_samples}, "
        f"models={','.join(config.models_to_run)}, "
        f"experiment_id={config.logging.experiment_id}"
    )
    # Import lazily so `--help` and configuration tests do not load model stacks.
    from main import main as run_pipeline

    completed = run_pipeline(config)
    experiment_root = validate_completed_run(config, completed)
    print(f"Smoke test passed. Artifacts: {experiment_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
