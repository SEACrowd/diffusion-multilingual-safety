from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import datasets
import diffusers
import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from config_parser import AppConfig, parse_app_config
from dataloader import create_manifest_dataloader, materialize_input_manifest
from evaluation.paired_outputs import create_paired_outputs
from logging_utils.writer import write_json
from models.diffusion_gemma import (
    create_diffusion_gemma_pipeline,
    run_diffusion_gemma_inference,
)
from models.gemma import create_gemma_model, run_gemma_inference


EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured model pipeline.")
    parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run in the background and write stdout/stderr to logging/_daemon/.",
    )
    return parser.parse_args(argv)


def launch_daemon(config: AppConfig) -> int:
    """Start a detached foreground-mode child and return its process ID."""
    experiment_id = resolve_experiment_id(config.logging.experiment_id)
    logging_root = Path(config.logging.root)
    daemon_root = logging_root / "_daemon"
    daemon_root.mkdir(parents=True, exist_ok=True)
    launch_id = uuid4().hex[:8]
    log_path = daemon_root / f"{experiment_id}-{launch_id}.log"
    pid_path = daemon_root / f"{experiment_id}-{launch_id}.pid"

    child_environment = os.environ.copy()
    child_environment["LOGGING_EXPERIMENT_ID"] = experiment_id
    child_environment["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    common_options = {
        "cwd": str(Path.cwd()),
        "env": child_environment,
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }

    with log_path.open("a", encoding="utf-8") as log_output:
        common_options["stdout"] = log_output
        if os.name == "nt":
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
            process = subprocess.Popen(
                command,
                creationflags=creation_flags,
                **common_options,
            )
        else:
            process = subprocess.Popen(
                command,
                start_new_session=True,
                **common_options,
            )

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    result_path = logging_root / experiment_id
    print(f"Pipeline started in the background (PID {process.pid}).")
    print(f"Experiment ID: {experiment_id}")
    print(f"Daemon log: {log_path.resolve()}")
    print(f"PID file: {pid_path.resolve()}")
    print(f"Results: {result_path.resolve()}")
    return process.pid


def cli(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = parse_cli_args(argv)
    config = parse_app_config()
    if args.daemon:
        launch_daemon(config)
        return 0
    main(config)
    return 0


def main(config: AppConfig) -> dict[str, int]:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    experiment_id = resolve_experiment_id(config.logging.experiment_id)
    experiment_root = Path(config.logging.root) / experiment_id
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {experiment_root}")
    experiment_root.mkdir(parents=True, exist_ok=True)

    inputs_path = experiment_root / "inputs.jsonl"
    dataset_metadata = materialize_input_manifest(
        config.data,
        manifest_path=inputs_path,
        metadata_path=experiment_root / "dataset.json",
        token=hf_token,
    )
    model_revisions = resolve_model_revisions(config, hf_token)
    write_experiment_manifest(
        experiment_root / "manifest.json",
        experiment_id=experiment_id,
        config=config,
        dataset_metadata=dataset_metadata,
        model_revisions=model_revisions,
    )

    completed: dict[str, int] = {}
    for model_kind in config.models_to_run:
        dataloader = create_manifest_dataloader(
            inputs_path,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory,
        )
        if model_kind == "gemma":
            completed[model_kind] = run_gemma(
                config,
                dataloader,
                experiment_id,
                experiment_root,
                model_revisions[model_kind],
                hf_token,
            )
        elif model_kind == "diffusion_gemma":
            completed[model_kind] = run_diffusion_gemma(
                config,
                dataloader,
                experiment_id,
                experiment_root,
                model_revisions[model_kind],
                hf_token,
            )
        release_cuda_memory()

    if {"gemma", "diffusion_gemma"}.issubset(completed):
        comparison_root = experiment_root / "comparison"
        completed["paired"] = create_paired_outputs(
            gemma_outputs_path=experiment_root / "gemma" / "outputs.jsonl",
            diffusion_outputs_path=experiment_root / "diffusion_gemma" / "outputs.jsonl",
            pairs_path=comparison_root / "pairs.jsonl",
        )

    print(f"Experiment {experiment_id}: {completed}")
    return completed


def run_gemma(
    config: AppConfig,
    dataloader,
    experiment_id: str,
    experiment_root: Path,
    revision: str,
    token: str | None,
) -> int:
    model, processor = create_gemma_model(
        model_name=config.gemma_model.model_name,
        processor_name=config.gemma_model.processor_name,
        revision=revision,
        dtype=config.gemma_model.dtype,
        device_map=config.gemma_model.device_map,
        token=token,
    )
    try:
        return run_gemma_inference(
            model,
            processor,
            dataloader,
            experiment_id=experiment_id,
            model_id=config.gemma_model.model_name,
            model_revision=revision,
            logging_root=experiment_root / "gemma",
            max_batches=config.inference_max_batches,
            seed=config.logging.seed,
            max_new_tokens=config.gemma_generation.max_new_tokens,
            enable_thinking=config.gemma_generation.enable_thinking,
            do_sample=config.gemma_generation.do_sample,
            temperature=config.gemma_generation.temperature,
            top_p=config.gemma_generation.top_p,
            top_k=config.gemma_generation.top_k,
            log_top_k=config.logging.top_k,
            log_logits=config.logging.log_logits,
            log_moe=config.logging.log_moe,
            save_full_logits=config.logging.save_full_logits,
        )
    finally:
        del model
        del processor


def run_diffusion_gemma(
    config: AppConfig,
    dataloader,
    experiment_id: str,
    experiment_root: Path,
    revision: str,
    token: str | None,
) -> int:
    pipeline = create_diffusion_gemma_pipeline(
        model_name=config.diffusion_gemma_model.model_name,
        processor_name=config.diffusion_gemma_model.processor_name,
        revision=revision,
        dtype=config.diffusion_gemma_model.dtype,
        device_map=config.diffusion_gemma_model.device_map,
        entropy_bound=config.diffusion_gemma_generation.entropy_bound,
        t_max=config.diffusion_gemma_generation.t_max,
        t_min=config.diffusion_gemma_generation.t_min,
        token=token,
    )
    try:
        return run_diffusion_gemma_inference(
            pipeline,
            dataloader,
            experiment_id=experiment_id,
            model_id=config.diffusion_gemma_model.model_name,
            model_revision=revision,
            logging_root=experiment_root / "diffusion_gemma",
            max_batches=config.inference_max_batches,
            seed=config.logging.seed,
            gen_length=config.diffusion_gemma_generation.gen_length,
            num_inference_steps=(
                config.diffusion_gemma_generation.max_denoising_steps
            ),
            stability_threshold=(
                config.diffusion_gemma_generation.stability_threshold
            ),
            confidence_threshold=(
                config.diffusion_gemma_generation.confidence_threshold
            ),
            log_top_k=config.logging.top_k,
            log_logits=config.logging.log_logits,
            log_moe=config.logging.log_moe,
            save_full_logits=config.logging.save_full_logits,
        )
    finally:
        del pipeline


def resolve_model_revisions(config: AppConfig, token: str | None) -> dict[str, str]:
    api = HfApi(token=token)
    revisions: dict[str, str] = {}
    if "gemma" in config.models_to_run:
        revisions["gemma"] = api.model_info(
            config.gemma_model.model_name,
            revision=config.gemma_model.revision,
        ).sha
    if "diffusion_gemma" in config.models_to_run:
        revisions["diffusion_gemma"] = api.model_info(
            config.diffusion_gemma_model.model_name,
            revision=config.diffusion_gemma_model.revision,
        ).sha
    return revisions


def resolve_experiment_id(configured: str | None) -> str:
    experiment_id = configured or (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "LOGGING_EXPERIMENT_ID may contain only letters, numbers, dot, underscore, and hyphen"
        )
    return experiment_id


def write_experiment_manifest(
    path: Path,
    *,
    experiment_id: str,
    config: AppConfig,
    dataset_metadata: dict,
    model_revisions: dict[str, str],
) -> None:
    config_data = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    write_json(
        path,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset_metadata,
            "resolved_model_revisions": model_revisions,
            "config": config_data,
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "diffusers": diffusers.__version__,
                "datasets": datasets.__version__,
                "cuda_runtime": torch.version.cuda,
            },
        },
    )


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(cli())
