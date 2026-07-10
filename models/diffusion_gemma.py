"""Stateless DiffusionGemma pipeline construction and inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from diffusers import DiffusionGemmaPipeline
from torch.utils.data import DataLoader
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from logging_utils.diffusion_gemma.callback import DiffusionLoggingCallback
from logging_utils.diffusion_gemma.canvas import CanvasLogger
from logging_utils.diffusion_gemma.logits import DiffusionLogitsLogger
from logging_utils.diffusion_gemma.router import DiffusionGemmaRouterTracer
from logging_utils.diffusion_gemma.scheduler import TracingEntropyBoundScheduler
from logging_utils.outcomes import OutputLogger
from logging_utils.performance import (
    PerformanceLogger,
    cuda_peak_memory,
    reset_cuda_peak_memory,
    synchronize_cuda,
    write_source_summary_from_jsonl,
)

from .common import (
    example_context,
    example_seed,
    iter_examples,
    resolve_dtype,
    selected_expert_count,
)


def create_diffusion_gemma_pipeline(
    *,
    model_name: str,
    processor_name: str | None = None,
    revision: str,
    dtype: str | torch.dtype = "auto",
    device_map: str | None = "auto",
    entropy_bound: float = 0.1,
    t_max: float = 0.8,
    t_min: float = 0.4,
    token: str | None = None,
) -> DiffusionGemmaPipeline:
    processor = AutoProcessor.from_pretrained(
        processor_name or model_name,
        revision=revision,
        token=token,
    )
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": resolve_dtype(dtype),
        "token": token,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_name, **model_kwargs)
    model.eval()
    scheduler = TracingEntropyBoundScheduler(
        entropy_bound=entropy_bound,
        t_max=t_max,
        t_min=t_min,
    )
    return DiffusionGemmaPipeline(model=model, scheduler=scheduler, processor=processor)


@torch.inference_mode()
def run_diffusion_gemma_inference(
    pipeline: DiffusionGemmaPipeline,
    dataloader: DataLoader,
    *,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    logging_root: str | Path,
    max_batches: int | None,
    seed: int,
    gen_length: int,
    num_inference_steps: int,
    stability_threshold: int,
    confidence_threshold: float | None,
    log_top_k: int,
    log_logits: bool,
    log_moe: bool,
    save_full_logits: bool,
) -> int:
    root = Path(logging_root)
    outputs = OutputLogger(root / "outputs.jsonl")
    performance = PerformanceLogger(root / "performance.jsonl")
    canvas = CanvasLogger(root / "canvas.jsonl", pipeline.processor)
    logits = (
        DiffusionLogitsLogger(
            root / "logits.jsonl",
            top_k=log_top_k,
            save_full_logits=save_full_logits,
            full_logits_directory=root / "full_logits",
        )
        if log_logits
        else None
    )
    router = (
        DiffusionGemmaRouterTracer(
            root / "moe.jsonl",
            selected_experts=selected_expert_count(pipeline.model),
        )
        if log_moe
        else None
    )
    if router is not None:
        router.attach(pipeline.model)

    sampler_configuration = scheduler_configuration(pipeline.scheduler)
    generation_configuration = {
        "gen_length": gen_length,
        "num_inference_steps": num_inference_steps,
        "stability_threshold": stability_threshold,
        "confidence_threshold": confidence_threshold,
        "scheduler": sampler_configuration,
    }
    count = 0
    try:
        for prompt, metadata, reference in iter_examples(dataloader, max_batches):
            run_id = uuid4().hex
            current_seed = example_seed(seed, str(metadata["id"]))
            context = example_context(
                experiment_id=experiment_id,
                model_kind="diffusion_gemma",
                model_id=model_id,
                model_revision=model_revision,
                run_id=run_id,
                metadata=metadata,
                seed=current_seed,
            )
            output_context = {
                **context,
                "prompt": prompt,
                "reference_response": reference,
                "generation_configuration": generation_configuration,
            }
            rendered_prompt = pipeline.processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded_prompt = pipeline.processor(text=rendered_prompt, return_tensors="pt")
            input_token_count = int(encoded_prompt["input_ids"].shape[-1])
            generator = torch.Generator(device=pipeline._execution_device).manual_seed(
                current_seed
            )
            callback = DiffusionLoggingCallback(
                context=context,
                canvas_logger=canvas,
                logits_logger=logits,
                performance_logger=performance,
                router_tracer=router,
            )

            if router is not None:
                router.begin_example(context)
            reset_cuda_peak_memory()
            synchronize_cuda()
            started_at = time.perf_counter()
            try:
                output = pipeline(
                    prompt=prompt,
                    gen_length=gen_length,
                    num_inference_steps=num_inference_steps,
                    stability_threshold=stability_threshold,
                    confidence_threshold=confidence_threshold,
                    generator=generator,
                    callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=["canvas"],
                )
                synchronize_cuda()
                if router is not None:
                    router.end_example()
            except BaseException:
                if router is not None:
                    router.abort_example()
                raise
            total_latency = time.perf_counter() - started_at

            generated_token_ids = output.sequences[0].detach().cpu().tolist()
            outputs.log(
                context=output_context,
                final_text=output.texts[0],
                final_token_ids=generated_token_ids,
                input_token_count=input_token_count,
                rendered_prompt=rendered_prompt,
            )
            performance.log_inference(
                context=context,
                total_latency=total_latency,
                input_token_count=input_token_count,
                output_token_count=len(generated_token_ids),
                instrumented=True,
                cuda_memory=cuda_peak_memory(),
            )
            count += 1
    finally:
        outputs.close()
        performance.close()
        canvas.close()
        if logits is not None:
            logits.close()
        if router is not None:
            router.close()

    write_source_summary_from_jsonl(
        root / "performance.jsonl",
        root / "source_summary.json",
    )
    return count


def scheduler_configuration(scheduler: Any) -> dict[str, Any]:
    configuration = getattr(scheduler, "config", {})
    return {
        key: value
        for key, value in dict(configuration).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
