"""Stateless DiffusionGemma pipeline construction and inference."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
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
from logging_utils.safety import best_effort, close_optional, create_optional

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


@contextmanager
def force_chat_template_thinking(processor: Any, enable_thinking: bool) -> Iterator[None]:
    """Diffusers' pipeline encode path omits enable_thinking; inject it here."""
    original = processor.apply_chat_template

    def apply_chat_template(*args: Any, **kwargs: Any) -> Any:
        kwargs["enable_thinking"] = enable_thinking
        return original(*args, **kwargs)

    processor.apply_chat_template = apply_chat_template
    try:
        yield
    finally:
        processor.apply_chat_template = original


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
    enable_thinking: bool,
    log_top_k: int,
    log_logits: bool,
    log_moe: bool,
    save_full_logits: bool,
) -> int:
    root = Path(logging_root)
    outputs = OutputLogger(root / "outputs.jsonl")
    canvas = create_optional(
        "DiffusionGemma canvas",
        lambda: CanvasLogger(root / "canvas.jsonl", pipeline.processor),
    )
    logits = (
        create_optional(
            "DiffusionGemma logits",
            lambda: DiffusionLogitsLogger(
                root / "logits.jsonl",
                top_k=log_top_k,
                save_full_logits=save_full_logits,
                full_logits_directory=root / "full_logits",
            ),
        )
        if log_logits
        else None
    )
    router = (
        create_optional(
            "DiffusionGemma MoE",
            lambda: DiffusionGemmaRouterTracer(
                root / "moe.jsonl",
                selected_experts=selected_expert_count(pipeline.model),
            ),
        )
        if log_moe
        else None
    )
    if router is not None and not best_effort(
        "DiffusionGemma MoE attachment",
        lambda: router.attach(pipeline.model),
    ):
        close_optional("DiffusionGemma MoE", router)
        router = None

    sampler_configuration = scheduler_configuration(pipeline.scheduler)
    generation_configuration = {
        "gen_length": gen_length,
        "num_inference_steps": num_inference_steps,
        "stability_threshold": stability_threshold,
        "confidence_threshold": confidence_threshold,
        "enable_thinking": enable_thinking,
        "scheduler": sampler_configuration,
    }
    count = 0
    disabled_components: set[str] = set()
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
                enable_thinking=enable_thinking,
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
                enable_thinking=enable_thinking,
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
                router_tracer=router,
                disabled_components=disabled_components,
            )

            if router is not None:
                router.begin_example(context)
            try:
                with force_chat_template_thinking(pipeline.processor, enable_thinking):
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
                if router is not None:
                    router.end_example()
            except BaseException:
                if router is not None:
                    router.abort_example()
                raise

            generated = output.sequences[0].detach().cpu()
            # ponytail: strip prompt only if pipeline returned prompt+canvas; gen region
            # is exactly gen_length, so a longer sequence means the prompt is prepended.
            if generated.shape[0] > gen_length:
                generated = generated[input_token_count:]
            generated_token_ids = generated.tolist()
            response_errors: list[str] = []
            try:
                raw_text = pipeline.processor.decode(
                    generated,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception as error:
                raw_text = None
                response_errors.append(f"raw decode: {type(error).__name__}: {error}")

            texts = getattr(output, "texts", None)
            if texts and isinstance(texts[0], str):
                final_text = texts[0]
            else:
                try:
                    final_text = pipeline.processor.decode(
                        generated,
                        skip_special_tokens=True,
                    )
                except Exception as error:
                    final_text = ""
                    response_errors.append(
                        f"clean decode: {type(error).__name__}: {error}"
                    )
            response_error = "; ".join(response_errors) or None
            if not final_text:
                warnings.warn(
                    "DiffusionGemma response decoding failed or returned empty text; "
                    "generated token IDs were preserved",
                    RuntimeWarning,
                    stacklevel=2,
                )
            outputs.log(
                context=output_context,
                final_text=final_text,
                final_token_ids=generated_token_ids,
                input_token_count=input_token_count,
                rendered_prompt=rendered_prompt,
                raw_response=raw_text,
                response_error=response_error,
            )
            best_effort(
                "DiffusionGemma response display",
                lambda: print(
                    f"[diffusion_gemma{'/thinking' if enable_thinking else '/non_thinking'}] "
                    f"response for {metadata['id']}:\n"
                    f"{final_text or '[decoding failed; token IDs saved]'}\n",
                    flush=True,
                ),
            )
            count += 1
    finally:
        outputs.close()
        close_optional("DiffusionGemma canvas", canvas)
        close_optional("DiffusionGemma logits", logits)
        close_optional("DiffusionGemma MoE", router)

    return count


def scheduler_configuration(scheduler: Any) -> dict[str, Any]:
    configuration = getattr(scheduler, "config", {})
    return {
        key: value
        for key, value in dict(configuration).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
