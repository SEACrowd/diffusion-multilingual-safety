"""Stateless Gemma 4 construction and inference."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

from logging_utils.gemma.logits import GemmaLogitsLogger
from logging_utils.gemma.router import GemmaRouterTracer
from logging_utils.gemma.tokens import GemmaTokenLogger
from logging_utils.outcomes import OutputLogger
from logging_utils.safety import best_effort, close_optional, create_optional

from .common import (
    batch_seed,
    ensure_left_padding,
    example_context,
    example_seed,
    iter_batches,
    model_primary_device,
    parse_response_text,
    resolve_dtype,
    resolve_response_boundary_token_ids,
    sanitize_generated_token_ids,
    seed_torch,
    selected_expert_count,
    strip_response_marker_text,
)


def create_gemma_model(
    *,
    model_name: str,
    processor_name: str | None = None,
    revision: str,
    dtype: str | torch.dtype = "auto",
    device_map: str | None = "auto",
    token: str | None = None,
) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        processor_name or model_name,
        revision=revision,
        token=token,
    )
    ensure_left_padding(processor)
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": resolve_dtype(dtype),
        "token": token,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, processor


@torch.inference_mode()
def run_gemma_inference(
    model: Any,
    processor: Any,
    dataloader: DataLoader,
    *,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    logging_root: str | Path,
    max_batches: int | None,
    seed: int,
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    log_top_k: int,
    log_logits: bool,
    log_moe: bool,
    save_full_logits: bool,
) -> int:
    root = Path(logging_root)
    outputs = OutputLogger(root / "outputs.jsonl")
    tokens = create_optional(
        "Gemma token",
        lambda: GemmaTokenLogger(root / "tokens.jsonl", processor),
    )
    logits = (
        create_optional(
            "Gemma logits",
            lambda: GemmaLogitsLogger(
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
            "Gemma MoE",
            lambda: GemmaRouterTracer(
                root / "moe.jsonl",
                selected_experts=selected_expert_count(model),
            ),
        )
        if log_moe
        else None
    )
    if router is not None and not best_effort(
        "Gemma MoE attachment",
        lambda: router.attach(model),
    ):
        close_optional("Gemma MoE", router)
        router = None

    generation_configuration = {
        "max_new_tokens": max_new_tokens,
        "enable_thinking": enable_thinking,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
    }
    device = model_primary_device(model)
    eos_token_id, pad_token_id, turn_end_token_id = resolve_response_boundary_token_ids(
        processor
    )
    ensure_left_padding(processor)
    count = 0
    moe_batch_warned = False
    try:
        for prompts, metadata_rows, references in iter_batches(dataloader, max_batches):
            batch_size = len(prompts)
            example_ids = [str(metadata["id"]) for metadata in metadata_rows]
            current_batch_seed = batch_seed(seed, example_ids)
            seed_torch(current_batch_seed)

            contexts: list[dict[str, Any]] = []
            output_contexts: list[dict[str, Any]] = []
            for prompt, metadata, reference in zip(
                prompts, metadata_rows, references, strict=True
            ):
                run_id = uuid4().hex
                current_seed = example_seed(seed, str(metadata["id"]))
                context = example_context(
                    experiment_id=experiment_id,
                    model_kind="gemma",
                    model_id=model_id,
                    model_revision=model_revision,
                    run_id=run_id,
                    metadata=metadata,
                    seed=current_seed,
                    enable_thinking=enable_thinking,
                )
                contexts.append(context)
                output_contexts.append(
                    {
                        **context,
                        "prompt": prompt,
                        "reference_response": reference,
                        "generation_configuration": generation_configuration,
                        "batch_size": batch_size,
                        "batch_seed": current_batch_seed,
                    }
                )

            rendered_prompts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for prompt in prompts
            ]
            model_inputs = processor(
                text=rendered_prompts,
                return_tensors="pt",
                padding=True,
            ).to(device)
            input_token_counts = model_inputs["attention_mask"].sum(dim=1).tolist()
            padded_prompt_length = int(model_inputs["input_ids"].shape[-1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_logits": logits is not None,
            }
            if do_sample:
                generation_kwargs.update(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

            use_router = router is not None and batch_size == 1
            if router is not None and batch_size > 1 and not moe_batch_warned:
                warnings.warn(
                    "Gemma MoE telemetry is skipped for batch_size > 1",
                    RuntimeWarning,
                    stacklevel=2,
                )
                moe_batch_warned = True
            if use_router:
                router.begin_example(contexts[0])
            try:
                generation_output = model.generate(**model_inputs, **generation_kwargs)
            finally:
                if use_router:
                    router.end_example()

            generated_batch = generation_output.sequences[:, padded_prompt_length:]
            score_steps = (
                tuple(generation_output.logits)
                if logits is not None and getattr(generation_output, "logits", None)
                else None
            )
            for index, (context, output_context, rendered_prompt) in enumerate(
                zip(contexts, output_contexts, rendered_prompts, strict=True)
            ):
                generated_token_ids = sanitize_generated_token_ids(
                    generated_batch[index].detach().cpu().tolist(),
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    turn_end_token_id=turn_end_token_id,
                )
                response_error = None
                try:
                    raw_text = strip_response_marker_text(
                        processor.decode(
                            generated_token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                    final_text = strip_response_marker_text(
                        parse_response_text(processor, raw_text)
                    )
                except Exception as error:
                    raw_text = None
                    final_text = ""
                    response_error = f"{type(error).__name__}: {error}"
                    warnings.warn(
                        "Gemma response decoding failed; generated token IDs were preserved",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                outputs.log(
                    context=output_context,
                    final_text=final_text,
                    final_token_ids=generated_token_ids,
                    input_token_count=int(input_token_counts[index]),
                    rendered_prompt=rendered_prompt,
                    raw_response=raw_text,
                    response_error=response_error,
                )
                best_effort(
                    "Gemma response display",
                    lambda context=context, final_text=final_text: print(
                        f"[gemma{'/thinking' if enable_thinking else '/non_thinking'}] "
                        f"response for {context['example_id']}:\n"
                        f"{final_text or '[decoding failed; token IDs saved]'}\n",
                        flush=True,
                    ),
                )
                if tokens is not None and not best_effort(
                    "Gemma token",
                    lambda context=context, generated_token_ids=generated_token_ids: (
                        tokens.log_generation(context, generated_token_ids)
                    ),
                ):
                    close_optional("Gemma token", tokens)
                    tokens = None
                if logits is not None and score_steps is not None:
                    example_scores = tuple(
                        step_scores[index : index + 1] for step_scores in score_steps
                    )
                    # Batched generate keeps logits for the full padded generation
                    # length; trim to this example's decoded token count.
                    example_scores = example_scores[: len(generated_token_ids)]
                    if not best_effort(
                        "Gemma logits",
                        lambda context=context,
                        example_scores=example_scores,
                        generated_token_ids=generated_token_ids: logits.log_generation(
                            context=context,
                            scores=example_scores,
                            generated_token_ids=generated_token_ids,
                        ),
                    ):
                        close_optional("Gemma logits", logits)
                        logits = None
                count += 1
    finally:
        outputs.close()
        close_optional("Gemma token", tokens)
        close_optional("Gemma logits", logits)
        close_optional("Gemma MoE", router)

    return count
