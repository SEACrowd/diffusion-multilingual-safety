"""Stateless Gemma 4 construction and autoregressive inference."""

from __future__ import annotations

import time
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
    parse_response_text,
    resolve_dtype,
    seed_torch,
    selected_expert_count,
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
    performance = PerformanceLogger(root / "performance.jsonl")
    tokens = GemmaTokenLogger(root / "tokens.jsonl", processor)
    logits = (
        GemmaLogitsLogger(
            root / "logits.jsonl",
            top_k=log_top_k,
            save_full_logits=save_full_logits,
            full_logits_directory=root / "full_logits",
        )
        if log_logits
        else None
    )
    router = (
        GemmaRouterTracer(
            root / "moe.jsonl",
            selected_experts=selected_expert_count(model),
        )
        if log_moe
        else None
    )
    if router is not None:
        router.attach(model)

    generation_configuration = {
        "max_new_tokens": max_new_tokens,
        "enable_thinking": enable_thinking,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
    }
    count = 0
    try:
        for prompt, metadata, reference in iter_examples(dataloader, max_batches):
            run_id = uuid4().hex
            current_seed = example_seed(seed, str(metadata["id"]))
            seed_torch(current_seed)
            context = example_context(
                experiment_id=experiment_id,
                model_kind="gemma",
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
            rendered_prompt = processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            model_inputs = processor(text=rendered_prompt, return_tensors="pt").to(model.device)
            input_token_count = int(model_inputs["input_ids"].shape[-1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_scores": log_logits,
            }
            if do_sample:
                generation_kwargs.update(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

            if router is not None:
                router.begin_example(context)
            reset_cuda_peak_memory()
            synchronize_cuda()
            started_at = time.perf_counter()
            try:
                generation_output = model.generate(**model_inputs, **generation_kwargs)
                synchronize_cuda()
            finally:
                if router is not None:
                    router.end_example()
            total_latency = time.perf_counter() - started_at

            generated = generation_output.sequences[0, input_token_count:]
            generated_token_ids = generated.detach().cpu().tolist()
            raw_text = processor.decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            final_text = parse_response_text(processor, raw_text)
            outputs.log(
                context=output_context,
                final_text=final_text,
                final_token_ids=generated_token_ids,
                input_token_count=input_token_count,
                rendered_prompt=rendered_prompt,
            )
            tokens.log_generation(context, generated_token_ids)
            if logits is not None:
                logits.log_generation(
                    context=context,
                    scores=tuple(generation_output.scores),
                    generated_token_ids=generated_token_ids,
                )
            performance.log_inference(
                context=context,
                total_latency=total_latency,
                input_token_count=input_token_count,
                output_token_count=len(generated_token_ids),
                instrumented=log_logits or log_moe,
                cuda_memory=cuda_peak_memory(),
            )
            count += 1
    finally:
        outputs.close()
        performance.close()
        tokens.close()
        if logits is not None:
            logits.close()
        if router is not None:
            router.close()

    write_source_summary_from_jsonl(
        root / "performance.jsonl",
        root / "source_summary.json",
    )
    return count
