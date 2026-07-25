"""Gemma 4 MoE construction and vLLM offline batched inference.

Response-only path: no per-token / logits / MoE tracing (those hooks require the
in-process HuggingFace model in models/gemma.py). vLLM is imported lazily inside
the functions so this module stays importable on machines without vLLM
(e.g. the Windows dev box); vLLM only installs/runs on Linux GPU hosts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor

from logging_utils.outcomes import OutputLogger

from .common import (
    example_context,
    example_seed,
    iter_examples,
    parse_response_text,
)


def create_gemma_vllm_engine(
    *,
    model_name: str,
    processor_name: str | None = None,
    revision: str,
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = False,
    max_model_len: int | None = None,
    token: str | None = None,
) -> tuple[Any, Any]:
    from vllm import LLM  # noqa: PLC0415 - lazy: vLLM is Linux/GPU only

    processor = AutoProcessor.from_pretrained(
        processor_name or model_name,
        revision=revision,
        token=token,
    )
    engine_kwargs: dict[str, Any] = {
        "model": model_name,
        "revision": revision,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len
    if token is not None:
        engine_kwargs["hf_token"] = token
    engine = LLM(**engine_kwargs)
    return engine, processor


def run_gemma_vllm_inference(
    engine: Any,
    processor: Any,
    dataloader: Any,
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
) -> int:
    from vllm import SamplingParams  # noqa: PLC0415 - lazy, see create fn

    root = Path(logging_root)
    outputs = OutputLogger(root / "outputs.jsonl")

    generation_configuration = {
        "max_new_tokens": max_new_tokens,
        "enable_thinking": enable_thinking,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "engine": "vllm",
    }

    prompts: list[str] = []
    params: list[Any] = []
    contexts: list[dict[str, Any]] = []
    for prompt, metadata, reference in iter_examples(dataloader, max_batches):
        current_seed = example_seed(seed, str(metadata["id"]))
        context = example_context(
            experiment_id=experiment_id,
            model_kind="gemma",
            model_id=model_id,
            model_revision=model_revision,
            run_id=uuid4().hex,
            metadata=metadata,
            seed=current_seed,
        )
        rendered_prompt = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        # do_sample=False -> greedy: temperature 0, top_p/top_k disabled in vLLM.
        sampling = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else -1,
            max_tokens=max_new_tokens,
            seed=current_seed,
            skip_special_tokens=False,  # keep specials so parse_response_text matches HF
        )
        prompts.append(rendered_prompt)
        params.append(sampling)
        contexts.append(
            {
                **context,
                "prompt": prompt,
                "reference_response": reference,
                "generation_configuration": generation_configuration,
                "rendered_prompt": rendered_prompt,
            }
        )

    count = 0
    try:
        # Single batched call; vLLM returns results in input order.
        results = engine.generate(prompts, params) if prompts else []
        for result, context in zip(results, contexts, strict=True):
            completion = result.outputs[0]
            raw_text = completion.text
            generated_token_ids = list(completion.token_ids)
            input_token_count = len(result.prompt_token_ids)
            rendered_prompt = context.pop("rendered_prompt")
            final_text = parse_response_text(processor, raw_text)
            outputs.log(
                context=context,
                final_text=final_text,
                final_token_ids=generated_token_ids,
                input_token_count=input_token_count,
                rendered_prompt=rendered_prompt,
                raw_response=raw_text,
                response_error=None,
            )
            print(
                f"[gemma] response for {context.get('example_id')}:\n"
                f"{final_text or '[empty response]'}\n",
                flush=True,
            )
            count += 1
    finally:
        outputs.close()
    return count
