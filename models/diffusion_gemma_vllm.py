"""DiffusionGemma construction and vLLM offline batched inference.

Response-only path. The diffusers-side per-step tracing (canvas / logits / MoE
router) has no offline vLLM equivalent, so it is dropped here. The diffusers
scheduler knobs num_inference_steps / stability_threshold / confidence_threshold
/ t_min / t_max likewise have no documented offline vLLM control and become fixed
sampler behavior; only canvas_length (== gen_length) and entropy_bound are
exposed. vLLM is imported lazily so this module imports without vLLM installed.

ponytail: the hf_overrides / diffusion_config keys below come from the vLLM
DiffusionGemma recipe (vLLM >= 0.24.0). Verify the exact key names against the
installed vLLM release before a real run; that is the calibration knob.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor

from logging_utils.outcomes import OutputLogger

from .common import example_context, example_seed, iter_examples


def create_diffusion_gemma_vllm_engine(
    *,
    model_name: str,
    processor_name: str | None = None,
    revision: str,
    gen_length: int,
    entropy_bound: float,
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 4,
    gpu_memory_utilization: float = 0.85,
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
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "hf_overrides": {
            "diffusion_sampler": "entropy_bound",
            "diffusion_entropy_bound": entropy_bound,
        },
        "diffusion_config": {"canvas_length": gen_length},
    }
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len
    if token is not None:
        engine_kwargs["hf_token"] = token
    engine = LLM(**engine_kwargs)
    return engine, processor


def run_diffusion_gemma_vllm_inference(
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
    gen_length: int,
    entropy_bound: float,
) -> int:
    from vllm import SamplingParams  # noqa: PLC0415 - lazy, see create fn

    root = Path(logging_root)
    outputs = OutputLogger(root / "outputs.jsonl")

    generation_configuration = {
        "gen_length": gen_length,
        "entropy_bound": entropy_bound,
        "engine": "vllm",
    }

    prompts: list[str] = []
    params: list[Any] = []
    contexts: list[dict[str, Any]] = []
    for prompt, metadata, reference in iter_examples(dataloader, max_batches):
        current_seed = example_seed(seed, str(metadata["id"]))
        context = example_context(
            experiment_id=experiment_id,
            model_kind="diffusion_gemma",
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
        )
        sampling = SamplingParams(
            temperature=0.0,  # deterministic denoising, matches the entropy-bound sampler
            max_tokens=gen_length,
            seed=current_seed,
            skip_special_tokens=True,
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
        results = engine.generate(prompts, params) if prompts else []
        for result, context in zip(results, contexts, strict=True):
            completion = result.outputs[0]
            final_text = completion.text
            generated_token_ids = list(completion.token_ids)
            input_token_count = len(result.prompt_token_ids)
            rendered_prompt = context.pop("rendered_prompt")
            response_error = None
            try:
                raw_text = processor.decode(
                    generated_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception as error:
                raw_text = None
                response_error = f"raw decode: {type(error).__name__}: {error}"
            if not final_text:
                warnings.warn(
                    "DiffusionGemma returned empty text; token IDs were preserved",
                    RuntimeWarning,
                    stacklevel=2,
                )
            outputs.log(
                context=context,
                final_text=final_text,
                final_token_ids=generated_token_ids,
                input_token_count=input_token_count,
                rendered_prompt=rendered_prompt,
                raw_response=raw_text,
                response_error=response_error,
            )
            print(
                f"[diffusion_gemma] response for {context.get('example_id')}:\n"
                f"{final_text or '[empty response]'}\n",
                flush=True,
            )
            count += 1
    finally:
        outputs.close()
    return count
