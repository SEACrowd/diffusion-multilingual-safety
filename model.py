"""Model loading and no-grad inference utilities."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion


def create_model_and_processor(
    model_name: str,
    processor_name: str | None = None,
    dtype: str = "auto",
    device_map: str | None = "auto",
) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(processor_name or model_name)
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return processor, model


def run_no_grad_inference(
    model: Any,
    dataloader: DataLoader,
    max_batches: int | None = 1,
) -> list[dict[str, Any]]:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive or None")

    model.eval()
    results: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            model_inputs = batch_to_model_inputs(batch, model)
            outputs = model(**model_inputs)
            results.append(summarize_outputs(batch_index, outputs, batch))

    return results


def batch_to_model_inputs(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    accepted_args, accepts_kwargs = forward_input_names(model)
    device = input_device(model)
    model_inputs: dict[str, Any] = {}

    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        if not accepts_kwargs and key not in accepted_args:
            continue
        model_inputs[key] = value.to(device)

    if "input_ids" not in model_inputs:
        raise ValueError("Batch does not contain tensor input_ids for model inference")

    return model_inputs


def forward_input_names(model: Any) -> tuple[set[str], bool]:
    signature = inspect.signature(model.forward)
    accepted_args = set(signature.parameters)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return accepted_args, accepts_kwargs


def input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def summarize_outputs(
    batch_index: int,
    outputs: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    loss = getattr(outputs, "loss", None)
    logits = getattr(outputs, "logits", None)

    summary: dict[str, Any] = {"batch_index": batch_index}
    if loss is not None:
        summary["loss"] = float(loss.detach().cpu())
    if logits is not None:
        summary["logits_shape"] = tuple(logits.shape)
    if "metadata" in batch:
        summary["metadata"] = batch["metadata"]

    return summary
