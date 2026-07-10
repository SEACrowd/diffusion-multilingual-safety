from __future__ import annotations

import os

from dotenv import load_dotenv
from huggingface_hub import login

from config_parser import AppConfig, parse_app_config
from dataloader import create_multilingual_safety_dataloader
import model as model_module


def create_data(processor, config: AppConfig):
    return create_multilingual_safety_dataloader(
        processor=processor,
        config=config.data,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        max_length=config.dataloader.max_length,
        mask_prompt_labels=config.dataloader.mask_prompt_labels,
        return_metadata=config.dataloader.return_metadata,
        shuffle_buffer_size=config.dataloader.shuffle_buffer_size,
    )


def main(config: AppConfig):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    login(token=hf_token) if hf_token else login()

    processor, diffusion_model = model_module.create_model_and_processor(
        model_name=config.model.model_name,
        processor_name=config.model.processor_name,
        dtype=config.model.dtype,
        device_map=config.model.device_map,
    )
    train_loader = create_data(processor, config)
    inference_results = model_module.run_no_grad_inference(
        diffusion_model,
        train_loader,
        max_batches=config.model.inference_max_batches,
    )

    print(f"Ran no-grad inference over {len(inference_results)} batch(es)")
    return inference_results


if __name__ == "__main__":
    load_dotenv()
    main(parse_app_config())
