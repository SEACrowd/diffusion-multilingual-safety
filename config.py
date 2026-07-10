"""Default configuration values.

This module intentionally has no environment reads. Environment parsing lives in
`config_parser.py`, and secrets such as HF_TOKEN are read at the entrypoint.
"""

HF_MODEL_NAME = "google/diffusiongemma-26B-A4B-it"
HF_PROCESSOR_NAME = HF_MODEL_NAME
MODEL_DTYPE = "auto"
MODEL_DEVICE_MAP = "auto"
INFERENCE_MAX_BATCHES = 1

DATASET_NAME = "tackhwa/multilingual_safety"
DATASET_SPLIT = "train"
DATASET_STREAMING = True
DATASET_MAX_SAMPLES = None
DATASET_LANGUAGES = []
DATASET_PROMPT_LABELS = []
DATASET_RESPONSE_LABELS = []
DATASET_TAGS = []
DATASET_SOURCES = []
DATASET_SEED = 42
DATASET_FILTER_NUM_PROC = None
DATASET_MAP_NUM_PROC = None
DATASET_NORMALIZE = True
HF_DATASETS_CACHE = None

BATCH_SIZE = 2
SHUFFLE_DATASET = True
NUM_WORKERS = 0
PIN_MEMORY = False
MAX_LENGTH = 2048
MASK_PROMPT_LABELS = True
RETURN_METADATA = True
SHUFFLE_BUFFER_SIZE = 10_000

