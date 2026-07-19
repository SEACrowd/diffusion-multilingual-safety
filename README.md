# Gemma 4 Multilingual Safety Comparison

Runs the same pinned multilingual-safety examples through Gemma 4 26B A4B and
DiffusionGemma 26B A4B. Each model writes separate JSONL output, performance,
logit, and Mixture-of-Experts traces. A final streaming join creates paired
records by dataset example ID.

## Installation

Create and activate a Python 3.11 environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip before installing anything else:

```powershell
python -m pip install --upgrade pip
```

Install the newest stable PyTorch release currently published for the required
CUDA 13.0 wheel index:

```powershell
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

PyTorch 2.13 uses CUDA 13.2 wheels. The command above intentionally remains on
the newest stable `cu130` build.

Install the remaining dependencies. The requirements pin the stable releases
needed for DiffusionGemma: Transformers 5.13.0 and Diffusers 0.39.0.

```powershell
python -m pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Authentication

Accept the Gemma model licenses on Hugging Face. Put only the secret token in
`.env`:

```text
HF_TOKEN=hf_your_token
```

All non-secret settings live in `config.py`. Environment-variable overrides are
typed and validated by `config_parser.py`.

## Running

The safe default processes one batch with both models:

```powershell
python main.py
```

All experiment settings come from `config.py` (or typed environment-variable
overrides), so no model or dataset flags are required. To detach the same run in
the background, use:

```powershell
python main.py -d
```

In Google Colab, prefix it with `!`:

```python
!python main.py -d
```

The command prints the experiment ID, result directory, PID file, and daemon log
path before returning. The detached process continues while the Colab runtime is
alive. Follow the printed log path with `!tail -f <log-path>`.

Set `INFERENCE_MAX_BATCHES = None` in `config.py` to process the complete pinned
manifest. Models run sequentially so both 26B checkpoints are never intentionally
resident at the same time. BF16 inference requires roughly 58 GB for weights
before cache and telemetry allocations, so a single 80 GB A100 is the intended
configuration. A 40 GB A100 requires sharding, quantization, or offload.

Important defaults:

```python
MODELS_TO_RUN = ["gemma", "diffusion_gemma"]
INFERENCE_MAX_BATCHES = 1
LOG_MOE = True
LOG_LOGITS = True
LOG_SAVE_FULL_LOGITS = False
```

Full vocabulary logits are disabled because their storage cost is extreme.
Compact top-k, entropy, selected-token, and router-utilization summaries remain
enabled.

## Ten-example smoke test

Run the real end-to-end pipeline on the first 10 matching dataset records:

```powershell
python smoke_test_pipeline.py
```

The smoke runner uses both real models, writes the normal telemetry, creates the
paired comparison, and then checks the manifest, output counts, and core output
files. It forces `DATASET_MAX_SAMPLES = 10`, disables dataset shuffling, and
removes `INFERENCE_MAX_BATCHES` for that invocation so all 10 records are
processed regardless of batch size. It does not modify `config.py`.

Useful overrides:

```powershell
# Try only two records with Gemma first
python smoke_test_pipeline.py --samples 2 --models gemma

# Run both models on 10 records under a recognizable ID
python smoke_test_pipeline.py --experiment-id smoke-manual-10
```

This is a real model run, not a mocked unit test, so it still downloads/loads the
configured 26B checkpoints and needs the hardware described above.

`config.py` is intentionally a Python file because `config_parser.py` imports it
as a module (`import config`). Python import statements omit the `.py` suffix,
but the file itself needs it. A data-only configuration would instead use a file
such as `config.toml` or `config.yaml` plus corresponding loading code; an
extensionless `config` file would not work with the current loader.

## Experiment Data

Each invocation resolves the dataset and model revisions to exact Hub commit
hashes. It then creates one immutable `inputs.jsonl` before loading either model.
Both runners consume that file in the same order.

```text
logging/<experiment_id>/
  manifest.json
  dataset.json
  inputs.jsonl
  gemma/
    outputs.jsonl
    performance.jsonl
    tokens.jsonl
    logits.jsonl
    moe.jsonl
    source_summary.json
  diffusion_gemma/
    outputs.jsonl
    performance.jsonl
    canvas.jsonl
    logits.jsonl
    moe.jsonl
    source_summary.json
  comparison/
    pairs.jsonl
    source_summary.json
```

`performance.jsonl` records whether telemetry was enabled. Instrumented timing
should not be treated as a clean throughput benchmark because tensor reduction,
CPU transfer, and JSON serialization add overhead.
