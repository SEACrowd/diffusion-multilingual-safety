# Diffusion Multilingual Safety

## Installation

Create and activate a Python environment first. PyTorch stable currently requires Python 3.10 or newer.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install or upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

Install stable PyTorch with CUDA support from the official PyTorch pip index:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

This uses the CUDA 13.0 PyTorch wheel index (`cu130`). If pip cannot find matching wheels, check the official PyTorch install selector for the currently published stable CUDA wheel for your OS and Python version.

Then install the rest of the Hugging Face and project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the PyTorch install:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```