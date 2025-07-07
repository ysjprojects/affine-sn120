# Affine

## Prerequisites
1. **Register on Chutes**: [https://github.com/rayonlabs/chutes](https://github.com/rayonlabs/chutes) - Get your API keys
2. **Enable Developer Account**: [https://chutes.ai/app/docs](https://chutes.ai/app/docs) - Required for deployment


## Installation
```bash
# Install uv Astral
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate
uv pip install -e .

# Verify installation
af --help
```


## Quick Start
```bash
# Set your chutes api key
uv run af set CHUTES_API_KEY <your_key_value>
uv run af set HF_TOKEN <your_HF_TOKEN>

# Eval uid:5 miner on the SAT env with 10 samples
uv run af -vv run 5 SAT -n 10

# Deploy models
af deploy /path/to/model
```

# Affine – Validator

```bash
export AFFINE_WINDOW=360           # spec-compliant window length

# validator (per-block tests)
af -vvv validate                    # SAT & ABD; add RES/GAIA in _cycle_once()

# force epoch roll-up (every 360 blocks)
python -m affine.rollup            # prints weight vector, ready for on-chain

# diagnostics
pytest -q                          # unit tests
``` 