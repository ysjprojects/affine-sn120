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
# Set secrets (will be read as ENV at runtime)
uv run af set CHUTES_API_KEY <your_key_value>
uv run af set HF_TOKEN        <your_HF_TOKEN>
export AWS_ACCESS_KEY_ID=XXXX
export AWS_SECRET_ACCESS_KEY=YYYY

# Evaluate UID 5 on the SAT env with 10 samples
uv run af -vv run 5 SAT -n 10

# Deploy models (optional helper)
af deploy /path/to/model
```

## Running the validator in Docker (production)

```bash
# 1 — Build + push the image (requires Docker Hub login)
make push                        # builds affinefdn/validator:latest

# 2 — On the server
cd ops
docker compose up -d             # redis + validator + watchtower

# Watchtower auto-pulls the tag every 5 min (see docker-compose.yml)
docker compose logs -f watchtower
```

Validator container entry-point:
```
python -m affine validate --delay 60
```
It loops over all registered environments (>10 synthetic ones) and pushes
gzip traces to the S3 bucket defined by `$TRACE_BUCKET` (default
`affine-traces`).

## Manual roll-up of weights

```bash
# After an epoch length ($AFFINE_WINDOW$ blocks)
docker compose exec validator python -m affine.rollup_job
# commits `strength:{hotkey}` in Redis and uploads a snapshot to S3
```

## Developer workflow

```bash
pytest -q          # fast unit tests (Elo, power_rule, etc.)
make build         # local image build only
af -vv validate    # local validator loop (Redis must be running)
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