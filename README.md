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

## First-time setup pitfalls & required secrets

### 1. Docker prerequisites
* Docker Engine ≥20.10 with the **Compose v2 plugin** (`docker compose version` must work)
* Outbound network (Docker Hub + S3).

### 2. Mandatory secrets
The validator never ships secrets inside the image; it expects them as
**environment variables** at runtime (Compose or host).  Missing vars cause the
container to exit immediately.

| Variable | Why it is needed |
|----------|------------------|
| `CHUTES_API_KEY`           | query LLM models & deployment helper |
| `AWS_ACCESS_KEY_ID`        | upload gzip traces & weight snapshots to S3 |
| `AWS_SECRET_ACCESS_KEY`    | idem |
| `TRACE_BUCKET`             | S3 bucket name (defaults to `affine-traces`) |
| `REDIS_HOST`               | host:port for the Redis instance (`redis` in compose) |
| `BT_COLDKEY` / `BT_HOTKEY` | (optional) deploy command, not used by validator |
| `HF_TOKEN`                 | (optional) model deploy helper |

Place them **once** in `~/.affine/config.env` so both CLI & Docker can load
them automatically:

```bash
mkdir -p ~/.affine
cat > ~/.affine/config.env <<'EOF'
CHUTES_API_KEY=xxx
AWS_ACCESS_KEY_ID=yyy
AWS_SECRET_ACCESS_KEY=zzz
TRACE_BUCKET=affine-traces
EOF
```

### 3. Local state directories
The app writes small files under `~/.affine/`:

```
~/.affine/samples      – env sample cache (auto-created)
~/.affine/results.json – raw validator outcomes
~/.affine/score.json   – 20-round rolling table
~/.affine/winners.json – per-round winners
```
You don’t need to create them; the code ensures parents exist, but make sure
your user has write permission.

### 4. First run checklist
```bash
# 1. source secrets OR ensure ~/.affine/config.env exists
source ~/.affine/config.env

# 2. Launch services
cd ops && docker compose up -d            # redis + validator + watchtower

# 3. Tail validator until it stays Up (healthy)
docker compose logs -f validator

# 4. Tail watchtower and observe first pull (5-min default)
docker compose logs -f watchtower
```

If the validator container restarts, run `docker compose logs validator` and
look for `get_conf()` complaining about a missing var.

---

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

## Challenge environments

Active set:

* **SAT** – Boolean satisfiability puzzles
* **ABDUCTION** – simple propositional-logic abduction (find the missing cause)
* **MATH** – elementary arithmetic word problems (GSM8K-style)

The registry lives in `affine/envs/__init__.py`; plug-ins follow the `BaseEnv` interface.

## Deployment pipeline (HF → Commit-Reveal → Chutes)

`af deploy <local_model_dir>` performs the full release flow:

1. Pushes `local_model_dir` to HuggingFace under a **private** repo (or re-uses `--existing-repo`).
2. Captures the commit SHA and computes the next Bittensor round (10-block cadence); emits `set_reveal_commitment` with JSON payload `{"model": …, "revision": <sha>}`.
3. Waits until the reveal block is reached, then switches the HF repo to **public**.
4. Generates a Chutes template embedding `--revision <sha>` so the exact code is deployed.
5. Deploys to Chutes and performs a warm-up request.

All credentials are read from `~/.affine/config.env`; use `af set KEY VALUE` to configure.

## Validator docker stack

The validator can run unattended via Docker Compose + Watchtower:

```
ops/docker/
├── validator.Dockerfile
└── docker-compose.validator.yml
```

```bash
# build & launch
make overnight   # or docker compose -f ops/docker/docker-compose.yml up -d
```

Watchtower checks for new images every 5 min and restarts the `validator` service automatically.

### Snapshots

Every 10-block window the validator uploads a snapshot to Cloudflare R2:

* `snapshots/<window_start>.json` – window summary & weight vector
* `snapshots/scores/<window_start>.json` – full exponentially-smoothed score matrix

Raw challenge results are stored under `raw/<wallet_hotkey>/<block>.json`.

---

See `affine/__init__.py` for implementation details and additional CLI commands. 