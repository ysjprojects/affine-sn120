# Affine Miner Setup Guide

This guide summarizes the complete setup process for running an Affine miner on Subnet 64 (Chutes).

## Prerequisites

**CRITICAL**: You need a **developer-enabled account** on Chutes to mine. Regular API keys cannot deploy chutes.

### Required Accounts & Keys

1. **Chutes Developer Account** (`chutes.ai`)
   - Sign up and get approved for developer access
   - Obtain API key starting with `cpk_...`

2. **Hugging Face Account** (`huggingface.co`)
   - Create account and get username
   - Generate write-access token starting with `hf_...`

3. **Bittensor Wallet**
   - Coldkey (for registration/ownership)
   - Hotkey (for mining operations)

## Setup Steps

### 1. Install Dependencies

```bash
# Install uv Astral
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify installation
af
```

### 2. Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Chutes Configuration
CHUTES_API_KEY=cpk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CHUTE_USER=myusername

# Hugging Face Configuration
HF_USER=myaccount
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Bittensor Wallet Names
BT_WALLET_COLD=default
BT_WALLET_HOT=default

# Subtensor Configuration
SUBTENSOR_ENDPOINT="finney"
SUBTENSOR_FALLBACK="wss://lite.sub.latent.to:443"
```

### 3. Register Chutes Account

```bash
chutes register
```

### 4. Register on Subnet 64

```bash
btcli subnet register --wallet.name <your_cold> --wallet.hotkey <your_hot>
```

## Mining Workflow

### 1. Pull Existing Model

Pull a model from the network to improve:

```bash
af -vvv pull <uid_to_pull> --model_path ./my_model
```

### 2. Improve the Model

Implement your RL improvements on the downloaded model:

```bash
# Your custom training/improvement process
# ... magic RL stuff ...
```

### 3. Push Updated Model

Deploy your improved model:

```bash
af -vvv push --coldkey <your_cold> --hotkey <your_hot> --model_path ./my_model
```

## What the Push Process Does

The `af push` command automates several critical steps:

1. **Hugging Face Deployment**
   - Creates/updates private HF repository
   - Uploads model files with concurrency limits
   - Makes repository public after upload

2. **Chutes Deployment**
   - Generates SGLang chute configuration
   - Deploys model as inference endpoint
   - Configures for 8x GPUs with 24GB+ VRAM each

3. **On-Chain Registration**
   - Commits model metadata to Bittensor subnet
   - Includes model repo, revision hash, and chute ID
   - Handles quota limits with automatic retry

4. **Model Warming**
   - Sends SAT challenges to warm up the model
   - Continues until model is marked "hot" and ready

## Configuration Details

### Chute Specifications

Models are deployed with:
- **Image**: `chutes/sglang:0.4.9.post3`
- **Concurrency**: 20 parallel requests
- **GPU Requirements**: 8x GPUs with 24GB+ VRAM each
- **Engine Args**: `--trust-remote-code`

### Environment Variables

- `AFFINE_UPLOAD_CONCURRENCY`: Controls HF upload concurrency (default: 2)
- File uploads skip hidden files, cache directories, and lock files

## Troubleshooting

### Common Issues

1. **Chutes Deployment Failure**
   - Verify developer account status
   - Check API key validity
   - Ensure sufficient GPU quota

2. **HuggingFace Upload Issues**
   - Verify write token permissions
   - Check repository naming conflicts
   - Monitor for 429 rate limit errors

3. **On-Chain Commitment Issues**
   - `SpaceLimitExceeded`: Tool waits one block and retries
   - Verify wallet has sufficient balance for transactions

### Support

- [Affine Discord](https://discord.com/invite/3T9X4Yn23e)
- Check logs with `-vvv` flag for detailed debugging

## Security Notes

- Keep API keys and tokens secure
- HF repositories are made private during upload, then public
- Wallets should follow Bittensor security best practices