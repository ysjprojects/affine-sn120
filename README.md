# Affine

## Introduction

Affine is a multi-RL environment incentive landscape which pays miners for making incremental improvements on a set of tasks (for instance, program abduction or coding). The mechanism is sybil-proof (you cant cheat by deploying multiple miners), decoy-proof (you cant cheat the mechanism by packing models into certain environments), copy-proof (you cant win by simply stealing the best model), overfitting-proof (you can't cheat by overfitting to the a single benchmark instead of all of them). 

How does Affine work? Affine validators incentivize miners to submit models to Subnet 64 on Bittensor (a.k.a Chutes) where they are inference load balanced and publicly available. These models are evaluated on a set of RL-environment heirarchies with validators looking for the model which dominates the pareto frontier -- namely the model which out competes all other models on all envs (see `af validator`) The network is winners-take-all where miners are forced to copy, download and improve the pareto frontier model.

Why affine? Directed incentives for RL have never been achieved. The ability to direct intelligence, and aggregate the work-effort of a large non-permissioned group of individuals on RL tasks will allow us to unlock fast advancement in intelligence, we intend to commoditize reasoning (intelligence's highest form) and break the intelligence sound barrier.

## Installation
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

## Validating

Before running your validator you need to make a chutes account and get your API key. 
```bash
# You need a chutes API key to run the validator.
af set CHUTES_API_KEY <your chutes API key> 
```

Optional: Add an R2 bucket to help telemetry on the network.
```bash
af set R2_BUCKET_ID <r2 bucket id>
af set R2_ACCOUNT_ID <r2 account id>
af set R2_WRITE_ACCESS_KEY_ID <r2 write access key>
af set R2_WRITE_SECRET_ACCESS_KEY <r2 secret access key>
```

Run the validator.
```bash
# Start the validator with trace on.
af -vvv validate --coldkey <your coldkey> --hotkey <your hotkey>
```

# Mining

First you need to set up chutes developer account a huggingface account.
NOTE: A developer account costs 5k USD in TAO but you get this back after. 
NOTE: You need to register to the affine network netuid=120. 
```bash
# Install chutes on top of your env
uv pip install chutes

# Register to chutes you need an account etc.
chutes register 

# Set your chutes key
af set CHUTES_API_KEY <your chutes API key> 
af set CHUTE_USER <your chutes username> 

# Set your HF account/token (for model uploading)
af set HF_TOKEN <your hf token>
af set HF_USER <your hf user> 
```

Pull a model off the network.
```bash
af -vvv pull <uid to pull> --model_path <model_location>
```

Improve the model
```bash
... magic RL stuff ...
```

Push the model to my key.
```bash
af -vvv push  --coldkey <your coldkey> --hotkey <your hotkey> --model_path <model_location>
```
