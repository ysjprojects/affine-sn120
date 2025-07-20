# Affine

## Introduction

Affine is a an incentivized RL environment which pays miners which make incremental improvements on a set of tasks (for instance, program abduction or coding). The mechanism is sybil-proof (you cant cheat by deploying multiple miners), decoy-proof (you cant cheat by packing models into certain environments), copy-proof (you cant win by simply stealing the best model), overfitting-proof (you can't cheat by overfitting to the a single benchmark). 

How does Affine work? Affine validators incentivize miners to submit models to Subnet 64 on Bittensor (a.k.a Chutes) where they are inference load balanced and publicly available. These models are evaluated on a set of RL-environments with validators looking for the model which dominates the pareto frontier -- namely the model which out competes all other models on all envs (see `af validator`) The network is winners-take-all where miners are forced to copy, download and improve the pareto frontier model.

Why affine? Directed incentives for RL have never been achieved. The ability to direct intelligence and aggregate the work-effort of a large non-permissioned group of individuals on RL tasks will unlock fast advancement in intelligence, we intend to commoditize reasoning (intelligence's highest form) and break the intelligence sound barrier.

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

Set env vars.
```bash
# Copy .env and fill out items.
cp .env.validator.example .env
```

Recommended: Run the validator with docker.
```bash
# Run the validator with watchtower.
docker-compose pull && docker-compose up -d
```

Run the validator locally
```bash
# Start the validator with debug.
af -vv validate
```

# Mining

Miners need a chutes developer account to deploy models and this requires an upfront 5k USD in TAO (which you get this back later).
Visit `chutes.ai` set your account up. You will also need a Huggingface account to deploy new models. Once you have all of these requirements do the following: 
```bash
# Register to chutes.
chutes register 

# Set your chutes api key information.
af set CHUTES_API_KEY <your chutes API key> 
af set CHUTE_USER <your chutes username> 

# Set your HF account/token.
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


# SDK
Affine is also an SDK you can use to generate and evaluate envs.
```python
import affine as af

# Get all miner model endpoints
miners = await af.miners()

# Get a miner endpoint for a UID
miner = await af.miners( 5 )

# Generate a SAT challenge
chal = await af.SAT.generate() 

# Generate a bunch.
chals = await af.ABDUCTION.many( 10 )
chals = await af.DEDUCTION.many( 10 )

# Query the model directly.
response = await af.query( chal.prompt, model = miner.model )

# Evaluate the response
evaluation = chal.evaluate( response ) 
print( evaluation.score )

# Query the miner and do the eval all in one go.
results = await af.run( chals, miners )
```

## Docker Deployment

This project includes a Docker setup for running the validator in a containerized environment.

### Prerequisites

- Docker
- Docker Compose

### Setup

1.  **Environment Variables**:
    The service requires two environment files:
    -   `~/.affine/config.env`: For user-specific configurations.
    -   `.env`: For project-specific variables.

    An example file, `.env.validator.example`, is provided. Copy it to `.env` and customize it as needed:
    ```bash
    cp .env.validator.example .env
    ```

2.  **Build and Run with Docker Compose**:
    To build and start the `validator` and `watchtower` services, run:
    ```bash
    docker-compose up --build -d
    ```
    The `-d` flag runs the containers in detached mode.

3.  **Watchtower**:
    The `watchtower` service will automatically monitor for new versions of the `affine-validator` image and redeploy the service when an update is available. This ensures the validator is always running the latest code.
