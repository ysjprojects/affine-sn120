# Affine

Affine.

## Installation

To get started with Affine, you'll need to install it from source using `uv`, a fast Python package installer from Astral.

First, install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, clone the Affine repository and install it.

```bash
git clone https://github.com/chutes/affine.git
cd affine
uv pip install -e .
```

## Getting Started

### Initialization

Before you can use Affine, you need to configure it with your Chutes API key. Run the `init` command and follow the prompts:

```bash
af init
```
This command creates a `~/.affine` directory on your local machine, which will store your configuration in `config.ini` and place all evaluation results in a `results/` subdirectory.

### Running Evaluations

You can run an evaluation using the `run` command. You need to specify the model you want to test and the environment to use.

```bash
af run -m <model_name> -e <environment_name> -n <number_of_questions>
```

For example, to run 10 questions against the `unsloth/gemma-3-4b-it` model using the `SAT1` environment:
```bash
af run -m unsloth/gemma-3-4b-it -e SAT1 -n 10
```

### Running a Validator

Affine also includes a validator to check the integrity and correctness of your evaluation setup. To run the validator, you need to provide your wallet's coldkey and hotkey:

```bash
af validate --coldkey <your-coldkey> --hotkey <your-hotkey>
```
This will run a series of checks to ensure that your environment is correctly configured and that the evaluation process can run smoothly.

## Configuration Management

You can manage your configuration using the `af config` command:

- `af config show`: Displays the current configuration.
- `af config get <section>.<key>`: Retrieves a specific configuration value.
- `af config set <section>.<key> <value>`: Sets or updates a configuration value.
