# Affine Tool

This is a command-line tool to batch-dispatch generated prompts to an LLM endpoint and save responses along with verification metrics to a JSON file.

## Installation

To install the package, run the following command in your terminal. This command should be run from the root of the project directory (where `pyproject.toml` is located).

```bash
pip install .
```

For development, you can install it in editable mode. This allows you to make changes to the code and have them reflected immediately without reinstalling.

```bash
pip install -e .
```

## Usage

After installation, you can use the `affine` command from anywhere in your terminal:

```bash
affine --model "unsloth/gemma-3-4b-it" --n 2 --out file.json -e MathSynthEnv
```

### Options

- `--model, -m`: (Required) Model name (e.g. `unsloth/gemma-3-4b-it`).
- `--n, -n`: Number of questions to generate (default: 1).
- `--out, -o`: (Required) Output file path (e.g. `results.json`).
- `--env-class, -e`: (Required) Which Env to use. Currently supports `MathSynthEnv`.
- `--log-level, -l`: Set logging level (`TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Default is `INFO`.
- `--debug`: Enable debug logging.
- `--trace`: Enable trace logging (maximum verbosity).
- `--quiet, -q`: Quiet mode, only show warnings and errors.

## Environment Variables

The tool requires the `CHUTES_API_KEY` environment variable to be set for authenticating with the LLM API.

```bash
export CHUTES_API_KEY="your-api-key"
```

You can also configure the following optional variables:
- `LLM_API_URL`
- `LLM_TIMEOUT`
- `LLM_MAX_RETRIES`
- `LLM_BACKOFF_BASE`
