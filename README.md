# Affine
---
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
---
```bash
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv pip install -e .
```
---
```bash
af init --api-key <your-chutes-api-key>
```
---
```bash
af run -m <model_name> -e <environment_name> -n <number_of_questions>
# e.g. af run -m unsloth/gemma-3-4b-it -e SAT1 -n 10
```
---
```bash
af validate --coldkey <your-coldkey> --hotkey <your-hotkey>
```
