# Affine

1.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2.
```bash
git clone https://github.com/chutes/affine.git
cd affine
uv pip install -e .
```

3.
```bash
af init
```

4.
```bash
af run -m <model_name> -e <environment_name> -n <number_of_questions>
# e.g. af run -m unsloth/gemma-3-4b-it -e SAT1 -n 10
```

5.
```bash
af validate --coldkey <your-coldkey> --hotkey <your-hotkey>
```
