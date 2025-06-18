import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from rich.table import Table
from rich.console import Console
from affine.environments.base import BaseEnv
from affine.config_utils import RESULTS_DIR

console = Console()

def get_summary_table(results_by_model: Dict[str, List[Dict[str, Any]]]) -> Table:
    """Generates a Rich table from aggregated results."""
    table = Table(header_style="table.header", box=None, show_header=True, title_style="", caption_style="")
    table.add_column("Model", justify="left", no_wrap=True)
    table.add_column("Accuracy", justify="center")
    table.add_column("Correct", justify="center")
    table.add_column("Incorrect", justify="center")
    table.add_column("Total", justify="center")
    table.add_column("Avg Latency (s)", justify="right")

    for model, results in results_by_model.items():
        valid_results = [r for r in results if r.get("error") is None]
        correct_count = sum(1 for r in valid_results if r.get("metrics", {}).get("correct"))
        total_for_model = len(valid_results)
        accuracy = correct_count / total_for_model if total_for_model > 0 else 0
        avg_latency = sum(r.get("latency_seconds", 0) for r in valid_results) / len(valid_results) if valid_results else 0
        
        incorrect_count = total_for_model - correct_count
        table.add_row(
            model,
            f"{accuracy:.1%}",
            str(correct_count),
            str(incorrect_count),
            str(total_for_model),
            f"{avg_latency:.2f}s"
        )
    return table

def load_results(
    models: List[str],
    env_name: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads all results for a given list of models and an environment.
    """
    results_by_model: Dict[str, List[Dict[str, Any]]] = {model: [] for model in models}
    base_path = RESULTS_DIR / env_name

    for model in models:
        sanitized_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        model_path = base_path / sanitized_model_name
        
        if not model_path.exists():
            continue

        for result_file in model_path.glob("*.json"):
            if result_file.name == "latest.json":
                continue
            with open(result_file, 'r') as f:
                data = json.load(f)
                # The structure is nested, we want the list of individual results
                if "model_results" in data and model in data["model_results"]:
                    model_run_data = data["model_results"][model]
                    if "results" in model_run_data:
                         results_by_model[model].extend(r for r in model_run_data["results"] if r is not None)

    return results_by_model

def get_results(
    models: List[str],
    env: Type[BaseEnv] = None,
    env_name: Optional[str] = None,
    display: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Main function to get results for the SDK and CLI.
    Can either take an environment class or an environment name string.
    """
    if env:
        name = env.name
    elif env_name:
        name = env_name
    else:
        raise ValueError("Must provide either 'env' or 'env_name'")

    results_data = load_results(models, name)
    
    if display:
        if any(results_data.values()):
            table = get_summary_table(results_data)
            console.print(table)
        else:
            console.print(f"No results found for models {models} in environment '{name}'.")

    # Format for SDK return value
    sdk_results = [results_data[model] for model in models]
    return sdk_results 

def get_results_files(
    models: List[str],
    env_name: str,
    latest: bool = False
) -> List[Path]:
    """Loads all results for a given list of models and an environment."""
    results_files = []
    base_path = RESULTS_DIR / env_name
    if not base_path.exists():
        return []

    for model_name in models:
        sanitized_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        model_path = base_path / sanitized_model_name
        if model_path.exists():
            for result_file in model_path.glob("*.json"):
                if result_file.name == "latest.json":
                    continue
                results_files.append(result_file)
    return results_files 