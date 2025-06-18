import os
import re
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from affine.config import settings
from affine.logging_config import AppLogHandler
from affine.config_utils import RESULTS_DIR

console = Console()
logger = logging.getLogger("affine")


def setup_logging():
    """Dynamically sets up logging based on the application settings."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_level = logging.getLevelName(settings.app.log_level.upper())
    
    # Add our custom styled handler
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[AppLogHandler()]
    )
    
    # Disable noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_env_kwargs(kwargs_tuple: tuple[str]) -> Dict[str, Any]:
    """
    Parses a tuple of command-line arguments into a dictionary of keyword arguments.
    Handles --key value pairs and --flag boolean flags.
    """
    kwargs_dict: Dict[str, Any] = {}
    i = 0
    while i < len(kwargs_tuple):
        arg = kwargs_tuple[i]
        if arg.startswith('--'):
            key = arg[2:].replace('-', '_') # Convert kebab-case to snake_case for kwargs
            if (i + 1) < len(kwargs_tuple) and not kwargs_tuple[i+1].startswith('--'):
                # This is a key-value pair
                value_str = kwargs_tuple[i+1]
                # Attempt to convert to numeric types or booleans
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() == 'true':
                            value = True
                        elif value_str.lower() == 'false':
                            value = False
                        else:
                            value = value_str  # It's a string
                kwargs_dict[key] = value
                i += 2
            else:
                # This is a boolean flag
                kwargs_dict[key] = True
                i += 1
        else:
            # This should not happen with the -- separator, but we'll be robust
            logger.warning(f"Skipping unexpected argument: {arg}")
            i += 1
    return kwargs_dict


def get_output_path(model_name: str, env_name: str, custom_path: Optional[str] = None) -> Path:
    """
    Determines the output path for results.
    If a `custom_path` is provided, it's used directly.
    Otherwise, a structured path `~/.affine/results/{env_name}/{model_name}/{timestamp}.json` is generated.
    """
    if custom_path:
        path = Path(custom_path)
        # If custom_path is a directory, create a timestamped filename
        if path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
            return path / f"{timestamp}.json"
        return path

    sanitized_model_name = sanitize_model_name(model_name)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = RESULTS_DIR / env_name / sanitized_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{timestamp}.json"


def sanitize_model_name(model_name: str) -> str:
    # Sanitize model name for use in a directory path
    sanitized_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    return sanitized_model_name 