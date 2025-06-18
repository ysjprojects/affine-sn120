#!/usr/bin/env python3
import sys
import asyncio
import logging
from typing import Optional, Dict, Any, List

import click
from rich.console import Console

from affine.utils import setup_logging, parse_env_kwargs
from affine.runner import run_llm_batch
from affine.environments import ENV_REGISTRY
from affine.config import settings
from affine.exceptions import AffineError
from affine.results import get_results
from affine.validator import run_validator
from affine.config_utils import update_config, get_config, CONFIG_FILE

logger = logging.getLogger("tool")
console = Console()

@click.group()
def cli():
    """A CLI tool for running evaluations on LLMs."""
    pass

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--models", "-m", "models", multiple=True, required=True, help="Model name(s) to run.")
@click.option("--n", "-n", type=int, default=1, help="Number of questions to generate per model.")
@click.option(
    "--out", "-o", 
    default=None,
    help="Output file path (optional, defaults to structured path in /results)"
)
@click.option(
    "--env-class", "-e",
    type=click.Choice(list(ENV_REGISTRY.keys()), case_sensitive=False),
    default=next(iter(ENV_REGISTRY.keys())),  # Set the first key as the default
    help="Which Env to use (default: first available)."
)
@click.option("--concurrency", "-c", type=int, default=None, help=f"Number of concurrent LLM queries (default: {settings.app.concurrency}).")
@click.option(
    "--log-level", "-l",
    type=click.Choice(["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default=None,
    help="Set logging level (default: INFO)"
)
@click.option("--debug", is_flag=True, help="Enable debug logging (equivalent to --log-level DEBUG)")
@click.option("--trace", is_flag=True, help="Enable trace logging (equivalent to --log-level TRACE)")
@click.option("--quiet", "-q", is_flag=True, help="Quiet mode - only show warnings and errors")
@click.pass_context
def run(ctx, models: tuple[str], n: int, out: Optional[str], env_class: str, concurrency: Optional[int], log_level: Optional[str], debug: bool, trace: bool, quiet: bool):
    """
    Batch‐dispatch generated prompts via an Env and save responses + verification metrics to JSON.
    
    To pass arguments to the environment, use a '--' separator after the run options.
    Example: af run -e SAT1 -m <model> -- --n 10 --k 3
    """
    env_kwargs = parse_env_kwargs(ctx.args)
    
    # Determine the effective log level
    if trace:
        level = "TRACE"
    elif debug:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    elif log_level:
        level = log_level.upper()
    else:
        level = "INFO" # Default
    
    if concurrency:
        settings.app.concurrency = concurrency
    
    settings.app.log_level = level
    setup_logging()

    logger.log(logging.DEBUG - 5 if settings.app.log_level == "TRACE" else logging.DEBUG, 
               f"CLI args: models={models}, n={n}, out={out}, env_class={env_class}, env_kwargs={env_kwargs}, concurrency={settings.app.concurrency}, log_level={settings.app.log_level}")
    
    # Instantiate the chosen Env from the registry
    EnvCls = ENV_REGISTRY[env_class]
    try:
        env_instance = EnvCls(**env_kwargs)
    except (TypeError, ValueError) as e:
        logger.error(f"Error instantiating environment '{env_class}': {e}", exc_info=True)
        click.echo(f"Error: Invalid arguments for environment '{env_class}'.\n{e}", err=True)
        sys.exit(1)
        
    try:
        asyncio.run(run_llm_batch(models, n, out, env_instance))
    except AffineError as e:
        logger.error(f"A handled error occurred: {e}", exc_info=False)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting.")
        sys.exit(1)

@cli.command()
@click.option("--models", "-m", "models", multiple=True, required=True, help="Model name(s) to load results for.")
@click.option(
    "--env", "-e", "env_name",
    type=click.Choice(list(ENV_REGISTRY.keys()), case_sensitive=False),
    required=True,
    help="Which Env to load results for."
)
def results(models: tuple[str], env_name: str):
    """
    Load, aggregate, and display previous results from JSON files.
    """
    get_results(models=list(models), env_name=env_name, display=True)

@cli.command()
@click.option("--coldkey", type=str, required=True, help="Coldkey name for the wallet.")
@click.option("--hotkey", type=str, required=True, help="Hotkey name for the wallet.")
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def validator(coldkey: str, hotkey: str, debug: bool):
    """
    Run the validator logic with the specified coldkey and hotkey.
    """
    if debug:
        settings.app.log_level = "DEBUG"
    else:
        settings.app.log_level = "INFO"

    setup_logging()
    logger.info(f"Starting validator with coldkey: {coldkey} and hotkey: {hotkey}")
    try:
        asyncio.run(run_validator(coldkey, hotkey, debug=debug))
    except AffineError as e:
        logger.error(f"A handled error occurred: {e}", exc_info=False)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting.")
        sys.exit(1)

@cli.command()
@click.option('--api-key', prompt="Chutes API Key", help="Your Chutes API key.")
def init(api_key: str):
    """Initialize affine with your Chutes API key."""
    update_config('chutes', 'api_key', api_key)
    console.print(f"API key set in {CONFIG_FILE}")

@click.group()
def config():
    """Manage affine configuration."""
    pass

@config.command()
def show():
    """Show the current configuration."""
    config = get_config()
    for section in config.sections():
        print(f"[{section}]")
        for key, value in config.items(section):
            print(f"{key} = {value}")

@config.command()
@click.argument('key')
def get(key: str):
    """Get a configuration value."""
    section, option = key.split('.')
    config = get_config()
    if section in config and option in config[section]:
        console.print(config.get(section, option))
    else:
        console.print(f"Key {key} not found.", style="bold red")

@config.command()
@click.argument('key')
@click.argument('value')
def set(key: str, value: str):
    """Set a configuration value."""
    section, option = key.split('.')
    update_config(section, option, value)
    console.print(f"Set {key} to {value} in {CONFIG_FILE}")

cli.add_command(config)

if __name__ == "__main__":
    cli()