import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from affine.config_utils import get_config, ensure_affine_dir

# Ensure the affine directory and config file exist
ensure_affine_dir()

# Load the configuration
config = get_config()

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class LLMSettings(BaseSettings):
    api_key: str = None
    api_url: str = "https://llm.chutes.ai/v1/chat/completions"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 0.5
    model_config = SettingsConfigDict(env_prefix='LLM_')

class AppSettings(BaseSettings):
    log_level: LogLevel = "INFO"
    concurrency: int = 10
    # More app-specific settings can be added here
    model_config = SettingsConfigDict(env_prefix='APP_')

class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    llm: LLMSettings = LLMSettings(api_key=config.get('chutes', 'api_key', fallback=os.getenv("CHUTES_API_KEY")))

settings = Settings() 