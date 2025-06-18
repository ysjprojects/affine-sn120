import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='LLM_')

    api_key: str
    api_url: str = "https://llm.chutes.ai/v1/chat/completions"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_base: float = 0.5

class AppSettings(BaseSettings):
    log_level: LogLevel = "INFO"
    concurrency: int = 10
    # More app-specific settings can be added here

class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings(api_key=os.getenv("CHUTES_API_KEY"))
    app: AppSettings = AppSettings()

settings = Settings() 