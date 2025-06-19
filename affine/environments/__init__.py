from typing import Dict, Type
from .base import BaseEnv

# Import all environments to ensure they can be registered
from .SAT1 import SAT1Env

# A registry to map environment names to their classes
ENV_REGISTRY: Dict[str, Type[BaseEnv]] = {
    "SAT1": SAT1Env,
} 