from typing import Dict, Type
from .base import BaseEnv
# Import all environments to ensure they are registered
from .SAT1 import SAT1Env as SAT1

# A registry to map environment names to their classes
# It discovers all BaseEnv subclasses and registers them by their `name` attribute.
ENV_REGISTRY: Dict[str, Type[BaseEnv]] = {
    cls.name: cls for cls in BaseEnv.__subclasses__() if hasattr(cls, 'name')
} 