import importlib
import pkgutil
from typing import Type

from affine.environments.base import BaseEnv


class Envs:
    def __getattr__(self, name: str) -> Type[BaseEnv]:
        try:
            module = importlib.import_module(f"affine.environments.{name}")
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, BaseEnv) and attribute is not BaseEnv:
                    # Cache it
                    setattr(self, name, attribute)
                    return attribute
            raise AttributeError(f"Could not find a BaseEnv subclass in module affine.environments.{name}")
        except ImportError:
            raise AttributeError(f"Environment {name} not found.")


envs = Envs() 