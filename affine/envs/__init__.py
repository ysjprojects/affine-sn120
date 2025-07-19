from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path

# Local environment imports
from .sat import SAT
from .abduction import ABDUCTION
from .gpqa import GPQA
from .deduction import DEDUCTION

# Public re-exports
__all__ = ["SAT", "ABDUCTION", "GPQA", "DEDUCTION"]

# Auto-import all env modules in this package
pkg_path = str(Path(__file__).parent)
for _, name, _ in iter_modules([pkg_path]):
    if name.startswith("__"):
        continue
    mod = import_module(f"{__name__}.{name}")
    if hasattr(mod, "ENV_CLASS"):
        globals()[name.upper()] = mod.ENV_CLASS
        __all__.append(name.upper())
