import importlib
import pkgutil

# import every submodule in this package (except private/dunder).
for m in pkgutil.iter_modules(__path__):
    name = m.name
    if name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{name}")

# build __all__ dynamically
__all__ = [m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")]
