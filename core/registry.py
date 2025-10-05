
from typing import Dict

_MODELS: Dict[str, type] = {}
_SAMPLERS: Dict[str, type] = {}

def register_model(name: str):
    def deco(cls):
        _MODELS[name] = cls
        return cls
    return deco

def get_model(name: str):
    if name not in _MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_MODELS)}")
    return _MODELS[name]

def register_sampler(name: str):
    def deco(cls):
        _SAMPLERS[name] = cls
        return cls
    return deco

def get_sampler(name: str):
    if name not in _SAMPLERS:
        raise KeyError(f"Unknown sampler '{name}'. Registered: {list(_SAMPLERS)}")
    return _SAMPLERS[name]
