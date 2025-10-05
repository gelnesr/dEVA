from abc import ABC, abstractmethod
from typing import List, Dict, Any

from core.registry import get_model
from evolve.individual import Individual

class BaseModel(ABC):
    """Pluggable scoring/prediction models must implement this interface."""
    @abstractmethod
    def setup(self, config: str, device: str) -> None:
        """Load weights, initialize device, etc."""

    @abstractmethod
    def score(self, individual: Individual):
        """Return a dict of objective_name -> float."""

def build_models(specs: List[str]):
    models = {}
    for spec in specs:
        if ":" in spec:
            name, argstr = spec.split(":", 1)
            kwargs: Dict[str, Any] = eval(argstr, {}, {})  # replace with safer parser if needed
        else:
            name, kwargs = spec, {}
        ModelCls = get_model(name)
        models[spec] = ModelCls(**kwargs)
    return models