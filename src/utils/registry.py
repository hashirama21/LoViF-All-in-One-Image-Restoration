"""
Central registry — maps string keys to classes.
Pattern: @Registry.register("key") on any class.
Usage:  model = ModelRegistry.build("foundir", **kwargs)
"""
from __future__ import annotations
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry. Subclass once per domain (models, losses, …)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str):
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._registry:
                raise KeyError(f"[{self._name}] '{key}' is already registered.")
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, key: str, **kwargs: Any) -> Any:
        if key not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"[{self._name}] Unknown key '{key}'. Available: {available}"
            )
        return self._registry[key](**kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self):
        return list(self._registry.keys())


# Domain-specific registries — import these everywhere
ModelRegistry = Registry("models")
LossRegistry = Registry("losses")
DatasetRegistry = Registry("datasets")
SchedulerRegistry = Registry("schedulers")
