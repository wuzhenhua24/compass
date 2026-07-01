"""Evaluator registry for managing evaluator plugins."""

from typing import Type

from compass.eval.base import Evaluator


_evaluator_registry: dict[str, Type[Evaluator]] = {}


def register_evaluator(name: str):
    """Decorator to register an evaluator.

    Args:
        name: Name to register the evaluator under.

    Returns:
        Decorator function.

    Example:
        @register_evaluator("my_eval")
        class MyEvaluator(Evaluator):
            ...
    """

    def decorator(cls: Type[Evaluator]) -> Type[Evaluator]:
        cls.name = name
        _evaluator_registry[name] = cls
        return cls

    return decorator


def get_evaluator(name: str) -> Type[Evaluator]:
    """Get an evaluator by name.

    Args:
        name: Name of the evaluator.

    Returns:
        Evaluator class.

    Raises:
        KeyError: If evaluator not found.
    """
    if name not in _evaluator_registry:
        raise KeyError(f"Evaluator '{name}' not found. Available: {list(_evaluator_registry.keys())}")
    return _evaluator_registry[name]


def list_evaluators() -> list[str]:
    """List all registered evaluators.

    Returns:
        List of evaluator names.
    """
    return list(_evaluator_registry.keys())


def unregister_evaluator(name: str) -> None:
    """Unregister an evaluator.

    Args:
        name: Name of the evaluator to remove.
    """
    _evaluator_registry.pop(name, None)
