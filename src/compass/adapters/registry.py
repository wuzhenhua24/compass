"""Adapter registry for managing agent adapters."""


from collections.abc import Callable

from compass.adapters.base import Adapter

_adapter_registry: dict[str, type[Adapter]] = {}


def register_adapter(name: str) -> Callable[[type[Adapter]], type[Adapter]]:
    """Decorator to register an adapter.

    Args:
        name: Name to register the adapter under.

    Returns:
        Decorator function.

    Example:
        @register_adapter("my_agent")
        class MyAgentAdapter(Adapter):
            ...
    """

    def decorator(cls: type[Adapter]) -> type[Adapter]:
        cls.name = name
        _adapter_registry[name] = cls
        return cls

    return decorator


def get_adapter(name: str) -> type[Adapter]:
    """Get an adapter by name.

    Args:
        name: Name of the adapter.

    Returns:
        Adapter class.

    Raises:
        KeyError: If adapter not found.
    """
    if name not in _adapter_registry:
        raise KeyError(f"Adapter '{name}' not found. Available: {list(_adapter_registry.keys())}")
    return _adapter_registry[name]


def list_adapters() -> list[str]:
    """List all registered adapters.

    Returns:
        List of adapter names.
    """
    return list(_adapter_registry.keys())


def unregister_adapter(name: str) -> None:
    """Unregister an adapter.

    Args:
        name: Name of the adapter to remove.
    """
    _adapter_registry.pop(name, None)
