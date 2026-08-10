"""Sandbox registry for managing sandbox implementations."""


from collections.abc import Callable

from compass.sandbox.base import Sandbox

_sandbox_registry: dict[str, type[Sandbox]] = {}


def register_sandbox(name: str) -> Callable[[type[Sandbox]], type[Sandbox]]:
    """Decorator to register a Sandbox subclass.

    Args:
        name: Name to register the sandbox under.

    Returns:
        Decorator function.

    Example:
        @register_sandbox("local")
        class LocalSandbox(Sandbox):
            ...
    """

    def decorator(cls: type[Sandbox]) -> type[Sandbox]:
        _sandbox_registry[name] = cls
        return cls

    return decorator


def get_sandbox_class(name: str) -> type[Sandbox]:
    """Look up a Sandbox class by its registered name.

    Args:
        name: Name of the sandbox type.

    Returns:
        Sandbox class.

    Raises:
        KeyError: If the sandbox type is not registered.
    """
    if name not in _sandbox_registry:
        raise KeyError(
            f"Sandbox '{name}' not found. Available: {list(_sandbox_registry.keys())}"
        )
    return _sandbox_registry[name]


def list_sandboxes() -> list[str]:
    """List all registered sandbox names.

    Returns:
        List of sandbox names.
    """
    return list(_sandbox_registry.keys())


def unregister_sandbox(name: str) -> None:
    """Unregister a sandbox.

    Args:
        name: Name of the sandbox to remove.
    """
    _sandbox_registry.pop(name, None)
