"""Grader registry for managing grader plugins."""


from collections.abc import Callable

from compass.graders.base import Grader, GraderType

_grader_registry: dict[str, type[Grader]] = {}


def register_grader(
    name: str, grader_type: GraderType | None = None
) -> Callable[[type[Grader]], type[Grader]]:
    """Decorator to register a grader.

    Args:
        name: Name to register the grader under.
        grader_type: Optional type override.

    Returns:
        Decorator function.

    Example:
        @register_grader("my_grader")
        class MyGrader(CodeGrader):
            ...
    """

    def decorator(cls: type[Grader]) -> type[Grader]:
        cls.name = name
        if grader_type:
            cls.grader_type = grader_type
        _grader_registry[name] = cls
        return cls

    return decorator


def get_grader(name: str) -> type[Grader]:
    """Get a grader by name.

    Args:
        name: Name of the grader.

    Returns:
        Grader class.

    Raises:
        KeyError: If grader not found.
    """
    if name not in _grader_registry:
        raise KeyError(
            f"Grader '{name}' not found. Available: {list(_grader_registry.keys())}"
        )
    return _grader_registry[name]


def list_graders(grader_type: GraderType | None = None) -> list[str]:
    """List all registered graders.

    Args:
        grader_type: Optional filter by grader type.

    Returns:
        List of grader names.
    """
    if grader_type is None:
        return list(_grader_registry.keys())

    return [
        name
        for name, cls in _grader_registry.items()
        if cls.grader_type == grader_type
    ]


def unregister_grader(name: str) -> None:
    """Unregister a grader.

    Args:
        name: Name of the grader to remove.
    """
    _grader_registry.pop(name, None)
