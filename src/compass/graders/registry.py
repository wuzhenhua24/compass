"""Grader registry for managing grader plugins.

Built-in framework graders register at import time. Domain graders
(``compass.graders.domains``) do not — they are loaded the first time a
lookup misses, so that importing Compass costs nothing for domains a user
is not evaluating. The fallback is what makes that invisible: a scenario
naming ``sql_equivalence`` gets it, with nothing to declare.
"""


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
        # A miss may just mean the domain that owns this grader has not been
        # imported yet. Pay for that import once, here, rather than at startup.
        from compass.graders import domains

        domains.load_all()
    if name not in _grader_registry:
        raise KeyError(
            f"Grader '{name}' not found. Available: {sorted(_grader_registry)}"
        )
    return _grader_registry[name]


def list_graders(grader_type: GraderType | None = None) -> list[str]:
    """List all registered graders.

    Args:
        grader_type: Optional filter by grader type.

    Returns:
        List of grader names.
    """
    # `compass list` is exactly where the full picture is wanted, so this
    # always loads the domains rather than reporting whatever happens to be in.
    from compass.graders import domains

    domains.load_all()

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
