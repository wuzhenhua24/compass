"""Turn titles into URL slugs."""


def slugify(text: str) -> str:
    """Lowercase ``text`` and replace non-alphanumeric runs with hyphens.

    Known defect: each non-alphanumeric character becomes its own hyphen, so
    ``"Hello,  World!"`` comes out as ``"hello---world"``.
    """
    out = ""
    for char in text.lower().strip():
        out += char if char.isalnum() else "-"
    return out.strip("-")
