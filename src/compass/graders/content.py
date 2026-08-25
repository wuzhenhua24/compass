"""Getting the thing to be graded out of a :class:`GradeContext`.

A grader is handed the whole context and has to find its subject in it — the
text artifact, a field of ``output_data``, a file in the code artifact. Three
graders in the framework and any number in ``domains/`` ask the same question,
so the answer lives here rather than inside whichever grader happened to need
it first.
"""

from __future__ import annotations

from typing import Any

from compass.graders.base import GradeContext


def extract_content(context: GradeContext, source: str) -> str | dict[str, Any] | None:
    """Extract content from context based on source specification.

    Args:
        context: Grade context.
        source: Source specification:
            - "output_data" or "output": from outcome.output_data
            - "text" or "text_artifact": from TextArtifact.content
            - "code" or "code_artifact": from CodeArtifact files
            - "metadata.<key>": from outcome.metadata
            - "output_data.<key>": from specific field in output_data

    Returns:
        Extracted content as string or dict.
    """
    if not context.outcome:
        return None

    if source in ("output_data", "output"):
        return context.outcome.output_data

    if source in ("text", "text_artifact"):
        text = context.text_artifact
        if text and hasattr(text, "content"):
            return text.content
        return None

    if source in ("code", "code_artifact"):
        code = context.code_artifact
        if code and hasattr(code, "files"):
            # Return concatenated file contents
            contents = []
            for f in code.files:
                if hasattr(f, "content"):
                    contents.append(f.content)
            return "\n".join(contents)
        return None

    if source.startswith("metadata."):
        key = source[9:]
        return context.outcome.metadata.get(key)

    if source.startswith("output_data."):
        key = source[12:]
        data = context.outcome.output_data
        if isinstance(data, dict):
            return data.get(key)

    return None
