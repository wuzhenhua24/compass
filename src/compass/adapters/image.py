"""Generic image generation adapter."""

from __future__ import annotations

import time
from typing import Any

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter


@register_adapter("image")
class ImageAdapter(Adapter):
    """Generic adapter for image generation agents.

    This is a skeleton adapter that provides the basic structure
    for image generation workflows. Subclass or configure it for
    specific backends (Stable Diffusion, DALL-E, Midjourney, etc.).

    Config options:
        endpoint: API endpoint URL (default "http://localhost:8000").
        timeout: Request timeout in seconds (default 300).
        model: Model identifier (optional).
    """

    name = "image"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.endpoint: str = self.config.get("endpoint", "http://localhost:8000")
        self.timeout: float = self.config.get("timeout", 300)
        self.model: str = self.config.get("model", "")

    async def run(self, input: AgentInput) -> AgentOutput:
        """Run image generation with given input.

        Override this method to integrate with a specific backend.
        """
        start_time = time.time()
        try:
            # Placeholder: subclass should implement actual generation
            raise NotImplementedError(
                "ImageAdapter.run() must be overridden or configured "
                "for a specific image generation backend."
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_tool_call(
                input,
                tool_name="image.generate",
                tool_type="image",
                input={"prompt": input.prompt, "params": input.params},
                status="error",
                duration_ms=duration_ms,
                error={"type": type(e).__name__, "message": str(e)},
                metadata={"endpoint": self.endpoint},
            )
            return AgentOutput(error=str(e))
