"""Integrations that ingest external agent traces into Compass Transcripts.

These let you evaluate an agent that ran *outside* Compass — you don't write a
Compass adapter, you bridge the agent framework's own trace/telemetry into a
Compass ``Transcript`` and grade it with the normal transcript-scope graders.
"""

from compass.integrations.openai_agents import (
    CompassTraceProcessor,
    install_openai_agents_processor,
)

__all__ = [
    "CompassTraceProcessor",
    "install_openai_agents_processor",
]
