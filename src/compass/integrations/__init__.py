"""Integrations that ingest external agent traces into Compass Transcripts.

These let you evaluate an agent that ran *outside* Compass — you don't write a
Compass adapter, you bridge the agent framework's own trace/telemetry into a
Compass ``Transcript`` and grade it with the normal transcript-scope graders.
"""

from compass.integrations.openai_agents import (
    CompassTraceProcessor,
    install_openai_agents_processor,
)
from compass.integrations.otlp import (
    OTLPImportError,
    import_otlp_file,
    load_otlp,
    otlp_to_transcripts,
)
from compass.integrations.pi_sessions import (
    PiSessionError,
    import_pi_session,
    import_pi_sessions,
    load_pi_session,
)

__all__ = [
    "CompassTraceProcessor",
    "install_openai_agents_processor",
    "PiSessionError",
    "import_pi_session",
    "import_pi_sessions",
    "load_pi_session",
    "OTLPImportError",
    "import_otlp_file",
    "load_otlp",
    "otlp_to_transcripts",
]
