"""Integrations that ingest external agent traces into Compass Transcripts.

These let you evaluate an agent that ran *outside* Compass — you don't write a
Compass adapter, you bridge the agent framework's own trace/telemetry into a
Compass ``Transcript`` and grade it with the normal transcript-scope graders.
"""

from compass.integrations.atif import (
    ATIFImportError,
    atif_to_transcript,
    import_atif_dir,
    import_atif_file,
    load_atif,
    looks_like_atif,
)
from compass.integrations.claude_agent import (
    WireReconstructor,
    import_claude_stream_json,
    reconstruct_transcript,
    reconstruct_transcript_from_stream,
    reconstruct_transcript_from_wire,
)
from compass.integrations.codex_exec import (
    CodexStreamError,
    CodexStreamReconstructor,
    import_codex_stream_json,
    load_codex_stream,
    looks_like_codex_stream,
)
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
    PiStreamReconstructor,
    import_pi_session,
    import_pi_sessions,
    import_pi_stream_json,
    load_pi_session,
)

__all__ = [
    "ATIFImportError",
    "atif_to_transcript",
    "import_atif_dir",
    "import_atif_file",
    "load_atif",
    "looks_like_atif",
    "CompassTraceProcessor",
    "install_openai_agents_processor",
    "reconstruct_transcript",
    "reconstruct_transcript_from_stream",
    "reconstruct_transcript_from_wire",
    "WireReconstructor",
    "import_claude_stream_json",
    "CodexStreamError",
    "CodexStreamReconstructor",
    "import_codex_stream_json",
    "load_codex_stream",
    "looks_like_codex_stream",
    "PiSessionError",
    "PiStreamReconstructor",
    "import_pi_session",
    "import_pi_sessions",
    "import_pi_stream_json",
    "load_pi_session",
    "OTLPImportError",
    "import_otlp_file",
    "load_otlp",
    "otlp_to_transcripts",
]
