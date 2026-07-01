"""Trace collection for observability."""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator


@dataclass
class Span:
    """A single span in a trace."""

    name: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "running"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def end(self, status: str = "ok") -> None:
        """End the span."""
        self.end_time = time.time()
        self.status = status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Trace:
    """A complete trace containing multiple spans."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_span(self, span: Span) -> None:
        """Add a span to the trace."""
        self.spans.append(span)

    @property
    def root_span(self) -> Span | None:
        """Get the root span."""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def total_duration_ms(self) -> float | None:
        """Get total trace duration."""
        root = self.root_span
        return root.duration_ms if root else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "total_duration_ms": self.total_duration_ms,
        }


class TraceCollector:
    """Collects and manages traces."""

    def __init__(self):
        self._traces: dict[str, Trace] = {}
        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

    @contextmanager
    def trace(self, name: str) -> Generator[Trace, None, None]:
        """Context manager for creating a trace.

        Args:
            name: Name of the trace.

        Yields:
            The trace object.
        """
        trace = Trace(name=name)
        self._current_trace = trace

        try:
            yield trace
        finally:
            self._traces[trace.trace_id] = trace
            self._current_trace = None

    @contextmanager
    def span(self, name: str) -> Generator[Span, None, None]:
        """Context manager for creating a span.

        Args:
            name: Name of the span.

        Yields:
            The span object.
        """
        parent_id = self._span_stack[-1].span_id if self._span_stack else None
        span = Span(name=name, parent_id=parent_id)

        self._span_stack.append(span)
        if self._current_trace:
            self._current_trace.add_span(span)

        try:
            yield span
        except Exception as e:
            span.end(status="error")
            span.set_attribute("error", str(e))
            raise
        else:
            span.end(status="ok")
        finally:
            self._span_stack.pop()

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def list_traces(self) -> list[Trace]:
        """List all traces."""
        return list(self._traces.values())

    def clear(self) -> None:
        """Clear all traces."""
        self._traces.clear()

    def export_json(self) -> list[dict[str, Any]]:
        """Export all traces as JSON-serializable dicts."""
        return [t.to_dict() for t in self._traces.values()]


# Global collector instance
_collector = TraceCollector()


def get_collector() -> TraceCollector:
    """Get the global trace collector."""
    return _collector
