from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from app.core.config import get_settings


@dataclass
class TraceEvent:
    trace_id: str
    span_id: str
    name: str
    latency_ms: float
    attributes: dict[str, Any]


def _build_otel_tracer():
    settings = get_settings()
    if not settings.otel_enabled:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return None

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    exporter_kwargs = {}
    if settings.otel_exporter_otlp_endpoint:
        exporter_kwargs["endpoint"] = settings.otel_exporter_otlp_endpoint
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._otel_tracer = _build_otel_tracer()

    @contextmanager
    def span(self, trace_id: str, name: str, **attributes: Any):
        start = time.perf_counter()
        span_id = f"span-{uuid4().hex[:10]}"
        otel_span_cm = self._otel_tracer.start_as_current_span(name) if self._otel_tracer else None
        otel_span = otel_span_cm.__enter__() if otel_span_cm else None
        if otel_span:
            otel_span.set_attribute("app.trace_id", trace_id)
            otel_span.set_attribute("app.span_id", span_id)
            for key, value in attributes.items():
                if isinstance(value, str | int | float | bool):
                    otel_span.set_attribute(key, value)
        try:
            yield
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            if otel_span:
                otel_span.set_attribute("latency_ms", latency_ms)
                otel_span_cm.__exit__(None, None, None)
            self._events.append(
                TraceEvent(
                    trace_id=trace_id,
                    span_id=span_id,
                    name=name,
                    latency_ms=latency_ms,
                    attributes=attributes,
                )
            )

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return [asdict(event) for event in self._events[-limit:]]


trace_recorder = TraceRecorder()
