from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor


def init_tracer(service_name: str = "ai-perf-portfolio", use_console: bool = True, otlp_endpoint: Optional[str] = None):
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if use_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


@contextmanager
def traced_span(name: str, tracer=None):
    tracer = tracer or trace.get_tracer("ai-perf-portfolio")
    with tracer.start_as_current_span(name) as span:
        yield span
