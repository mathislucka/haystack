# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from haystack.core.component import component
from haystack.core.pipeline import AsyncPipeline
from haystack import tracing
from haystack.tracing import opentelemetry as otel_tracing
from haystack.lazy_imports import LazyImport
from test.tracing.utils import setup_opentelemetry, get_spans_from_exporter

with LazyImport("Run 'pip install opentelemetry-sdk'") as opentelemetry_import:
    import opentelemetry


@component
class SimpleComponent:
    @component.output_types(output=str)
    def run(self, text: str):
        return {"output": f"processed: {text}"}


@component
class AsyncComponent:
    @component.output_types(output=str)
    async def run_async(self, text: str):
        await asyncio.sleep(0.1)  # Simulate async work
        return {"output": f"async processed: {text}"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_telemetry_client")
async def test_async_pipeline_tracing_context_propagation():
    """Test that OpenTelemetry context is properly propagated from AsyncPipeline to components."""
    # Skip test if OpenTelemetry is not installed
    if "opentelemetry" not in globals():
        pytest.skip("OpenTelemetry not installed")
    
    # Initialize tracing with an in-memory exporter for testing
    exporter = setup_opentelemetry()
    
    # Setup components and pipeline
    comp1 = SimpleComponent()
    comp2 = AsyncComponent()
    
    pipeline = AsyncPipeline()
    pipeline.add_component("comp1", comp1)
    pipeline.add_component("comp2", comp2)
    pipeline.connect("comp1.output", "comp2.text")
    
    # Run the pipeline
    result = await pipeline.run_async({"comp1": {"text": "hello"}})
    
    # Verify the pipeline ran correctly
    assert "comp2" in result
    assert result["comp2"]["output"] == "async processed: processed: hello"
    
    # Get spans from the exporter
    spans = get_spans_from_exporter(exporter)
    
    # Find the pipeline span and component spans
    pipeline_spans = [span for span in spans if span.name == "haystack.async_pipeline.run"]
    comp1_spans = [span for span in spans if span.name == "haystack.component.run" and 
                  span.attributes.get("haystack.component.name") == "comp1"]
    comp2_spans = [span for span in spans if span.name == "haystack.component.run" and 
                  span.attributes.get("haystack.component.name") == "comp2"]
    
    # Verify we have all expected spans
    assert len(pipeline_spans) == 1, "Should have exactly one pipeline span"
    assert len(comp1_spans) == 1, "Should have exactly one span for comp1"
    assert len(comp2_spans) == 1, "Should have exactly one span for comp2"
    
    # Get the pipeline span and component span IDs
    pipeline_span_id = pipeline_spans[0].context.span_id
    pipeline_trace_id = pipeline_spans[0].context.trace_id
    comp1_span = comp1_spans[0]
    comp2_span = comp2_spans[0]
    
    # Verify that component spans have correct parent span
    assert comp1_span.parent.span_id == pipeline_span_id, "comp1 span should have pipeline span as parent"
    assert comp2_span.parent.span_id == pipeline_span_id, "comp2 span should have pipeline span as parent"
    
    # Verify all spans are in the same trace
    assert comp1_span.context.trace_id == pipeline_trace_id, "comp1 should be in the same trace as pipeline"
    assert comp2_span.context.trace_id == pipeline_trace_id, "comp2 should be in the same trace as pipeline"
