import json
from unittest.mock import MagicMock

from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.tools.get_span import GetSpanTool, SpanResult, _MAX_CHUNK_SIZE_BYTES
from mlflow.types.llm import ToolDefinition


def _make_mock_span(span_id, span_dict):
    """Create a mock span with the given ID and to_dict() return value."""
    span = MagicMock()
    span.span_id = span_id
    span.to_dict.return_value = span_dict
    return span


def _make_trace(spans):
    """Create a Trace with mock TraceData containing the given spans."""
    trace_info = TraceInfo(
        trace_id="test-trace-id",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=100,
    )
    trace_data = MagicMock(spec=TraceData)
    trace_data.spans = spans
    trace = Trace(info=trace_info, data=trace_data)
    return trace


# --- Tool name and definition tests ---


def test_get_span_tool_name():
    tool = GetSpanTool()
    assert tool.name == "get_span"


def test_get_span_tool_get_definition():
    tool = GetSpanTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.type == "function"
    assert definition.function.name == "get_span"
    assert "span" in definition.function.description
    assert "page_token" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == ["span_id"]

    properties = definition.function.parameters.properties
    assert "span_id" in properties
    assert "attributes" in properties
    assert "page_token" in properties
    assert properties["span_id"].type == "string"
    assert properties["attributes"].type == "array"
    assert properties["page_token"].type == "string"


# --- Successful span retrieval tests ---


def test_invoke_returns_span_result():
    tool = GetSpanTool()
    span_dict = {"span_id": "abc", "name": "test-span", "attributes": {"key": "value"}}
    span = _make_mock_span("abc", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="abc")

    assert isinstance(result, SpanResult)
    assert result.span_id == "abc"
    assert result.error is None
    assert result.page_token is None


def test_invoke_returns_correct_json_content():
    tool = GetSpanTool()
    span_dict = {
        "span_id": "span-1",
        "name": "retriever",
        "attributes": {"model": "gpt-4"},
        "events": [],
    }
    span = _make_mock_span("span-1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="span-1")

    parsed = json.loads(result.content)
    assert parsed == span_dict
    assert result.content_size_bytes == len(json.dumps(span_dict).encode("utf-8"))


def test_invoke_selects_correct_span_from_multiple():
    tool = GetSpanTool()
    span_a = _make_mock_span("span-a", {"span_id": "span-a", "name": "first"})
    span_b = _make_mock_span("span-b", {"span_id": "span-b", "name": "second"})
    span_c = _make_mock_span("span-c", {"span_id": "span-c", "name": "third"})
    trace = _make_trace([span_a, span_b, span_c])

    result = tool.invoke(trace, span_id="span-b")

    assert result.error is None
    parsed = json.loads(result.content)
    assert parsed["name"] == "second"


# --- Error scenario tests ---


def test_invoke_missing_span_id():
    tool = GetSpanTool()
    trace = _make_trace([])

    result = tool.invoke(trace)

    assert isinstance(result, SpanResult)
    assert result.span_id == ""
    assert result.content == ""
    assert result.content_size_bytes == 0
    assert result.page_token is None
    assert result.error == "span_id is required"


def test_invoke_span_not_found():
    tool = GetSpanTool()
    span = _make_mock_span("existing-span", {"span_id": "existing-span"})
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="nonexistent")

    assert result.span_id == "nonexistent"
    assert result.content == ""
    assert result.content_size_bytes == 0
    assert result.page_token is None
    assert "not found" in result.error


def test_invoke_empty_trace():
    tool = GetSpanTool()
    trace = _make_trace([])

    result = tool.invoke(trace, span_id="any-id")

    assert result.error is not None
    assert "not found" in result.error
    assert result.content == ""


# --- Attribute filtering tests ---


def test_invoke_with_attribute_filter():
    tool = GetSpanTool()
    span_dict = {
        "span_id": "s1",
        "name": "llm-call",
        "attributes": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
        },
    }
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="s1", attributes=["model", "temperature"])

    parsed = json.loads(result.content)
    assert parsed["attributes"] == {"model": "gpt-4", "temperature": 0.7}
    assert "max_tokens" not in parsed["attributes"]


def test_invoke_with_empty_attribute_filter():
    tool = GetSpanTool()
    span_dict = {
        "span_id": "s1",
        "attributes": {"key1": "val1", "key2": "val2"},
    }
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="s1", attributes=[])

    parsed = json.loads(result.content)
    assert parsed["attributes"] == {}


def test_invoke_with_nonexistent_attribute_keys():
    tool = GetSpanTool()
    span_dict = {
        "span_id": "s1",
        "attributes": {"real_key": "real_val"},
    }
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="s1", attributes=["nonexistent_key"])

    parsed = json.loads(result.content)
    assert parsed["attributes"] == {}


def test_invoke_without_attribute_filter_returns_all():
    tool = GetSpanTool()
    span_dict = {
        "span_id": "s1",
        "attributes": {"a": 1, "b": 2, "c": 3},
    }
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="s1")

    parsed = json.loads(result.content)
    assert parsed["attributes"] == {"a": 1, "b": 2, "c": 3}


def test_invoke_filter_on_span_without_attributes_key():
    tool = GetSpanTool()
    span_dict = {"span_id": "s1", "name": "no-attrs"}
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="s1", attributes=["anything"])

    parsed = json.loads(result.content)
    assert "attributes" not in parsed
    assert result.error is None


# --- Pagination tests ---


def test_invoke_small_content_no_pagination():
    tool = GetSpanTool()
    span_dict = {"span_id": "s1", "name": "small"}
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="s1")

    assert result.page_token is None
    assert result.content_size_bytes == len(json.dumps(span_dict).encode("utf-8"))
    assert result.content_size_bytes < _MAX_CHUNK_SIZE_BYTES


def test_invoke_large_content_returns_first_chunk():
    tool = GetSpanTool()
    # Build content that exceeds _MAX_CHUNK_SIZE_BYTES
    large_value = "x" * (_MAX_CHUNK_SIZE_BYTES + 50_000)
    span_dict = {"span_id": "big", "data": large_value}
    span = _make_mock_span("big", span_dict)
    trace = _make_trace([span])

    result = tool.invoke(trace, span_id="big")

    full_size = len(json.dumps(span_dict).encode("utf-8"))
    assert result.content_size_bytes == full_size
    assert len(result.content.encode("utf-8")) == _MAX_CHUNK_SIZE_BYTES
    assert result.page_token is not None
    assert result.error is None


def test_invoke_pagination_second_page():
    tool = GetSpanTool()
    large_value = "y" * (_MAX_CHUNK_SIZE_BYTES + 50_000)
    span_dict = {"span_id": "big", "data": large_value}
    span = _make_mock_span("big", span_dict)
    trace = _make_trace([span])

    first_result = tool.invoke(trace, span_id="big")
    assert first_result.page_token is not None

    second_result = tool.invoke(trace, span_id="big", page_token=first_result.page_token)

    assert second_result.span_id == "big"
    assert second_result.content_size_bytes == first_result.content_size_bytes
    assert second_result.error is None
    assert len(second_result.content.encode("utf-8")) > 0


def test_invoke_pagination_reconstructs_full_content():
    tool = GetSpanTool()
    large_value = "z" * (_MAX_CHUNK_SIZE_BYTES * 3)
    span_dict = {"span_id": "huge", "data": large_value}
    span = _make_mock_span("huge", span_dict)
    trace = _make_trace([span])

    full_json = json.dumps(span_dict)
    full_bytes = full_json.encode("utf-8")

    # Collect all chunks
    chunks = []
    page_token = None
    while True:
        result = tool.invoke(trace, span_id="huge", page_token=page_token)
        assert result.error is None
        assert result.content_size_bytes == len(full_bytes)
        chunks.append(result.content.encode("utf-8"))
        if result.page_token is None:
            break
        page_token = result.page_token

    reconstructed = b"".join(chunks)
    assert reconstructed == full_bytes


def test_invoke_pagination_last_page_has_no_token():
    tool = GetSpanTool()
    large_value = "w" * (_MAX_CHUNK_SIZE_BYTES + 100)
    span_dict = {"span_id": "s1", "data": large_value}
    span = _make_mock_span("s1", span_dict)
    trace = _make_trace([span])

    # Walk to the final page
    page_token = None
    last_result = None
    while True:
        result = tool.invoke(trace, span_id="s1", page_token=page_token)
        last_result = result
        if result.page_token is None:
            break
        page_token = result.page_token

    assert last_result.page_token is None
    assert last_result.error is None
    assert len(last_result.content) > 0


def test_invoke_content_exactly_at_chunk_boundary():
    tool = GetSpanTool()
    # Build content whose JSON-encoded size is exactly _MAX_CHUNK_SIZE_BYTES.
    # json.dumps({"d": "..."}) adds overhead for {"d": ""}  which is 8 bytes.
    overhead = len(json.dumps({"d": ""}).encode("utf-8"))
    filler = "a" * (_MAX_CHUNK_SIZE_BYTES - overhead)
    span_dict = {"d": filler}
    span = _make_mock_span("exact", span_dict)
    trace = _make_trace([span])

    # Verify our setup: total size should be exactly at the boundary
    assert len(json.dumps(span_dict).encode("utf-8")) == _MAX_CHUNK_SIZE_BYTES

    result = tool.invoke(trace, span_id="exact")

    assert result.page_token is None
    assert result.content_size_bytes == _MAX_CHUNK_SIZE_BYTES
    assert json.loads(result.content) == span_dict
