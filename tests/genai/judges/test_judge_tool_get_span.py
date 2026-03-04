import json
from unittest.mock import MagicMock

import pytest

from mlflow.genai.judges.tools.get_span import GetSpanTool, SpanResult, _MAX_CHUNK_SIZE_BYTES
from mlflow.types.llm import ToolDefinition


def _make_mock_span(span_id, span_dict):
    """Create a mock span with the given ID and to_dict() return value."""
    span = MagicMock()
    span.span_id = span_id
    span.to_dict.return_value = span_dict
    return span


def _make_mock_trace(spans):
    """Create a mock trace containing the given list of mock spans."""
    trace = MagicMock()
    trace.data.spans = spans
    return trace


# --- Tool metadata tests ---


def test_get_span_tool_name():
    tool = GetSpanTool()
    assert tool.name == "get_span"


def test_get_span_tool_get_definition():
    tool = GetSpanTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.type == "function"
    assert definition.function.name == "get_span"
    assert "span" in definition.function.description.lower()
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == ["span_id"]
    assert "span_id" in definition.function.parameters.properties
    assert "attributes" in definition.function.parameters.properties
    assert "page_token" in definition.function.parameters.properties


# --- Successful retrieval tests ---


def test_invoke_returns_span_result():
    tool = GetSpanTool()
    span_dict = {"name": "llm_call", "attributes": {"model": "gpt-4"}}
    trace = _make_mock_trace([_make_mock_span("span-1", span_dict)])

    result = tool.invoke(trace, span_id="span-1")

    assert isinstance(result, SpanResult)
    assert result.span_id == "span-1"
    assert result.error is None


def test_invoke_content_is_valid_json():
    tool = GetSpanTool()
    span_dict = {"name": "retrieve", "inputs": {"query": "hello"}, "outputs": [1, 2, 3]}
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1")
    parsed = json.loads(result.content)

    assert parsed == span_dict


def test_invoke_content_size_bytes_matches():
    tool = GetSpanTool()
    span_dict = {"name": "op", "value": 42}
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1")
    expected_size = len(json.dumps(span_dict).encode("utf-8"))

    assert result.content_size_bytes == expected_size


def test_invoke_small_content_no_pagination():
    tool = GetSpanTool()
    span_dict = {"name": "small_span"}
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1")

    assert result.page_token is None


def test_invoke_selects_correct_span_among_multiple():
    tool = GetSpanTool()
    span_a = _make_mock_span("a", {"name": "span_a"})
    span_b = _make_mock_span("b", {"name": "span_b"})
    span_c = _make_mock_span("c", {"name": "span_c"})
    trace = _make_mock_trace([span_a, span_b, span_c])

    result = tool.invoke(trace, span_id="b")

    assert result.span_id == "b"
    assert json.loads(result.content)["name"] == "span_b"
    assert result.error is None


# --- Error scenario tests ---


def test_invoke_missing_span_id():
    tool = GetSpanTool()
    trace = _make_mock_trace([])

    result = tool.invoke(trace)

    assert result.error == "span_id is required"
    assert result.span_id == ""
    assert result.content == ""
    assert result.content_size_bytes == 0
    assert result.page_token is None


def test_invoke_span_not_found():
    tool = GetSpanTool()
    trace = _make_mock_trace([_make_mock_span("existing", {"name": "exists"})])

    result = tool.invoke(trace, span_id="nonexistent")

    assert result.error == "Span with ID 'nonexistent' not found in trace"
    assert result.span_id == "nonexistent"
    assert result.content == ""
    assert result.content_size_bytes == 0
    assert result.page_token is None


def test_invoke_empty_trace():
    tool = GetSpanTool()
    trace = _make_mock_trace([])

    result = tool.invoke(trace, span_id="any-id")

    assert result.error == "Span with ID 'any-id' not found in trace"
    assert result.content == ""


# --- Attribute filtering tests ---


def test_invoke_attribute_filtering():
    tool = GetSpanTool()
    span_dict = {
        "name": "llm",
        "attributes": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100},
    }
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1", attributes=["model", "temperature"])
    parsed = json.loads(result.content)

    assert parsed["attributes"] == {"model": "gpt-4", "temperature": 0.7}
    assert "max_tokens" not in parsed["attributes"]


def test_invoke_attribute_filtering_preserves_non_attribute_fields():
    tool = GetSpanTool()
    span_dict = {
        "name": "op",
        "inputs": {"x": 1},
        "attributes": {"key1": "val1", "key2": "val2"},
    }
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1", attributes=["key1"])
    parsed = json.loads(result.content)

    assert parsed["name"] == "op"
    assert parsed["inputs"] == {"x": 1}
    assert parsed["attributes"] == {"key1": "val1"}


def test_invoke_attribute_filtering_no_matching_keys():
    tool = GetSpanTool()
    span_dict = {"name": "op", "attributes": {"a": 1, "b": 2}}
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1", attributes=["nonexistent"])
    parsed = json.loads(result.content)

    assert parsed["attributes"] == {}


def test_invoke_attribute_filtering_span_without_attributes_key():
    tool = GetSpanTool()
    span_dict = {"name": "op", "inputs": {"q": "hi"}}
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1", attributes=["anything"])
    parsed = json.loads(result.content)

    # attributes filter should be a no-op when span has no "attributes" key
    assert "attributes" not in parsed
    assert parsed["name"] == "op"


def test_invoke_empty_attributes_list_returns_no_attributes():
    tool = GetSpanTool()
    span_dict = {"name": "op", "attributes": {"a": 1, "b": 2}}
    trace = _make_mock_trace([_make_mock_span("s1", span_dict)])

    result = tool.invoke(trace, span_id="s1", attributes=[])
    parsed = json.loads(result.content)

    assert parsed["attributes"] == {}


# --- Pagination tests ---


def _make_large_span(span_id, size_bytes):
    """Create a mock span whose JSON serialization exceeds the given byte size."""
    # Account for JSON overhead of {"data": "..."}
    overhead = len(json.dumps({"data": ""}).encode("utf-8"))
    padding = "x" * (size_bytes - overhead)
    span_dict = {"data": padding}
    return _make_mock_span(span_id, span_dict), span_dict


def test_invoke_pagination_first_page():
    tool = GetSpanTool()
    target_size = _MAX_CHUNK_SIZE_BYTES * 2 + 500
    mock_span, _ = _make_large_span("big", target_size)
    trace = _make_mock_trace([mock_span])

    result = tool.invoke(trace, span_id="big")

    assert result.error is None
    assert result.page_token is not None
    assert len(result.content.encode("utf-8")) == _MAX_CHUNK_SIZE_BYTES
    assert result.content_size_bytes == target_size


def test_invoke_pagination_second_page():
    tool = GetSpanTool()
    target_size = _MAX_CHUNK_SIZE_BYTES * 2 + 500
    mock_span, _ = _make_large_span("big", target_size)
    trace = _make_mock_trace([mock_span])

    first = tool.invoke(trace, span_id="big")
    second = tool.invoke(trace, span_id="big", page_token=first.page_token)

    assert second.error is None
    assert second.page_token is not None
    assert len(second.content.encode("utf-8")) == _MAX_CHUNK_SIZE_BYTES
    assert second.content_size_bytes == target_size


def test_invoke_pagination_last_page_has_no_token():
    tool = GetSpanTool()
    target_size = _MAX_CHUNK_SIZE_BYTES + 100
    mock_span, _ = _make_large_span("big", target_size)
    trace = _make_mock_trace([mock_span])

    first = tool.invoke(trace, span_id="big")
    second = tool.invoke(trace, span_id="big", page_token=first.page_token)

    assert second.page_token is None
    assert second.error is None


def test_invoke_pagination_full_reconstruction():
    tool = GetSpanTool()
    target_size = _MAX_CHUNK_SIZE_BYTES * 3 + 7
    mock_span, span_dict = _make_large_span("big", target_size)
    trace = _make_mock_trace([mock_span])

    chunks = []
    token = None
    while True:
        result = tool.invoke(trace, span_id="big", page_token=token)
        assert result.error is None
        chunks.append(result.content.encode("utf-8"))
        if result.page_token is None:
            break
        token = result.page_token

    reconstructed = b"".join(chunks)
    expected = json.dumps(span_dict).encode("utf-8")

    assert reconstructed == expected
    assert len(chunks) == 4  # 3 full chunks + 1 partial


def test_invoke_pagination_content_size_bytes_consistent_across_pages():
    tool = GetSpanTool()
    target_size = _MAX_CHUNK_SIZE_BYTES * 2 + 1
    mock_span, _ = _make_large_span("big", target_size)
    trace = _make_mock_trace([mock_span])

    sizes = []
    token = None
    while True:
        result = tool.invoke(trace, span_id="big", page_token=token)
        sizes.append(result.content_size_bytes)
        if result.page_token is None:
            break
        token = result.page_token

    # content_size_bytes should report the total size on every page
    assert all(s == target_size for s in sizes)


def test_invoke_exactly_at_chunk_boundary():
    tool = GetSpanTool()
    mock_span, _ = _make_large_span("exact", _MAX_CHUNK_SIZE_BYTES)
    trace = _make_mock_trace([mock_span])

    result = tool.invoke(trace, span_id="exact")

    # Content fits exactly in one chunk, so no pagination needed
    assert result.page_token is None
    assert result.error is None
    assert result.content_size_bytes == _MAX_CHUNK_SIZE_BYTES


def test_invoke_page_token_is_string():
    tool = GetSpanTool()
    target_size = _MAX_CHUNK_SIZE_BYTES + 1
    mock_span, _ = _make_large_span("big", target_size)
    trace = _make_mock_trace([mock_span])

    result = tool.invoke(trace, span_id="big")

    assert isinstance(result.page_token, str)
