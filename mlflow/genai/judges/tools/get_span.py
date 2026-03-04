"""
Get span tool for MLflow GenAI judges.

This module provides a tool for retrieving a specific span by ID from a trace,
with optional attribute filtering and pagination support.
"""

from dataclasses import dataclass

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import (
    FunctionToolDefinition,
    ParamProperty,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.search_utils import SearchUtils

_MAX_CHUNK_SIZE_BYTES = 100_000


@experimental(version="3.4.0")
@dataclass
class SpanResult:
    """Result of a get_span tool invocation.

    Args:
        span_id: The unique identifier of the retrieved span.
        content: JSON string representation of the span data.
        content_size_bytes: Size of the content in bytes.
        page_token: Token for retrieving the next page of results, or None if
            there are no more pages.
        error: Error message if the span retrieval failed, or None on success.
    """

    span_id: str
    content: str
    content_size_bytes: int
    page_token: str | None
    error: str | None


@experimental(version="3.4.0")
class GetSpanTool(JudgeTool):
    """
    Tool for retrieving a specific span by ID from a trace.

    This tool looks up a span within the trace's data by its span ID and returns
    the span content as a JSON string. It supports optional attribute filtering
    to return only specific attributes, and pagination for large span data.
    """

    @property
    def name(self) -> str:
        return ToolNames.GET_SPAN

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_SPAN,
                description=(
                    "Retrieve a specific span from the trace by its span ID. Returns the span "
                    "data as a JSON string including inputs, outputs, attributes, and events. "
                    "Use the 'attributes' parameter to filter and return only specific span "
                    "attributes. For large span data, use the 'page_token' parameter to paginate "
                    "through results. This is useful for inspecting individual steps within a "
                    "trace to understand their behavior, inputs, outputs, and metadata."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "span_id": ParamProperty(
                            type="string",
                            description="The unique identifier of the span to retrieve.",
                        ),
                        "attributes": ParamProperty(
                            type="array",
                            description=(
                                "Optional list of attribute keys to include in the result. "
                                "If not provided, all attributes are returned."
                            ),
                            items=ParamProperty(type="string"),
                        ),
                        "page_token": ParamProperty(
                            type="string",
                            description=(
                                "Optional token for pagination. Use the page_token from a "
                                "previous response to retrieve the next page of results."
                            ),
                        ),
                    },
                    required=["span_id"],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace, **kwargs) -> SpanResult:
        """
        Retrieve a specific span from the trace by ID.

        Args:
            trace: The MLflow trace object to search within.
            kwargs: Additional keyword arguments:
                span_id (str): The ID of the span to retrieve.
                attributes (list[str], optional): Attribute keys to filter by.
                page_token (str, optional): Pagination token for resuming from
                    a byte offset.

        Returns:
            SpanResult containing the span data or an error message.
        """
        import json

        span_id = kwargs.get("span_id")
        attributes = kwargs.get("attributes")
        page_token = kwargs.get("page_token")

        if not span_id:
            return SpanResult(
                span_id="",
                content="",
                content_size_bytes=0,
                page_token=None,
                error="span_id is required",
            )

        # Search for the span in the trace data
        target_span = None
        for span in trace.data.spans:
            if span.span_id == span_id:
                target_span = span
                break

        if target_span is None:
            return SpanResult(
                span_id=span_id,
                content="",
                content_size_bytes=0,
                page_token=None,
                error=f"Span with ID '{span_id}' not found in trace",
            )

        span_dict = target_span.to_dict()

        # Filter attributes if specified
        if attributes is not None and "attributes" in span_dict:
            span_dict["attributes"] = {
                k: v for k, v in span_dict["attributes"].items() if k in attributes
            }

        full_content_bytes = json.dumps(span_dict).encode("utf-8")
        total_size = len(full_content_bytes)

        # Parse byte offset from page_token (0 if no token provided)
        byte_offset = SearchUtils.parse_start_offset_from_page_token(page_token)

        # Slice the content bytes for the current chunk
        chunk_bytes = full_content_bytes[byte_offset : byte_offset + _MAX_CHUNK_SIZE_BYTES]
        chunk_end = byte_offset + len(chunk_bytes)

        # Generate a next page token if there is remaining content
        next_page_token = None
        if chunk_end < total_size:
            token = SearchUtils.create_page_token(chunk_end)
            next_page_token = token.decode("utf-8") if isinstance(token, bytes) else token

        return SpanResult(
            span_id=span_id,
            content=chunk_bytes.decode("utf-8", errors="replace"),
            content_size_bytes=total_size,
            page_token=next_page_token,
            error=None,
        )
