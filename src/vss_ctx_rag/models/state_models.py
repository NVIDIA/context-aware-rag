from typing import Any, TypedDict


class Metadata(TypedDict, total=False):
    """
    Metadata for documents and chunks in the RAG system.

    .. code-block:: python

        metadata = Metadata(
            asset_dirs=["path/to/assets"],
            length=1024,
            streamId="stream123",
            file="document.txt",
            start_pts=100,
            end_pts=200,
            is_first=True,
            is_last=False,
            uuid="unique-id-123",
            linked_summary_chunks="chunk1,chunk2"
        )
    """

    asset_dirs: list[str]
    length: int
    streamId: str
    file: str
    start_pts: int
    end_pts: int
    is_first: bool
    is_last: bool
    uuid: str
    linked_summary_chunks: str


class SourceDocs(TypedDict, total=False):
    """
    Source docs for the retriever function.

    .. code-block:: python

        doc = SourceDocs(
            page_content="Hello world",
            metadata={"source": "file.txt"}
        )
    """

    page_content: str
    metadata: dict[str, Any]


class RetrieverFunctionState(TypedDict, total=False):
    """
    State of the retriever function used for both input and output for RAG types: vector-rag, graph-rag, foundation-rag

    .. code-block:: python

        state = {
            "question": "What is the capital of France?",
            "response_method": "text",
            "response_schema": None,
            "response": "Paris",
            "error": None,
            "source_docs": [
                {
                    "page_content": "Paris is the capital and most populous city of France.",
                    "metadata": {"source": "geography.txt", "page": 1}
                }
            ]
        }
    """

    question: str
    response_method: str | None
    response_schema: dict[str, Any] | None
    response: str | dict | None
    error: str | None
    source_docs: list[SourceDocs] | None
    formatted_docs: list[str] | None
