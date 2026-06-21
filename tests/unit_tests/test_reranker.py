"""Tests for the reranker utility module."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from agent.utils.reranker import get_reranker, rerank_with_flashrank


def test_get_reranker_none():
    """Test get_reranker with 'none' provider returns passthrough."""
    reranker = get_reranker(provider="none", top_k=3)
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    result = reranker(docs, "query")
    assert len(result) == 3
    assert result[0].page_content == "doc0"


def test_get_reranker_none_fewer_docs():
    """Test passthrough doesn't truncate when fewer docs than top_k."""
    reranker = get_reranker(provider="none", top_k=5)
    docs = [Document(page_content=f"doc{i}") for i in range(2)]
    result = reranker(docs, "query")
    assert len(result) == 2


def test_get_reranker_invalid_provider():
    """Test get_reranker raises error for unknown provider."""
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        get_reranker(provider="invalid", top_k=3)


@patch("flashrank.Ranker")
@patch("flashrank.RerankRequest")
def test_rerank_with_flashrank(mock_rerank_request, mock_ranker):
    """Test rerank_with_flashrank calls FlashRank correctly."""
    mock_ranker_instance = MagicMock()
    mock_ranker.return_value = mock_ranker_instance

    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    # Simulate flashrank returning results with scores
    mock_ranker_instance.rerank.return_value = [
        {"id": 1, "score": 0.9, "text": "doc2"},
        {"id": 0, "score": 0.5, "text": "doc1"},
    ]

    result = rerank_with_flashrank(docs, "query", top_k=2)

    mock_ranker.assert_called_once()
    assert len(result) == 2
    # Higher score doc should be first
    assert result[0].page_content == "doc2"
    assert result[1].page_content == "doc1"


def test_rerank_with_flashrank_empty_docs():
    """Test rerank_with_flashrank handles empty documents."""
    result = rerank_with_flashrank([], "query", top_k=3)
    assert result == []
