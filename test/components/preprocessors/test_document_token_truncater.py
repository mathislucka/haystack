from typing import List

import pytest

from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentTokenTruncater


class MockTokenizer:
    def encode(self, text: str) -> List[int]:
        # Simple mock that treats each word as a token
        return text.split()

    def decode(self, tokens: List[int]) -> str:
        # Join tokens back into text
        return " ".join(tokens)


@pytest.fixture
def tokenizer():
    return MockTokenizer()


def test_init_with_invalid_strategy():
    with pytest.raises(ValueError):
        DocumentTokenTruncater(tokenizer=MockTokenizer(), max_token_len=100, strategy="invalid")


def test_empty_documents(tokenizer):
    truncater = DocumentTokenTruncater(tokenizer=tokenizer, max_token_len=10)
    result = truncater.run(documents=[])
    assert len(result["documents"]) == 0


def test_documents_within_limit(tokenizer):
    truncater = DocumentTokenTruncater(tokenizer=tokenizer, max_token_len=10)
    docs = [
        Document(content="one two three"),
        Document(content="four five")
    ]
    result = truncater.run(documents=docs)
    assert len(result["documents"]) == 2
    assert result["documents"][0].content == "one two three"
    assert result["documents"][1].content == "four five"


def test_truncate_end_strategy(tokenizer):
    truncater = DocumentTokenTruncater(tokenizer=tokenizer, max_token_len=5, strategy="end")
    docs = [
        Document(content="one two three four five"),
        Document(content="six seven eight")
    ]
    result = truncater.run(documents=docs)
    assert len(result["documents"]) == 2
    assert result["documents"][0].content == "one two three four five"
    assert "truncated" not in result["documents"][0].meta
    # Second document doesn't fit and is dropped
    assert result["documents"][1].meta.get("truncated") == True


def test_truncate_beginning_strategy(tokenizer):
    truncater = DocumentTokenTruncater(tokenizer=tokenizer, max_token_len=4, strategy="beginning")
    docs = [
        Document(content="one two three four five six")
    ]
    result = truncater.run(documents=docs)
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "three four five six"
    assert result["documents"][0].meta.get("truncated") == True


def test_truncate_equal_strategy(tokenizer):
    truncater = DocumentTokenTruncater(tokenizer=tokenizer, max_token_len=6, strategy="equal")
    docs = [
        Document(content="one two three four"),
        Document(content="five six seven eight")
    ]
    result = truncater.run(documents=docs)
    assert len(result["documents"]) == 2
    # With 6 tokens available, each document gets 3 tokens
    assert result["documents"][0].content == "one two three"
    assert result["documents"][1].content == "five six seven"
    assert result["documents"][0].meta.get("truncated") == True
    assert result["documents"][1].meta.get("truncated") == True


def test_reserve_tokens(tokenizer):
    truncater = DocumentTokenTruncater(
        tokenizer=tokenizer,
        max_token_len=10,
        reserve_token_len=5,
        strategy="end"
    )
    docs = [
        Document(content="one two three four five"),
        Document(content="six seven eight")
    ]
    result = truncater.run(documents=docs)
    # Only 5 tokens available after reserving 5
    assert len(result["documents"]) == 2
    assert result["documents"][0].content == "one two three four five"
    assert result["documents"][1].meta.get("truncated") == True


def test_documents_without_content(tokenizer):
    truncater = DocumentTokenTruncater(tokenizer=tokenizer, max_token_len=5)
    docs = [
        Document(content=None),
        Document(content="one two three"),
        Document(content=None)
    ]
    result = truncater.run(documents=docs)
    assert len(result["documents"]) == 3
    assert result["documents"][0].content is None
    assert result["documents"][1].content == "one two three"
    assert result["documents"][2].content is None


def test_zero_available_tokens(tokenizer):
    with pytest.raises(ValueError):
        truncater = DocumentTokenTruncater(
            tokenizer=tokenizer,
            max_token_len=5,
            reserve_token_len=6
        )
        truncater.run(documents=[Document(content="test")])