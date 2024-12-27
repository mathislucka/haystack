# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import Document
from haystack.lazy_imports import LazyImport

with LazyImport('Please install transformers to use the DocumentTokenTruncater') as tokenizers_import:
    import transformers


class TruncationStrategy(Enum):
    """Strategy to use when truncating documents"""
    BEGINNING = "beginning"      # Truncate from the beginning
    END = "end"                  # Truncate from the end
    EQUAL = "equal"              # Truncate equally from all documents to make them fit


@component
class DocumentTokenTruncater:
    """
    Truncates documents to fit within a specified token limit.

    The component truncates text content in documents to ensure they fit within a maximum token count.
    This is useful when you need to ensure your documents will fit within a model's context window.

    Different truncation strategies are available:
    - BEGINNING: Truncates from the beginning of each document
    - END: Truncates from the end of each document (default)
    - EQUAL: Truncates all documents equally to make them fit within the limit

    ### Usage example
    ```python
    from haystack.components.preprocessors import DocumentTokenTruncater
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    truncater = DocumentTokenTruncater(
        tokenizer=tokenizer,
        max_token_len=1000,
        strategy="end"
    )
    result = truncater.run(
        documents=[Document(content="Very long document..."), Document(content="Another long doc...")]
    )
    ```
    """

    def __init__(
        self,
        tokenizer: Any,
        max_token_len: int,
        strategy: str = "end",
        reserve_token_len: int = 0
    ):
        """
        Initialize the DocumentTokenTruncater.

        :param tokenizer: A tokenizer instance (e.g. from Hugging Face transformers)
        :param max_token_len: Maximum number of tokens allowed across all documents
        :param strategy: Strategy to use for truncation: "beginning", "end", or "equal"
        :param reserve_token_len: Number of tokens to reserve for other content (e.g. prompt instructions)
        """
        tokenizers_import.check()

        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        if not isinstance(strategy, (str, TruncationStrategy)):
            raise ValueError(f"strategy must be one of {[s.value for s in TruncationStrategy]}")
        self.strategy = TruncationStrategy(strategy) if isinstance(strategy, str) else strategy
        self.reserve_token_len = reserve_token_len

    def _get_token_len(self, text: str) -> int:
        """Get the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        if self._get_token_len(text) <= max_tokens:
            return text

        tokens = self.tokenizer.encode(text)
        if self.strategy == TruncationStrategy.END:
            truncated_tokens = tokens[:max_tokens]
        elif self.strategy == TruncationStrategy.BEGINNING:
            truncated_tokens = tokens[-max_tokens:]
        else:
            raise ValueError(f"Invalid truncation strategy for single document: {self.strategy}")

        return self.tokenizer.decode(truncated_tokens)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Truncate the documents to fit within the token limit.

        :param documents: List of documents to truncate
        :returns: Dictionary with truncated documents under the 'documents' key
        """
        if not documents:
            return {"documents": []}

        # Calculate available tokens after reserving space
        available_tokens = self.max_token_len - self.reserve_token_len
        if available_tokens <= 0:
            raise ValueError("No tokens available after reserving tokens")

        # Handle equal truncation strategy separately
        if self.strategy == TruncationStrategy.EQUAL and len(documents) > 0:
            total_tokens = sum(self._get_token_len(doc.content) for doc in documents if doc.content)
            if total_tokens > available_tokens:
                # Distribute tokens equally among documents with content
                docs_with_content = sum(1 for doc in documents if doc.content)
                tokens_per_doc = available_tokens // docs_with_content

                truncated_documents = []
                for doc in documents:
                    if not doc.content:
                        truncated_documents.append(doc)
                        continue

                    new_doc = Document(
                        content=self._truncate_text(doc.content, tokens_per_doc),
                        meta={**doc.meta, "truncated": True}
                    )
                    truncated_documents.append(new_doc)
                return {"documents": truncated_documents}

        # Handle beginning/end truncation
        truncated_documents = []
        current_tokens = 0

        for doc in documents:
            if not doc.content:
                truncated_documents.append(doc)
                continue

            doc_tokens = self._get_token_len(doc.content)
            if current_tokens + doc_tokens <= available_tokens:
                truncated_documents.append(doc)
                current_tokens += doc_tokens
            else:
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens <= 0:
                    break

                new_doc = Document(
                    content=self._truncate_text(doc.content, remaining_tokens),
                    meta={**doc.meta, "truncated": True}
                )
                truncated_documents.append(new_doc)
                break

        return {"documents": truncated_documents}
