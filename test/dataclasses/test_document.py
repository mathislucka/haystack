# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

from haystack import Document
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.sparse_embedding import SparseEmbedding


@pytest.mark.parametrize(
    "doc,doc_str",
    [
        (Document(content="test text"), "content: 'test text'"),
        (Document(blob=ByteStream(b"hello, test string")), "blob: 18 bytes"),
        (Document(content="test text", blob=ByteStream(b"hello, test string")), "content: 'test text', blob: 18 bytes"),
    ],
)
def test_document_str(doc, doc_str):
    assert f"Document(id={doc.id}, {doc_str})" == str(doc)


def test_init():
    doc = Document()
    assert doc.id == "9ba126c4e4109657a7f048b7c83496d8c289b391dfe78e6e45372d46116ad7f8"
    assert doc.content == None
    assert doc.blob == None
    assert doc.meta == {}
    assert doc.score == None
    assert doc.embedding == None
    assert doc.sparse_embedding == None


def test_init_with_wrong_parameters():
    with pytest.raises(TypeError):
        Document(text="")


def test_init_with_parameters():
    blob_data = b"some bytes"
    sparse_embedding = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3])
    doc = Document(
        content="test text",
        blob=ByteStream(data=blob_data, mime_type="text/markdown"),
        meta={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=sparse_embedding,
    )
    assert doc.id == "b3b2be3a55c7ddd949d06762cc6a3d0d20e6f0c1e5b2868d65355d3acc2572ae"
    assert doc.content == "test text"
    assert doc.blob.data == blob_data
    assert doc.blob.mime_type == "text/markdown"
    assert doc.meta == {"text": "test text"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.sparse_embedding == sparse_embedding


def test_init_with_legacy_fields():
    doc = Document(
        content="test text",
        content_type="text",
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],  # type: ignore
    )
    assert doc.id == "c342b9b422bde5efaa62fae80dd536aedc2211d45bb3f305fb715d9072c30bab"
    assert doc.content == "test text"
    assert doc.blob == None
    assert doc.meta == {}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.sparse_embedding == None


def test_init_with_legacy_field():
    doc = Document(
        content="test text",
        content_type="text",  # type: ignore
        id_hash_keys=["content"],  # type: ignore
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )
    assert doc.id == "4350de4f35ea989fb0aace39b58c139c56b15e235b66a4589864f4cec8a673e4"
    assert doc.content == "test text"
    assert doc.meta == {"date": "10-10-2023", "type": "article"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.sparse_embedding == None


def test_basic_equality_type_mismatch():
    doc = Document(content="test text")
    assert doc != "test text"


def test_basic_equality_id():
    doc1 = Document(content="test text")
    doc2 = Document(content="test text")

    assert doc1 == doc2

    doc1.id = "1234"
    doc2.id = "5678"

    assert doc1 != doc2


def test_to_dict():
    doc = Document()
    assert doc.to_dict() == {
        "id": doc._create_id(),
        "content": None,
        "blob": None,
        "score": None,
        "embedding": None,
        "sparse_embedding": None,
    }


def test_to_dict_without_flattening():
    doc = Document()
    assert doc.to_dict(flatten=False) == {
        "id": doc._create_id(),
        "content": None,
        "blob": None,
        "meta": {},
        "score": None,
        "embedding": None,
        "sparse_embedding": None,
    }


def test_to_dict_with_custom_parameters():
    doc = Document(
        content="test text",
        blob=ByteStream(b"some bytes", mime_type="application/pdf"),
        meta={"some": "values", "test": 10},
        score=0.99,
        embedding=[10.0, 10.0],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
    )

    assert doc.to_dict() == {
        "id": doc.id,
        "content": "test text",
        "blob": {"data": list(b"some bytes"), "mime_type": "application/pdf"},
        "some": "values",
        "test": 10,
        "score": 0.99,
        "embedding": [10.0, 10.0],
        "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
    }


def test_to_dict_with_custom_parameters_without_flattening():
    doc = Document(
        content="test text",
        blob=ByteStream(b"some bytes", mime_type="application/pdf"),
        meta={"some": "values", "test": 10},
        score=0.99,
        embedding=[10.0, 10.0],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
    )

    assert doc.to_dict(flatten=False) == {
        "id": doc.id,
        "content": "test text",
        "blob": {"data": list(b"some bytes"), "mime_type": "application/pdf"},
        "meta": {"some": "values", "test": 10},
        "score": 0.99,
        "embedding": [10, 10],
        "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
    }


def test_from_dict():
    assert Document.from_dict({}) == Document()


def from_from_dict_with_parameters():
    blob_data = b"some bytes"
    assert Document.from_dict(
        {
            "content": "test text",
            "blob": {"data": list(blob_data), "mime_type": "text/markdown"},
            "meta": {"text": "test text"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
        }
    ) == Document(
        content="test text",
        blob=ByteStream(blob_data, mime_type="text/markdown"),
        meta={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
    )


def test_from_dict_with_legacy_fields():
    assert Document.from_dict(
        {
            "content": "test text",
            "content_type": "text",
            "id_hash_keys": ["content"],
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
        }
    ) == Document(
        content="test text",
        content_type="text",
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],  # type: ignore
    )


def test_from_dict_with_legacy_field_and_flat_meta():
    assert Document.from_dict(
        {
            "content": "test text",
            "content_type": "text",
            "id_hash_keys": ["content"],
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "date": "10-10-2023",
            "type": "article",
        }
    ) == Document(
        content="test text",
        content_type="text",  # type: ignore
        id_hash_keys=["content"],  # type: ignore
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )


def test_from_dict_with_flat_meta():
    blob_data = b"some bytes"
    assert Document.from_dict(
        {
            "content": "test text",
            "blob": {"data": list(blob_data), "mime_type": "text/markdown"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
            "date": "10-10-2023",
            "type": "article",
        }
    ) == Document(
        content="test text",
        blob=ByteStream(blob_data, mime_type="text/markdown"),
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
        meta={"date": "10-10-2023", "type": "article"},
    )


def test_from_dict_with_flat_and_non_flat_meta():
    with pytest.raises(ValueError, match="Pass either the 'meta' parameter or flattened metadata keys"):
        Document.from_dict(
            {
                "content": "test text",
                "blob": {"data": list(b"some bytes"), "mime_type": "text/markdown"},
                "score": 0.812,
                "meta": {"test": 10},
                "embedding": [0.1, 0.2, 0.3],
                "date": "10-10-2023",
                "type": "article",
            }
        )


def test_content_type():
    assert Document(content="text").content_type == "text"

    with pytest.raises(ValueError):
        _ = Document().content_type
