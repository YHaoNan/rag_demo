from rag.vector_store import SQLiteVectorStore


def test_sqlite_vector_store_query(tmp_path):
    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        d1 = store.create_document("a.md", 1, "CharacterDocumentChunker", {"separator": "\\n\\n", "max_length": 10, "overlap": 0})
        store.add_many(
            d1,
            [
                ("c1", "alpha", [1.0, 0.0]),
                ("c2", "beta", [0.0, 1.0]),
                ("c3", "gamma", [0.9, 0.1]),
            ],
        )

        out = store.query([1.0, 0.0], top_k=2)

    assert len(out) == 2
    assert out[0]["chunk_id"] in {"c1", "c3"}
    assert out[0]["score"] >= out[1]["score"]


def test_sqlite_vector_store_query_with_document_filter(tmp_path):
    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        d1 = store.create_document("a.md", 1, "CharacterDocumentChunker", {})
        d2 = store.create_document("b.md", 1, "CharacterDocumentChunker", {})
        store.add_many(d1, [("c1", "alpha", [1.0, 0.0])])
        store.add_many(d2, [("c2", "beta", [1.0, 0.0])])

        out = store.query([1.0, 0.0], top_k=10, document_ids=[d2])

    assert len(out) == 1
    assert out[0]["document_id"] == d2
