from rag.contextualizer import ChunkContextualizer


def test_extract_context_snippet_around_chunk():
    full_text = "A" * 100 + "TARGET" + "B" * 100
    out = ChunkContextualizer._extract_context_snippet(full_text, "TARGET", 50)
    assert len(out) == 50
    assert "TARGET" in out


def test_extract_context_snippet_fallback_when_chunk_not_found():
    full_text = "X" * 200
    out = ChunkContextualizer._extract_context_snippet(full_text, "NOT_FOUND", 80)
    assert out == full_text[:80]
