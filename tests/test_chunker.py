from rag.chunker import CharacterDocumentChunker, MarkDownParentChildChunker
from rag.models import Document


def test_character_chunker_basic():
    doc = Document(doc_id="d1", text="a\n\nb\n\nc")
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=3, overlap=0)
    chunks = chunker.chunk(doc)

    texts = [c.text for c in chunks]
    assert texts == ["a", "b", "c"]


def test_character_chunker_split_long_unit_with_overlap():
    doc = Document(doc_id="d1", text="abcdefghij")
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=4, overlap=1)
    chunks = chunker.chunk(doc)

    texts = [c.text for c in chunks]
    assert texts == ["abcd", "defg", "ghij"]


def test_markdown_parent_child_constraints():
    text = "# H1\n\n## H2\n\n- a\n- b\n\n|c1|c2|\n|--|--|\n|x|y|\n\nparagraph line 1\nparagraph line 2\n"
    doc = Document(doc_id="md1", text=text)
    chunker = MarkDownParentChildChunker(parent_max_chars=80, child_max_chars=30)

    result = chunker.chunk_with_hierarchy(doc)

    assert len(result.children) > 0
    for c in result.children:
        assert len(c.text) <= 30

    for p in result.parents:
        assert len(p.text) <= 80
        combined = "".join([c.text for c in result.children if c.chunk_id in p.child_ids])
        assert combined == p.text

    # children are non-overlapping in produced sequence (simple concatenation length check)
    concat_len = sum(len(c.text) for c in result.children)
    assert concat_len == len("".join(c.text for c in result.children))


def test_markdown_parent_child_no_cross_markdown_boundary():
    text = "# 段落1\n\npara-one.\n\n# 段落2\n\npara-two.\n"
    doc = Document(doc_id="md2", text=text)
    chunker = MarkDownParentChildChunker(parent_max_chars=100, child_max_chars=100)

    result = chunker.chunk_with_hierarchy(doc)
    child_texts = [c.text for c in result.children]

    # Must not cross top-level section boundary.
    assert all(not ("para-one." in t and "para-two." in t) for t in child_texts)
    # Parent must also respect the same boundary rule.
    assert all(not ("para-one." in p.text and "para-two." in p.text) for p in result.parents)


def test_markdown_parent_child_merges_small_heading_within_same_h1_section():
    text = "# 概述\n一句话说明。\n"
    doc = Document(doc_id="md3", text=text)
    chunker = MarkDownParentChildChunker(parent_max_chars=2500, child_max_chars=500)
    result = chunker.chunk_with_hierarchy(doc)

    assert len(result.parents) == 1
    assert "# 概述" in result.parents[0].text
    assert "一句话说明。" in result.parents[0].text
