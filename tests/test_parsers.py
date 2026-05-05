from pathlib import Path

from rag.parsers import MarkdownParser


def test_markdown_parser_returns_original_text(tmp_path: Path):
    p = tmp_path / "a.md"
    content = "# title\n\nhello"
    p.write_text(content, encoding="utf-8")

    parser = MarkdownParser()
    doc = parser.parse(p)

    assert doc.doc_id == "a"
    assert doc.text == content
