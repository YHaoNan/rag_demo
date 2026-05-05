from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import Document


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> Document:
        raise NotImplementedError


class MarkdownParser(DocumentParser):
    """Read markdown as-is and return content unchanged."""

    def parse(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        return Document(doc_id=path.stem, text=text)
