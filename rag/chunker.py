from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .models import Chunk, Document


class DocumentChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError


class CharacterDocumentChunker(DocumentChunker):
    def __init__(self, separator: str = "\n\n", max_length: int = 500, overlap: int = 50):
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= max_length:
            raise ValueError("overlap must be < max_length")

        self.separator = separator
        self.max_length = max_length
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        units = document.text.split(self.separator) if self.separator else [document.text]
        chunks: list[Chunk] = []

        current = ""
        idx = 0

        for unit in units:
            candidate = unit if not current else f"{current}{self.separator}{unit}"
            if len(candidate) <= self.max_length:
                current = candidate
                continue

            if current:
                chunks.append(Chunk(doc_id=document.doc_id, chunk_id=f"{document.doc_id}-{idx}", text=current))
                idx += 1

            if len(unit) > self.max_length:
                start = 0
                while start < len(unit):
                    end = min(start + self.max_length, len(unit))
                    part = unit[start:end]
                    chunks.append(Chunk(doc_id=document.doc_id, chunk_id=f"{document.doc_id}-{idx}", text=part))
                    idx += 1
                    if end == len(unit):
                        break
                    start = end - self.overlap
                current = ""
            else:
                current = unit

        if current:
            chunks.append(Chunk(doc_id=document.doc_id, chunk_id=f"{document.doc_id}-{idx}", text=current))

        return chunks


@dataclass
class MarkdownNode:
    node_type: str
    level: int
    start: int
    end: int
    text: str
    children: list["MarkdownNode"] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return self.end - self.start


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    child_ids: list[str]


@dataclass
class ParentChildChunkResult:
    tree: MarkdownNode
    children: list[Chunk]
    parents: list[ParentChunk]


class MarkDownParentChildChunker(DocumentChunker):
    def __init__(self, parent_max_chars: int = 1200, child_max_chars: int = 300):
        if parent_max_chars <= 0 or child_max_chars <= 0:
            raise ValueError("parent_max_chars and child_max_chars must be > 0")
        if parent_max_chars < child_max_chars:
            raise ValueError("parent_max_chars must be >= child_max_chars")
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars

    def chunk(self, document: Document) -> list[Chunk]:
        return self.chunk_with_hierarchy(document).children

    def chunk_with_hierarchy(self, document: Document) -> ParentChildChunkResult:
        tree = self._build_tree(document.text)
        atoms = self._collect_atoms(tree, current_section=-1)
        children, child_boundaries = self._build_children(document.doc_id, document.text, atoms)
        parents = self._build_parents(document.doc_id, children, child_boundaries)
        return ParentChildChunkResult(tree=tree, children=children, parents=parents)

    def _build_tree(self, text: str) -> MarkdownNode:
        root = MarkdownNode(node_type="root", level=0, start=0, end=len(text), text=text)
        stack: list[MarkdownNode] = [root]

        lines = text.splitlines(keepends=True)
        pos = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            start = pos
            end = pos + len(line)

            if line.strip() == "":
                pos = end
                i += 1
                continue

            heading = re.match(r"^(#{1,6})\s+", line)
            if heading:
                level = len(heading.group(1))
                while stack and stack[-1].level >= level:
                    stack.pop()
                node = MarkdownNode("heading", level, start, end, line)
                stack[-1].children.append(node)
                stack.append(node)
                pos = end
                i += 1
                continue

            if self._is_list_line(line):
                b_start = start
                parts = [line]
                i += 1
                pos = end
                while i < len(lines) and (self._is_list_line(lines[i]) or lines[i].strip() == ""):
                    parts.append(lines[i])
                    pos += len(lines[i])
                    i += 1
                node_text = "".join(parts)
                node = MarkdownNode("list", stack[-1].level + 1, b_start, b_start + len(node_text), node_text)
                stack[-1].children.append(node)
                continue

            if "|" in line:
                b_start = start
                parts = [line]
                i += 1
                pos = end
                while i < len(lines) and "|" in lines[i]:
                    parts.append(lines[i])
                    pos += len(lines[i])
                    i += 1
                node_text = "".join(parts)
                node = MarkdownNode("table", stack[-1].level + 1, b_start, b_start + len(node_text), node_text)
                stack[-1].children.append(node)
                continue

            b_start = start
            parts = [line]
            i += 1
            pos = end
            while i < len(lines) and lines[i].strip() != "" and not re.match(r"^(#{1,6})\s+", lines[i]) and not self._is_list_line(lines[i]) and "|" not in lines[i]:
                parts.append(lines[i])
                pos += len(lines[i])
                i += 1
            node_text = "".join(parts)
            node = MarkdownNode("paragraph", stack[-1].level + 1, b_start, b_start + len(node_text), node_text)
            stack[-1].children.append(node)

        return root

    def _is_list_line(self, line: str) -> bool:
        return re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+)", line) is not None

    def _collect_atoms(self, node: MarkdownNode, current_section: int) -> list[tuple[MarkdownNode, int]]:
        if not node.children:
            return [(node, current_section)] if node.node_type != "root" else []
        out: list[tuple[MarkdownNode, int]] = []
        section_id = current_section
        if node.node_type == "heading" and node.level == 1:
            section_id = node.start
        if node.node_type == "heading":
            out.append((MarkdownNode("heading_title", node.level, node.start, node.end, node.text), section_id))
        for child in node.children:
            out.extend(self._collect_atoms(child, section_id))
        return out

    def _split_item(self, start: int, text: str) -> list[tuple[int, int, str]]:
        if len(text) <= self.child_max_chars:
            return [(start, start + len(text), text)]
        parts: list[tuple[int, int, str]] = []
        cur = 0
        while cur < len(text):
            end = min(cur + self.child_max_chars, len(text))
            parts.append((start + cur, start + end, text[cur:end]))
            cur = end
        return parts

    def _build_children(self, doc_id: str, full_text: str, atoms: list[tuple[MarkdownNode, int]]) -> tuple[list[Chunk], list[int]]:
        # item: (section_id, start, end, text, whole_atom)
        items: list[tuple[int, int, int, str, bool]] = []
        for atom, section_id in atoms:
            parts = self._split_item(atom.start, full_text[atom.start : atom.end])
            is_whole = len(parts) == 1
            for start, end, text in parts:
                items.append((section_id, start, end, text, is_whole))

        merged: list[tuple[int, int, int, str]] = []
        for section_id, start, end, text, whole in items:
            # Child cannot cross top-level markdown section boundary (# heading).
            if (
                merged
                and merged[-1][0] == section_id
                and whole
                and len(merged[-1][3]) + len(text) <= self.child_max_chars
            ):
                s_id, p_start, _, p_text = merged[-1]
                merged[-1] = (s_id, p_start, end, p_text + text)
            else:
                merged.append((section_id, start, end, text))

        children: list[Chunk] = []
        child_boundaries: list[int] = []
        for idx, (section_id, _, _, text) in enumerate(merged):
            children.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}-child-{idx}", text=text))
            child_boundaries.append(section_id)
        return children, child_boundaries

    def _build_parents(self, doc_id: str, children: list[Chunk], child_boundaries: list[int]) -> list[ParentChunk]:
        parents: list[ParentChunk] = []
        current_text = ""
        current_ids: list[str] = []
        current_boundary: int | None = None
        idx = 0

        for child, boundary_id in zip(children, child_boundaries):
            # Parent cannot cross top-level markdown section boundary (# heading).
            need_flush = (
                current_text
                and (
                    len(current_text) + len(child.text) > self.parent_max_chars
                    or (current_boundary is not None and boundary_id != current_boundary)
                )
            )
            if need_flush:
                parents.append(ParentChunk(parent_id=f"{doc_id}-parent-{idx}", text=current_text, child_ids=current_ids))
                idx += 1
                current_text = ""
                current_ids = []
                current_boundary = None

            current_text += child.text
            current_ids.append(child.chunk_id)
            current_boundary = boundary_id

        if current_ids:
            parents.append(ParentChunk(parent_id=f"{doc_id}-parent-{idx}", text=current_text, child_ids=current_ids))

        return parents
