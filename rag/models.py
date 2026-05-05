from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
