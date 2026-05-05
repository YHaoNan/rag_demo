from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np


class SQLiteVectorStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                chunker TEXT NOT NULL,
                chunker_params TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents_v2(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_chunks_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                parent_id TEXT NOT NULL,
                text TEXT NOT NULL,
                child_chunk_ids TEXT NOT NULL,
                position INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents_v2(id)
            )
            """
        )
        self.conn.commit()

    def create_document(self, source_name: str, chunk_count: int, chunker: str, chunker_params: dict) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO documents_v2(source_name, chunk_count, chunker, chunker_params, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_name, chunk_count, chunker, json.dumps(chunker_params), created_at),
        )
        return int(cur.lastrowid)

    def delete_document(self, document_id: int) -> None:
        self.conn.execute("DELETE FROM parent_chunks_v2 WHERE document_id = ?", (document_id,))
        self.conn.execute("DELETE FROM vectors_v2 WHERE document_id = ?", (document_id,))
        self.conn.execute("DELETE FROM documents_v2 WHERE id = ?", (document_id,))
        self.conn.commit()

    def list_documents(self) -> list[dict]:
        cur = self.conn.execute(
            """
            SELECT id, source_name, chunk_count, chunker, chunker_params, created_at
            FROM documents_v2
            ORDER BY id DESC
            """
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "source_name": row[1],
                "chunk_count": row[2],
                "chunker": row[3],
                "chunker_params": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
            }
            for row in rows
        ]

    def get_document(self, document_id: int) -> dict | None:
        cur = self.conn.execute(
            "SELECT id, source_name, chunk_count, chunker, chunker_params, created_at FROM documents_v2 WHERE id = ?",
            (document_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "source_name": row[1],
            "chunk_count": row[2],
            "chunker": row[3],
            "chunker_params": json.loads(row[4]) if row[4] else {},
            "created_at": row[5],
        }

    def list_chunks(self, document_id: int) -> list[dict]:
        cur = self.conn.execute(
            "SELECT chunk_id, text FROM vectors_v2 WHERE document_id = ? ORDER BY id ASC",
            (document_id,),
        )
        rows = cur.fetchall()
        return [{"chunk_id": row[0], "text": row[1]} for row in rows]

    def add_parent_chunks(self, document_id: int, rows: list[tuple[str, str, list[str], int]]) -> None:
        for parent_id, text, child_chunk_ids, position in rows:
            self.conn.execute(
                """
                INSERT INTO parent_chunks_v2(document_id, parent_id, text, child_chunk_ids, position)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, parent_id, text, json.dumps(child_chunk_ids), position),
            )
        self.conn.commit()

    def list_parent_chunks(self, document_id: int) -> list[dict]:
        cur = self.conn.execute(
            """
            SELECT parent_id, text, child_chunk_ids, position
            FROM parent_chunks_v2
            WHERE document_id = ?
            ORDER BY position ASC
            """,
            (document_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "parent_id": row[0],
                "text": row[1],
                "child_chunk_ids": json.loads(row[2]) if row[2] else [],
                "position": row[3],
            }
            for row in rows
        ]

    def add_many(self, document_id: int, rows: list[tuple[str, str, list[float]]]) -> None:
        for chunk_id, text, embedding in rows:
            self.conn.execute(
                """
                INSERT INTO vectors_v2(document_id, chunk_id, text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, chunk_id, text, json.dumps(embedding)),
            )
        self.conn.commit()

    def query(self, query_embedding: list[float], top_k: int = 3, document_ids: list[int] | None = None) -> list[dict]:
        sql = (
            "SELECT v.document_id, d.source_name, v.chunk_id, v.text, v.embedding "
            "FROM vectors_v2 v JOIN documents_v2 d ON d.id = v.document_id"
        )
        params: list = []
        if document_ids:
            placeholders = ",".join(["?"] * len(document_ids))
            sql += f" WHERE v.document_id IN ({placeholders})"
            params.extend(document_ids)

        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scored = []
        for document_id, source_name, chunk_id, text, emb_json in rows:
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            denom = q_norm * np.linalg.norm(emb)
            score = float(np.dot(q, emb) / denom) if denom != 0 else 0.0
            scored.append(
                {
                    "document_id": document_id,
                    "source_name": source_name,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def keyword_search(self, query: str, top_k: int = 20, document_ids: list[int] | None = None) -> list[dict]:
        token = query.strip()
        if not token:
            return []

        sql = (
            "SELECT v.document_id, d.source_name, v.chunk_id, v.text "
            "FROM vectors_v2 v JOIN documents_v2 d ON d.id = v.document_id "
            "WHERE LOWER(v.text) LIKE ?"
        )
        params: list = [f"%{token.lower()}%"]
        if document_ids:
            placeholders = ",".join(["?"] * len(document_ids))
            sql += f" AND v.document_id IN ({placeholders})"
            params.extend(document_ids)

        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        out: list[dict] = []
        for document_id, source_name, chunk_id, text in rows:
            score = float(text.lower().count(token.lower()))
            out.append(
                {
                    "document_id": document_id,
                    "source_name": source_name,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                }
            )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
