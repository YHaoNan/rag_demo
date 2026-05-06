from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template, request, url_for

from .config import LlamaRagConfig
from .ingest import parse_options_from_form, split_file, store_indexes
from .session_store import SplitSessionStore
from .task_store import TaskStore


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    cfg = LlamaRagConfig.from_env()
    if cfg.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    store = SplitSessionStore(ttl_seconds=cfg.session_ttl_seconds)
    task_store = TaskStore(cfg.task_store_file)
    upload_dir = Path(cfg.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html", last_result=None, last_store_result=None, error_msg="")

    @app.get("/tasks")
    def tasks():
        tasks_data = task_store.list_tasks()
        return render_template("tasks.html", tasks=tasks_data)

    @app.post("/split")
    def split():
        store.cleanup_expired()
        file = request.files.get("file")
        if file is None or not file.filename:
            return redirect(url_for("index"))
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".md", ".markdown", ".pdf", ".docx", ".doc"}:
            return redirect(url_for("index"))

        safe_name = f"{Path(file.filename).stem}-{uuid4().hex[:8]}{suffix}"
        path = upload_dir / safe_name
        file.save(path)

        options = parse_options_from_form(request.form, cfg)
        try:
            nodes, result = split_file(path, cfg, options)
            session_id = uuid4().hex
            store.set(session_id, {"nodes": nodes, "split_result": result})
            result["session_id"] = session_id
            result["expires_in_seconds"] = cfg.session_ttl_seconds
            return render_template("index.html", last_result=result, last_store_result=None, error_msg="")
        except Exception as e:
            return render_template("index.html", last_result=None, last_store_result=None, error_msg=str(e))

    @app.post("/index")
    def index_store():
        store.cleanup_expired()
        session_id = (request.form.get("session_id") or "").strip()
        index_types = request.form.getlist("index_types")
        if not session_id:
            return render_template("index.html", last_result=None, last_store_result=None, error_msg="session_id is required")
        if not index_types:
            return render_template("index.html", last_result=None, last_store_result=None, error_msg="Please choose at least one index type")
        item = store.get(session_id)
        if not item:
            return render_template(
                "index.html",
                last_result=None,
                last_store_result=None,
                error_msg="Session expired or not found. Please split again.",
            )
        try:
            result = store_indexes(session_id=session_id, nodes=item["nodes"], cfg=cfg, index_types=index_types)
            results = result.get("results", [])
            all_ok = bool(results) and all(x.get("ok") for x in results)
            if all_ok:
                split_result = item.get("split_result", {})
                task_store.add_completed_task(
                    {
                        "session_id": session_id,
                        "source_name": split_result.get("source_name", ""),
                        "split_mode": split_result.get("split_mode", ""),
                        "document_count": split_result.get("document_count", 0),
                        "chunk_count": split_result.get("chunk_count", 0),
                        "nodes": split_result.get("nodes", []),
                        "index_results": results,
                    }
                )
            split_result = item.get("split_result", {})
            split_result["session_id"] = session_id
            split_result["expires_in_seconds"] = cfg.session_ttl_seconds
            return render_template("index.html", last_result=split_result, last_store_result=result, error_msg="")
        except Exception as e:
            split_result = item.get("split_result", {})
            split_result["session_id"] = session_id
            split_result["expires_in_seconds"] = cfg.session_ttl_seconds
            return render_template("index.html", last_result=split_result, last_store_result=None, error_msg=str(e))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=True)
