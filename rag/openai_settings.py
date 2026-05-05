from __future__ import annotations

import os

from dotenv import load_dotenv


def get_openai_api_key() -> str:
    load_dotenv()
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_openai_base_url() -> str:
    load_dotenv()
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
