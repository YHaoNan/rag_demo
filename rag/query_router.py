from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APITimeoutError
from openai import OpenAI
from .openai_settings import get_openai_chat_api_key, get_openai_chat_base_url


@dataclass(frozen=True)
class QueryRoute:
    route: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class RouterConfig:
    model: str
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "RouterConfig | None":
        load_dotenv()
        api_key = get_openai_chat_api_key()
        model = (
            os.getenv("OPENAI_ROUTER_MODEL", "").strip()
            or os.getenv("OPENAI_CHAT_MODEL", "").strip()
            or os.getenv("OPENAI_CONTEXT_MODEL", "").strip()
        )
        if not api_key or not model:
            return None

        timeout_seconds = float(os.getenv("OPENAI_ROUTER_TIMEOUT_SECONDS", "60"))
        max_retries = int(os.getenv("OPENAI_ROUTER_MAX_RETRIES", "2"))
        return cls(
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


SCAN_HINTS = ("几次", "多少次", "出现", "所有", "列出", "where", "grep", "find all", "count")
SUMMARY_HINTS = ("总结", "概述", "主要讲了什么", "优缺点", "风险点", "overview", "summary")
STEP_HINTS = ("如何", "步骤", "流程", "怎么处理", "排查", "部署", "升级")
ALLOWED_ROUTES = {"scan", "summary", "steps", "semantic", "fact"}


def _rule_route(query: str) -> QueryRoute:
    text = query.strip().lower()
    if not text:
        return QueryRoute(route="fact", confidence=0.3, reason="empty query fallback")
    if any(k in text for k in SCAN_HINTS):
        return QueryRoute(route="scan", confidence=0.9, reason="scan keyword matched")
    if any(k in text for k in SUMMARY_HINTS):
        return QueryRoute(route="summary", confidence=0.85, reason="summary keyword matched")
    if any(k in text for k in STEP_HINTS):
        return QueryRoute(route="steps", confidence=0.8, reason="workflow keyword matched")
    if len(text) > 40:
        return QueryRoute(route="semantic", confidence=0.6, reason="long query defaults to semantic")
    return QueryRoute(route="fact", confidence=0.75, reason="default factual retrieval")


def _parse_router_json(content: str) -> QueryRoute | None:
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None
    route = str(obj.get("route", "")).strip().lower()
    if route not in ALLOWED_ROUTES:
        return None
    try:
        confidence = float(obj.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = max(0.0, min(confidence, 1.0))
    reason = str(obj.get("reason", "llm route")).strip() or "llm route"
    return QueryRoute(route=route, confidence=confidence, reason=reason)


def _llm_route(query: str, cfg: RouterConfig) -> QueryRoute | None:
    client = OpenAI(
        api_key=get_openai_chat_api_key(),
        base_url=get_openai_chat_base_url(),
        timeout=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )
    prompt = (
        "You are a query router for a RAG system.\n"
        "Choose one route from: fact, summary, scan, semantic, steps.\n"
        "- fact: When a user's question is asking about a simple fact, and the correct answer to that fact can usually be found somewhere in the text.\n"
        "\teg: API 的超时时间是多少？\n"
        "- summary: When a user’s question requires a summary of the entire text or chapter to answer.\n"
        "\teg: 这篇文档主要讲了什么？\n"
        "- scan: When a user's query requires the use of a regular expression search tool rather than a large language model.\n"
        "\teg: 这个类名在文件中出现了几次？\n"
        "- semantic: When the user's question isn't based on the original text at all, but requires understanding and reasoning based on a large amount of original text.\n"
        "\teg: 这个架构是否存在单点故障？\n"
        "- steps: When a user asks a question about a specific procedure, the answer should include clear, step-by-step instructions.\n"
        "\teg: 如何升级这个组件？\n"
        "Return strict JSON only: "
        '{"route":"fact|summary|scan|semantic|steps","confidence":0.0,"reason":"short reason"}.\n'
        "No markdown, no extra text.\n"
        "FALLBACK: When the issue category cannot be determined, use “fact” and set the ‘reason’ to “fallback”"
    )
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return _parse_router_json(content)


def route_query(query: str) -> QueryRoute:
    cfg = RouterConfig.from_env()
    if cfg is not None:
        try:
            llm_result = _llm_route(query, cfg)
            if llm_result is not None:
                return llm_result
            fallback = _rule_route(query)
            return QueryRoute(
                route=fallback.route,
                confidence=fallback.confidence,
                reason=f"llm_invalid_json_fallback: {fallback.reason}",
            )
        except APITimeoutError:
            fallback = _rule_route(query)
            return QueryRoute(
                route=fallback.route,
                confidence=fallback.confidence,
                reason=f"llm_timeout_fallback: {fallback.reason}",
            )
        except Exception:
            fallback = _rule_route(query)
            return QueryRoute(
                route=fallback.route,
                confidence=fallback.confidence,
                reason=f"llm_error_fallback: {fallback.reason}",
            )
    return _rule_route(query)
