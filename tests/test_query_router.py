from rag import query_router
from rag.query_router import _parse_router_json, route_query


def test_route_query_scan():
    route = route_query("这个模式出现几次")
    assert route.route == "scan"


def test_route_query_summary():
    route = route_query("这篇文档主要讲了什么")
    assert route.route == "summary"


def test_route_query_steps():
    route = route_query("如何部署这个系统")
    assert route.route == "steps"


def test_parse_router_json_ok():
    route = _parse_router_json('{"route":"semantic","confidence":0.88,"reason":"multi-hop inference"}')
    assert route is not None
    assert route.route == "semantic"
    assert route.confidence == 0.88


def test_parse_router_json_invalid():
    route = _parse_router_json("not-json")
    assert route is None


def test_route_query_llm_invalid_json_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_API_KEY", "k")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "m")

    def fake_llm_route(query, cfg):
        return None

    monkeypatch.setattr(query_router, "_llm_route", fake_llm_route)
    route = route_query("这篇文档主要讲了什么")
    assert route.route == "summary"
    assert route.reason.startswith("llm_invalid_json_fallback")
