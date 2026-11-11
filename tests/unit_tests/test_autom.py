"""Offline tests: every request goes through an httpx.MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from langchain_autom import AutomSearch, AutomSearchRetriever, _client

API_KEY = "test-autom-key-placeholder"  # pragma: allowlist secret
ROWS = [
    {"position": 1, "title": "LangChain", "link": "https://www.langchain.com/", "snippet": "Build agents."},
    {"position": 2, "title": "Only a title", "link": "https://example.com/"},
    "not an object",
    {"position": 3, "title": "News", "link": "https://news.example.com/", "snippet": "s", "date": "1 day ago"},
]

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[httpx.Request]]:
    """Route both sync and async httpx clients created by the package to ``handler``."""

    def install(handler: Handler) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        transport = httpx.MockTransport(record)
        real_client, real_async = httpx.Client, httpx.AsyncClient
        monkeypatch.setattr(_client.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
        monkeypatch.setattr(_client.httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw))
        return seen

    return install


def ok(rows: object = ROWS) -> Handler:
    return lambda request: httpx.Response(200, json={"organic_results": rows})


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOM_API_KEY", raising=False)


class TestAutomSearch:
    def test_invoke_sends_key_and_normalizes_rows(self, mock_http) -> None:
        seen = mock_http(ok())
        out = AutomSearch(api_key=API_KEY, max_results=2).invoke({"query": "  langchain  "})

        request = seen[0]
        assert request.url.path == "/v1/google/search"
        assert request.url.params["query"] == "langchain"
        assert request.headers["x-api-key"] == API_KEY
        assert out["query"] == "langchain"
        assert out["results"] == [
            {"title": "LangChain", "url": "https://www.langchain.com/", "content": "Build agents.", "position": 1},
            {"title": "Only a title", "url": "https://example.com/", "content": "", "position": 2},
        ]

    def test_google_options_and_time_range(self, mock_http) -> None:
        seen = mock_http(ok([]))
        tool = AutomSearch(api_key=API_KEY, country="US", language="fr", location="Paris, France")
        tool.invoke({"query": "q", "time_range": "week"})

        assert dict(seen[0].url.params) == {
            "query": "q",
            "gl": "us",
            "hl": "fr",
            "location": "Paris, France",
            "tbs": "qdr:w",
        }

    @pytest.mark.parametrize("engine", ["bing", "brave", "duckduckgo", "yahoo"])
    def test_other_engines_use_their_path_and_skip_google_options(self, mock_http, engine: str) -> None:
        seen = mock_http(ok([]))
        AutomSearch(api_key=API_KEY, engine=engine, country="us").invoke({"query": "q", "time_range": "day"})

        assert seen[0].url.path == f"/v1/{engine}/search"
        assert dict(seen[0].url.params) == {"query": "q"}

    def test_env_key_is_used_and_stripped(self, mock_http, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOM_API_KEY", f" {API_KEY}\n")
        seen = mock_http(ok([]))
        AutomSearch().invoke({"query": "q"})

        assert seen[0].headers["x-api-key"] == API_KEY

    def test_missing_key_is_returned_to_the_model(self, mock_http) -> None:
        seen = mock_http(ok())
        out = AutomSearch().invoke({"query": "q"})

        assert "AUTOM_API_KEY" in out
        assert seen == []

    def test_empty_query_is_returned_to_the_model(self, mock_http) -> None:
        seen = mock_http(ok())
        assert AutomSearch(api_key=API_KEY).invoke({"query": "   "}) == "The search query is empty."
        assert seen == []

    def test_missing_organic_results_is_an_empty_search(self, mock_http) -> None:
        mock_http(lambda request: httpx.Response(200, json={"search_parameters": {}}))
        assert AutomSearch(api_key=API_KEY).invoke({"query": "q"})["results"] == []

    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (401, {"error": "Invalid API key"}, "Autom rejected the API key (HTTP 401): Invalid API key"),
            (401, {"error": "Rate limit exceeded"}, "Autom rate limit exceeded (HTTP 401): Rate limit exceeded"),
            (402, {"message": "No credits"}, "Autom account is out of credits (HTTP 402): No credits"),
            (500, {"message": f"boom {API_KEY}"}, "Autom API error (HTTP 500): boom ***"),
        ],
    )
    def test_http_errors_are_returned_without_the_key(self, mock_http, status, body, expected) -> None:
        mock_http(lambda request: httpx.Response(status, json=body))
        assert AutomSearch(api_key=API_KEY).invoke({"query": "q"}) == expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            (httpx.Response(200, text="<html>"), "not JSON"),
            (httpx.Response(200, json=["x"]), "unexpected response shape"),
            (httpx.Response(200, json={"organic_results": "x"}), "not a list"),
        ],
    )
    def test_malformed_bodies(self, mock_http, response: httpx.Response, expected: str) -> None:
        mock_http(lambda request: response)
        assert expected in AutomSearch(api_key=API_KEY).invoke({"query": "q"})

    def test_timeout(self, mock_http) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        mock_http(slow)
        assert (
            AutomSearch(api_key=API_KEY, timeout=2).invoke({"query": "q"}) == "Autom did not answer within 2 seconds."
        )

    def test_network_error_without_the_key(self, mock_http) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"refused {API_KEY}", request=request)

        mock_http(down)
        out = AutomSearch(api_key=API_KEY).invoke({"query": "q"})
        assert out.startswith("Could not reach Autom") and API_KEY not in out

    async def test_ainvoke(self, mock_http) -> None:
        seen = mock_http(ok())
        out = await AutomSearch(api_key=API_KEY).ainvoke({"query": "q"})

        assert len(out["results"]) == 3
        assert out["results"][2]["date"] == "1 day ago"
        assert seen[0].headers["x-api-key"] == API_KEY

    def test_tool_call_returns_a_tool_message(self, mock_http) -> None:
        mock_http(ok())
        message = AutomSearch(api_key=API_KEY).invoke(
            {"type": "tool_call", "id": "call_1", "name": "autom_search", "args": {"query": "q"}}
        )
        assert message.tool_call_id == "call_1"
        assert json.loads(message.content)["results"][0]["url"] == "https://www.langchain.com/"


class TestAutomSearchRetriever:
    def test_documents(self, mock_http) -> None:
        seen = mock_http(ok())
        docs = AutomSearchRetriever(api_key=API_KEY, k=2, time_range="month").invoke("q")

        assert seen[0].url.params["tbs"] == "qdr:m"
        assert [d.page_content for d in docs] == ["Build agents.", "Only a title"]
        assert docs[0].metadata == {"title": "LangChain", "url": "https://www.langchain.com/", "position": 1}

    async def test_async_documents(self, mock_http) -> None:
        mock_http(ok())
        docs = await AutomSearchRetriever(api_key=API_KEY, engine="brave").ainvoke("q")
        assert len(docs) == 3

    def test_errors_raise(self, mock_http) -> None:
        mock_http(lambda request: httpx.Response(401, json={"error": "Invalid API key"}))
        with pytest.raises(_client.AutomAPIError, match="rejected the API key"):
            AutomSearchRetriever(api_key=API_KEY).invoke("q")


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        AutomSearch(engine="altavista")
