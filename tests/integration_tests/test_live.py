"""Live tests against api.autom.dev. They run only when AUTOM_API_KEY is set."""

import os

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_autom import AutomSearch, AutomSearchRetriever

pytestmark = pytest.mark.skipif(not os.environ.get("AUTOM_API_KEY"), reason="AUTOM_API_KEY is not set")


class TestAutomSearchLive(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[AutomSearch]:
        return AutomSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "langchain"}


def test_search_returns_results() -> None:
    out = AutomSearch(max_results=3).invoke({"query": "langchain"})
    assert out["results"] and out["results"][0]["url"].startswith("http")


def test_retriever_returns_documents() -> None:
    docs = AutomSearchRetriever(k=3).invoke("python programming language")
    assert docs and docs[0].metadata["url"].startswith("http")
