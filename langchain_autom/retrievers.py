"""Autom search retriever: web results as LangChain documents."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr

from langchain_autom import _client


class AutomSearchRetriever(BaseRetriever):
    """Retrieve web search results from Autom as documents.

    Each document's ``page_content`` is the result snippet; its metadata holds the
    ``title``, ``url``, ``position`` and, for news-like rows, ``date`` and ``source``.

    Example:
        .. code-block:: python

            from langchain_autom import AutomSearchRetriever

            retriever = AutomSearchRetriever(k=5)
            docs = retriever.invoke("what is retrieval augmented generation")
    """

    api_key: SecretStr | None = Field(default=None, description="Autom API key. Defaults to AUTOM_API_KEY.")
    engine: _client.Engine = "google"
    k: int = Field(default=5, ge=1, le=10)
    country: str | None = None
    language: str | None = None
    location: str | None = None
    time_range: Literal["day", "week", "month", "year"] | None = None
    base_url: str = _client.DEFAULT_BASE_URL
    timeout: float = _client.DEFAULT_TIMEOUT

    def _params(self, query: str) -> dict[str, str]:
        return _client.build_params(
            query.strip(),
            country=self.country,
            language=self.language,
            location=self.location,
            time_range=self.time_range,
            engine=self.engine,
        )

    def _key(self) -> str:
        return _client.resolve_api_key(self.api_key.get_secret_value() if self.api_key else None)

    def _documents(self, rows: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for row in rows[: self.k]:
            result = _client.to_result(row)
            content = result.pop("content") or result["title"]
            documents.append(Document(page_content=content, metadata=result))
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        rows = _client.search(
            self._params(query), api_key=self._key(), engine=self.engine, base_url=self.base_url, timeout=self.timeout
        )
        return self._documents(rows)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        rows = await _client.asearch(
            self._params(query), api_key=self._key(), engine=self.engine, base_url=self.base_url, timeout=self.timeout
        )
        return self._documents(rows)
