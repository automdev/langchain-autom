"""Autom search tool for LangChain agents."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SecretStr

from langchain_autom import _client


class AutomSearchInput(BaseModel):
    """Arguments the model can set."""

    query: str = Field(description="The search query. Search operators such as site: are supported.")
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description="Only return pages published in the past day, week, month or year. Google only.",
    )


class AutomSearch(BaseTool):
    """Search the web with Autom and return the organic results.

    Setup:
        Install ``langchain-autom`` and set ``AUTOM_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-autom
            export AUTOM_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_autom import AutomSearch

            tool = AutomSearch(max_results=5, engine="google", country="us")

    Invocation:
        .. code-block:: python

            tool.invoke({"query": "langchain agents"})

        Returns a dict with the ``query`` and a ``results`` list of
        ``title``, ``url``, ``content`` and ``position``.
    """

    name: str = "autom_search"
    description: str = (
        "Search the web through Autom and get the top organic results with title, url and snippet. "
        "Input is a search query; use it for current events and facts the model does not know."
    )
    args_schema: type[BaseModel] = AutomSearchInput
    handle_tool_error: bool = True

    api_key: SecretStr | None = Field(default=None, description="Autom API key. Defaults to AUTOM_API_KEY.")
    engine: _client.Engine = "google"
    """Search engine: google, bing, brave, duckduckgo or yahoo."""
    max_results: int = Field(default=5, ge=1, le=10)
    """Number of results returned to the model (one results page holds about 10)."""
    country: str | None = None
    """Two-letter country code to search from, such as us. Google only."""
    language: str | None = None
    """Language code of the results, such as en. Google only."""
    location: str | None = None
    """City or region to search from, such as "Paris, France". Google only."""
    base_url: str = _client.DEFAULT_BASE_URL
    timeout: float = _client.DEFAULT_TIMEOUT

    def _key(self) -> str:
        try:
            return _client.resolve_api_key(self.api_key.get_secret_value() if self.api_key else None)
        except _client.AutomAPIError as exc:
            raise ToolException(str(exc)) from exc

    def _params(self, query: str, time_range: str | None) -> dict[str, str]:
        query = query.strip()
        if not query:
            raise ToolException("The search query is empty.")
        return _client.build_params(
            query,
            country=self.country,
            language=self.language,
            location=self.location,
            time_range=time_range,
            engine=self.engine,
        )

    def _output(self, query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"query": query, "results": [_client.to_result(row) for row in rows[: self.max_results]]}

    def _run(
        self,
        query: str,
        time_range: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        params = self._params(query, time_range)
        try:
            rows = _client.search(
                params, api_key=self._key(), engine=self.engine, base_url=self.base_url, timeout=self.timeout
            )
        except _client.AutomAPIError as exc:
            raise ToolException(str(exc)) from exc
        return self._output(params["query"], rows)

    async def _arun(
        self,
        query: str,
        time_range: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        params = self._params(query, time_range)
        try:
            rows = await _client.asearch(
                params, api_key=self._key(), engine=self.engine, base_url=self.base_url, timeout=self.timeout
            )
        except _client.AutomAPIError as exc:
            raise ToolException(str(exc)) from exc
        return self._output(params["query"], rows)
