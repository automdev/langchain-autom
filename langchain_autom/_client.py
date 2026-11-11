"""HTTP client for the Autom SERP API (https://docs.autom.dev)."""

from __future__ import annotations

import os
from typing import Any, Literal

import httpx

DEFAULT_BASE_URL = "https://api.autom.dev"
DEFAULT_TIMEOUT = 60.0
USER_AGENT = "langchain-autom"

Engine = Literal["google", "bing", "brave", "duckduckgo", "yahoo"]
ENGINES: tuple[str, ...] = ("google", "bing", "brave", "duckduckgo", "yahoo")

_ERROR_DETAIL_LIMIT = 300


class AutomAPIError(Exception):
    """An Autom request failed. The message never contains the API key."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_key(value: str | None) -> str:
    """Return ``value`` or ``AUTOM_API_KEY``, stripped. Raises if neither is set."""
    key = (value or os.environ.get("AUTOM_API_KEY") or "").strip()
    if not key:
        raise AutomAPIError("Set AUTOM_API_KEY or pass api_key. Get a key at https://app.autom.dev")
    return key


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


def _error(response: httpx.Response, api_key: str) -> AutomAPIError:
    detail = ""
    try:
        body = response.json()
    except ValueError:
        detail = response.text.strip()
    else:
        if isinstance(body, dict):
            detail = str(body.get("message") or body.get("error") or "")
    detail = _redact(detail, api_key)[:_ERROR_DETAIL_LIMIT]
    status = response.status_code
    if status == 401 and "rate limit" in detail.lower():
        label = "Autom rate limit exceeded"
    elif status == 401:
        label = "Autom rejected the API key"
    elif status == 402:
        label = "Autom account is out of credits"
    elif status == 429:
        label = "Autom rate limit exceeded"
    else:
        label = "Autom API error"
    message = f"{label} (HTTP {status})" + (f": {detail}" if detail else "")
    return AutomAPIError(message, status)


def _parse(response: httpx.Response, api_key: str) -> list[dict[str, Any]]:
    if not response.is_success:
        raise _error(response, api_key)
    try:
        body = response.json()
    except ValueError as exc:
        raise AutomAPIError("Autom returned a response that is not JSON.") from exc
    if not isinstance(body, dict):
        raise AutomAPIError("Autom returned an unexpected response shape.")
    rows = body.get("organic_results")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise AutomAPIError("Autom returned organic_results that is not a list.")
    return [row for row in rows if isinstance(row, dict)]


def build_params(
    query: str,
    *,
    country: str | None = None,
    language: str | None = None,
    location: str | None = None,
    time_range: str | None = None,
    engine: str = "google",
) -> dict[str, str]:
    """Query parameters for ``/v1/{engine}/search``. Country, language, location and
    time range only apply to Google."""
    params = {"query": query}
    if engine == "google":
        if country:
            params["gl"] = country.strip().lower()
        if language:
            params["hl"] = language.strip()
        if location:
            params["location"] = location.strip()
        if time_range:
            params["tbs"] = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}[time_range]
    return params


def _request(api_key: str, base_url: str, engine: str) -> tuple[str, dict[str, str]]:
    if engine not in ENGINES:
        raise AutomAPIError(f"Unknown engine {engine!r}; use one of {', '.join(ENGINES)}.")
    headers = {"x-api-key": api_key, "Accept": "application/json", "User-Agent": USER_AGENT}
    return f"{base_url.rstrip('/')}/v1/{engine}/search", headers


def search(
    params: dict[str, str],
    *,
    api_key: str,
    engine: str = "google",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Run one search and return the organic result rows."""
    url, headers = _request(api_key, base_url, engine)
    try:
        if client is not None:
            response = client.get(url, params=params, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as owned:
                response = owned.get(url, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        raise AutomAPIError(f"Autom did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise AutomAPIError(f"Could not reach Autom: {_redact(str(exc), api_key)}") from exc
    return _parse(response, api_key)


async def asearch(
    params: dict[str, str],
    *,
    api_key: str,
    engine: str = "google",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Async version of :func:`search`."""
    url, headers = _request(api_key, base_url, engine)
    try:
        if client is not None:
            response = await client.get(url, params=params, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as owned:
                response = await owned.get(url, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        raise AutomAPIError(f"Autom did not answer within {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise AutomAPIError(f"Could not reach Autom: {_redact(str(exc), api_key)}") from exc
    return _parse(response, api_key)


def to_result(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one organic row to ``title``/``url``/``content`` plus the useful extras."""
    result: dict[str, Any] = {
        "title": row.get("title") or "",
        "url": row.get("link") or "",
        "content": row.get("snippet") or "",
    }
    for key in ("position", "date", "source"):
        if row.get(key) not in (None, ""):
            result[key] = row[key]
    return result
