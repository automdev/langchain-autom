# langchain-autom

LangChain integration for [Autom](https://www.autom.dev), a SERP API that returns live search engine results as JSON.

It adds two components:

- `AutomSearch`, a tool that agents call to search the web;
- `AutomSearchRetriever`, a retriever that returns the results as LangChain documents.

Both work with Google (the default), Bing, Brave, DuckDuckGo and Yahoo.

## Installation

```bash
pip install -U langchain-autom
```

Get an API key at [app.autom.dev](https://app.autom.dev) and set it in the environment:

```bash
export AUTOM_API_KEY="your-api-key"
```

You can also pass `api_key=...` to either component.

## Search tool

```python
from langchain_autom import AutomSearch

tool = AutomSearch(max_results=5)
tool.invoke({"query": "latest LangChain release"})
```

```python
{
    "query": "latest LangChain release",
    "results": [{"title": "...", "url": "https://...", "content": "...", "position": 1}, ...],
}
```

The model sets `query`, and optionally `time_range` (`day`, `week`, `month` or `year`) to keep only recent pages.

You set the other options when you create the tool:

| Option | Default | Description |
| --- | --- | --- |
| `engine` | `google` | `google`, `bing`, `brave`, `duckduckgo` or `yahoo` |
| `max_results` | `5` | Results returned to the model, 1 to 10 |
| `country` | | Two-letter country code, such as `us` (Google only) |
| `language` | | Language code of the results, such as `en` (Google only) |
| `location` | | City or region to search from, such as `Paris, France` (Google only) |
| `timeout` | `60` | Seconds to wait for Autom |

API errors are returned to the model as text instead of raising, and the key never appears in them. Autom reports a rate limit as HTTP 401 with "Rate limit exceeded"; the tool says "rate limit exceeded" in that case.

### With an agent

```python
from langchain.agents import create_agent
from langchain_autom import AutomSearch

agent = create_agent("openai:gpt-4.1-mini", tools=[AutomSearch()])
agent.invoke({"messages": [{"role": "user", "content": "What changed in Python 3.13?"}]})
```

## Retriever

```python
from langchain_autom import AutomSearchRetriever

retriever = AutomSearchRetriever(k=5, engine="brave")
docs = retriever.invoke("retrieval augmented generation")
```

Each document's `page_content` is the result snippet. Its metadata holds `title`, `url` and `position`, plus `date` and `source` when Autom returns them. The retriever takes the same options as the tool, with `k` in place of `max_results` and a fixed `time_range`.

Both components have async versions (`ainvoke`).

## Development

```bash
pip install -e ".[test]"
pytest tests/unit_tests                      # offline
AUTOM_API_KEY=... pytest tests/integration_tests  # live
```

## License

MIT
