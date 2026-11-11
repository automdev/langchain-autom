"""LangChain standard unit tests for the tool."""

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_autom import AutomSearch


class TestAutomSearchStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[AutomSearch]:
        return AutomSearch

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-autom-key-placeholder", "max_results": 3}  # pragma: allowlist secret

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "langchain agents"}
