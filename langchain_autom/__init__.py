"""LangChain integration for Autom, a SERP API for live search engine results."""

from importlib import metadata

from langchain_autom._client import AutomAPIError
from langchain_autom.retrievers import AutomSearchRetriever
from langchain_autom.tools import AutomSearch, AutomSearchInput

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = ["AutomAPIError", "AutomSearch", "AutomSearchInput", "AutomSearchRetriever", "__version__"]
