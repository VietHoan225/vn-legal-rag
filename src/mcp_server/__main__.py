"""Run the MCP server over stdio: `python -m src.mcp_server`.

Logs go to stderr only; stdout carries the MCP protocol and must stay clean.
"""

import logging
import sys

from src.agent.tools import LegalSearchTool
from src.config import settings
from src.mcp_server.law_catalog import LawCatalog
from src.mcp_server.server import LazyValue, create_server
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def build_search_tool() -> LegalSearchTool:
    """Load embedder, BM25 index and reranker (slow: done once)."""
    retriever = HybridRetriever(QdrantVectorStore(), Embedder())
    return LegalSearchTool(retriever, Reranker())


def build_catalog() -> LawCatalog:
    """Load the raw legal documents used by get_article and list_laws."""
    return LawCatalog.from_file(settings.raw_data_path)


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    search_tool = LazyValue(build_search_tool)
    catalog = LazyValue(build_catalog)
    search_tool.warm_up()
    catalog.warm_up()
    create_server(search_tool, catalog).run()


if __name__ == "__main__":
    main()
