"""MCP server exposing Vietnamese legal search as tools for AI clients."""

import logging
import threading
import time
from typing import Any, Callable, Generic, Protocol, TypeVar

from mcp.server.fastmcp import FastMCP

from src.agent.tools import SearchError
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_TOP_K = 10
DEFAULT_TOP_K = 5

DATA_NOTE = (
    "Dữ liệu là bộ văn bản Zalo Legal và có thể đã lỗi thời (ví dụ Nghị định 100/2019/NĐ-CP "
    "về xử phạt giao thông đã được thay bởi Nghị định 168/2024/NĐ-CP). Khi trả lời về mức phạt "
    "hay quy định cụ thể, hãy nhắc người dùng kiểm tra với văn bản pháp luật hiện hành."
)

SEARCH_LAW_DESCRIPTION = f"""Tìm các điều luật Việt Nam liên quan đến một câu hỏi hoặc chủ đề.

Dùng tool này TRƯỚC khi trả lời bất kỳ câu hỏi nào về quy định pháp luật Việt Nam (mức phạt,
quyền và nghĩa vụ, thủ tục...). Tìm theo ý nghĩa và từ khóa trong bộ văn bản pháp luật đã nạp.

Cách viết `query`: dùng thuật ngữ pháp lý như trong văn bản luật sẽ tìm tốt hơn cách nói thông
thường. Ví dụ dùng "không chấp hành hiệu lệnh của đèn tín hiệu giao thông" thay cho
"vượt đèn đỏ"; "không đội mũ bảo hiểm khi đi xe mô tô" thay cho "đi xe máy không mũ".
Nên nêu rõ loại phương tiện hoặc đối tượng (xe máy, ô tô, cá nhân, tổ chức).

Args:
    query: Câu hỏi hoặc từ khóa pháp lý cần tra cứu.
    top_k: Số đoạn luật trả về (1-{MAX_TOP_K}, mặc định {DEFAULT_TOP_K}).

Returns:
    Danh sách `results` xếp theo độ liên quan giảm dần. Mỗi mục có `law` (số hiệu văn bản),
    `article` (số Điều), `clause` (số Khoản, có thể null), `citation` (chuỗi trích dẫn sẵn),
    `relevance_score` (điểm của reranker: càng cao càng liên quan, KHÔNG phải xác suất 0-1)
    và `text` (nội dung đoạn luật). Hãy trích dẫn đúng `citation` khi trả lời.

Lưu ý: {DATA_NOTE}"""


class SearchTool(Protocol):
    """What the server needs from the search backend (LegalSearchTool satisfies it)."""

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]: ...


class LazyValue(Generic[T]):
    """Build a heavy object on first use (models, indexes) and keep it.

    A failed build is not cached, so the next call retries.
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._loaded = False
        self._value: T | None = None

    def get(self) -> T:
        with self._lock:
            if not self._loaded:
                self._value = self._factory()
                self._loaded = True
            return self._value  # type: ignore[return-value]

    def warm_up(self) -> None:
        """Start building in the background so the first tool call does not pay for it."""

        def run() -> None:
            try:
                self.get()
            except Exception:
                logger.exception("Warm-up failed; will retry on first tool call")

        threading.Thread(target=run, name="warm-up", daemon=True).start()


def format_hit(rank: int, hit: SearchResult) -> dict[str, Any]:
    """Convert a search hit into the compact structure returned to the client."""
    law = hit.title.removeprefix("Văn bản số ").strip()
    return {
        "rank": rank,
        "law": law,
        "article": hit.article_number,
        "clause": hit.clause_number,
        "citation": f"{law}, {hit.location()}",
        "relevance_score": round(hit.score, 3),
        "text": hit.text,
    }


def create_server(search_tool: Callable[[], SearchTool] | LazyValue[SearchTool]) -> FastMCP:
    """Build the MCP server; `search_tool` is a factory or LazyValue so models load lazily."""
    get_search_tool = search_tool.get if isinstance(search_tool, LazyValue) else search_tool
    mcp = FastMCP("luat-vn")

    @mcp.tool(name="search_law", description=SEARCH_LAW_DESCRIPTION)
    def search_law(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("`query` must not be empty.")
        top_k = max(1, min(top_k, MAX_TOP_K))
        logger.info("search_law query=%r top_k=%d", query, top_k)
        t0 = time.perf_counter()
        try:
            hits = get_search_tool().search(query, top_k=top_k)
            logger.info("search_law: %d hit(s) in %.2fs", len(hits), time.perf_counter() - t0)
        except SearchError as e:
            logger.error("search_law failed: %s", e)
            raise RuntimeError(
                "Tra cứu thất bại (cơ sở dữ liệu vector hoặc model không sẵn sàng). "
                "Đây là lỗi hệ thống, không có nghĩa là không có quy định phù hợp."
            ) from e
        return {
            "query": query,
            "count": len(hits),
            "results": [format_hit(i, h) for i, h in enumerate(hits, start=1)],
            "note": DATA_NOTE,
        }

    return mcp
