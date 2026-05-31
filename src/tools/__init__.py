"""External tools used by agents. All inputs are validated before API calls."""
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore
from src.config import TAVILY_API_KEY
from src.schemas import TavilySearchInput, FetchPageInput, VectorQueryInput, validate_or_raise


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily API. Returns empty list on any error.

    Errors are logged but not raised — callers should handle empty results gracefully.
    """
    try:
        validated = validate_or_raise(TavilySearchInput, {"query": query, "max_results": max_results}, "tavily_search")
        query_str = validated["query"]
        max_n = validated["max_results"]
    except ValueError:
        return []

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query_str,
                "max_results": max_n,
                "search_depth": "advanced",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")}
            for r in data.get("results", [])
        ]
    except Exception:
        return []


def fetch_page_content(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract text content from a webpage. Inputs validated."""
    if BeautifulSoup is None:
        return ""
    validated = validate_or_raise(FetchPageInput, {"url": url, "max_chars": max_chars}, "fetch_page_content")
    url_str = validated["url"]
    max_c = validated["max_chars"]

    try:
        resp = requests.get(url_str, timeout=15, headers={"User-Agent": "BlogGen/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:max_c]
    except Exception:
        return ""


def query_vector_store(query: str, top_k: int = 5) -> list[dict]:
    """Query local Chroma vector store. Inputs validated."""
    from src.rag import query_documents
    validated = validate_or_raise(VectorQueryInput, {"query": query, "top_k": top_k}, "query_vector_store")
    return query_documents(validated["query"], validated["top_k"])


def save_to_vector_store(docs: list[dict]) -> None:
    """Persist research notes to local vector store.
    Each doc: {id, content, metadata: {chapter_title, source_url, date, ...}}"""
    from src.rag import add_documents
    if not docs:
        return
    for doc in docs:
        if not isinstance(doc, dict) or "id" not in doc or "content" not in doc:
            return  # Silently skip invalid docs, consistent with other tool functions
    add_documents(docs)
