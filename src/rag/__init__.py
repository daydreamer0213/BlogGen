"""RAG module: Hybrid search (BM25 + Vector) + optional BGE Reranker.

Three-stage retrieval:
  Stage 1: BM25 (keyword) + Vector (semantic) in parallel → merge via RRF
  Stage 2: BGE Reranker cross-encoder → re-score → return top_k (optional)

Lazy init with fallback: all components degrade gracefully.
"""
from __future__ import annotations
import jieba

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore

from src.config import BGE_EMBEDDING_MODEL, BGE_RERANKER_MODEL, CHROMA_PERSIST_DIR

_embedding_model = None
_reranker_model = None
_chroma_client = None
_collection = None
_rag_available = None
_reranker_available = None

# BM25 index: mirrors Chroma collection
_bm25_index: BM25Okapi | None = None
_bm25_docs: list[dict] = []  # [{id, content}], ordered same as BM25 corpus
_bm25_available = False


# ============================================================
# Init & availability checks
# ============================================================

def is_rag_available() -> bool:
    global _rag_available
    if _rag_available is None:
        try:
            get_embedding_model()
            get_collection()
            _rag_available = True
        except Exception:
            _rag_available = False
    return _rag_available


def is_reranker_available() -> bool:
    get_reranker()
    return _reranker_available or False


def is_bm25_available() -> bool:
    _ensure_bm25()
    return _bm25_available


# ============================================================
# Embedding model
# ============================================================

def get_embedding_model():
    global _embedding_model, _rag_available
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(BGE_EMBEDDING_MODEL)
        except Exception as e:
            _rag_available = False
            raise RuntimeError(f"Failed to load BGE model from {BGE_EMBEDDING_MODEL}: {e}") from e
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# ============================================================
# Chroma vector store
# ============================================================

def get_collection():
    global _chroma_client, _collection, _rag_available
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as e:
            _rag_available = False
            raise RuntimeError(f"Failed to open Chroma at {CHROMA_PERSIST_DIR}: {e}") from e
    if _collection is None:
        try:
            _collection = _chroma_client.get_or_create_collection(
                name="bloggen_research",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            _rag_available = False
            raise RuntimeError(f"Failed to init Chroma collection: {e}") from e
    return _collection


# ============================================================
# BM25 keyword index
# ============================================================

def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese text with jieba, filtering whitespace-only tokens."""
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip()]


def _build_bm25_from_collection() -> None:
    """Rebuild BM25 index from all documents in Chroma."""
    global _bm25_index, _bm25_docs, _bm25_available
    try:
        collection = get_collection()
        # Fetch all docs from Chroma
        results = collection.get()
        if not results or not results.get("ids"):
            _bm25_available = False
            return
        _bm25_docs = []
        corpus = []
        for i, doc_id in enumerate(results["ids"]):
            content = results["documents"][i] if results.get("documents") else ""
            _bm25_docs.append({"id": doc_id, "content": content})
            corpus.append(_tokenize(content))
        _bm25_index = BM25Okapi(corpus)
        _bm25_available = True
    except Exception:
        _bm25_available = False


def _ensure_bm25() -> None:
    """Lazy-init BM25 index on first query."""
    global _bm25_available
    if _bm25_index is None:
        _build_bm25_from_collection()


def _bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 keyword search. Returns [{id, content, bm25_score}]."""
    _ensure_bm25()
    if not _bm25_available or _bm25_index is None:
        return []
    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)
    # Get top_k indices
    if len(scores) == 0:
        return []
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**_bm25_docs[i], "bm25_score": float(scores[i])}
        for i in indices if i < len(_bm25_docs)
    ]


# ============================================================
# Add documents (updates all indexes)
# ============================================================

def add_documents(docs: list[dict]) -> None:
    """Add documents to Chroma + BM25. Silently skips if unavailable."""
    if not docs:
        return
    # Chroma
    try:
        collection = get_collection()
        texts = [d["content"] for d in docs]
        ids = [d["id"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = embed_texts(texts)
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        # BM25: invalidate only after successful Chroma add
        global _bm25_index
        _bm25_index = None
        _bm25_available = False
    except Exception:
        pass


# ============================================================
# Reranker
# ============================================================

def get_reranker():
    global _reranker_model, _reranker_available
    if _reranker_available is None:
        try:
            _reranker_model = CrossEncoder(BGE_RERANKER_MODEL)
            _reranker_available = True
        except Exception:
            _reranker_available = False
    return _reranker_model if _reranker_available else None


def rerank(query: str, docs: list[dict], top_n: int = 5) -> list[dict]:
    """Re-score with cross-encoder. Returns re-ranked list."""
    if not docs or not is_reranker_available():
        return docs[:top_n]
    model = get_reranker()
    if model is None:
        return docs[:top_n]
    try:
        pairs = [[query, d["content"][:2000]] for d in docs]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_n]]
    except Exception:
        return docs[:top_n]


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 15,
) -> list[dict]:
    """Merge two ranked lists via Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each list the doc appears in.
    k=60 is the standard value.
    """
    scores: dict[str, dict] = {}  # id → {doc, score}

    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        scores[doc_id] = {"doc": doc, "score": 1.0 / (k + rank + 1)}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id in scores:
            scores[doc_id]["score"] += 1.0 / (k + rank + 1)
        else:
            scores[doc_id] = {"doc": doc, "score": 1.0 / (k + rank + 1)}

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:top_k]]


# ============================================================
# Unified query: hybrid search + optional rerank
# ============================================================

def query_documents(query: str, top_k: int = 5, use_rerank: bool = True) -> list[dict]:
    """Hybrid search: BM25 + Vector → RRF merge → optional Rerank.

    Three stages:
      1. BM25 (keyword) + Vector (semantic) run in parallel
      2. RRF merges the two result lists → top_k * 3 candidates
      3. BGE Reranker re-scores → returns top_k (if available)

    Falls back gracefully: no BM25 → pure vector; no reranker → skip stage 3.
    """
    # Stage 1: parallel search
    vector_docs = _vector_search(query, top_k * 3)
    bm25_docs = _bm25_search(query, top_k * 3)

    # Stage 2: RRF merge
    if bm25_docs:
        candidates = _rrf_merge(vector_docs, bm25_docs, top_k=top_k * 3)
    else:
        candidates = vector_docs

    # Stage 3: rerank
    if use_rerank and is_reranker_available() and len(candidates) > top_k:
        candidates = rerank(query, candidates, top_n=top_k)

    return candidates[:top_k]


def _vector_search(query: str, fetch_k: int) -> list[dict]:
    """Pure vector (semantic) search via Chroma."""
    try:
        collection = get_collection()
        query_embedding = embed_texts([query])
        results = collection.query(query_embeddings=query_embedding, n_results=fetch_k)
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        return docs
    except Exception:
        return []


# ============================================================
# Seed vector store from local markdown files
# ============================================================

def seed_from_markdown(filepath: str, chunk_size: int = 800) -> int:
    """Chunk a markdown file and add to vector store. Returns number of chunks added."""
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        return 0

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return 0

    # Simple chunking: split on ## headings, then by paragraph size
    chunks = []
    for section in text.split("\n## "):
        section = section.strip()
        if not section:
            continue
        if not section.startswith("#"):
            section = "## " + section
        # Split long sections into sub-chunks
        if len(section) > chunk_size:
            paragraphs = section.split("\n\n")
            current = ""
            for p in paragraphs:
                if len(current) + len(p) < chunk_size:
                    current = (current + "\n\n" + p).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = p
            if current:
                chunks.append(current)
        else:
            chunks.append(section)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "id": f"seed_{path.stem}_{i}",
            "content": chunk,
            "metadata": {"source": str(path), "chapter_title": chunk.split("\n")[0][:100]},
        })

    add_documents(docs)
    return len(docs)
