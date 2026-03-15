"""
services.pinecone_indexer
=========================
Reusable Pinecone indexing service for RSS-ingested news articles.

This module provides a single, standardised path for embedding and upserting
news articles into the ``politics-news`` Pinecone index using the same
embedding model, metadata structure, and batch approach already used by
``src/load_news_to_supabase_and_pinecone.py``.

**Design goals:**

* Reuse the repo's standard methodology — same index, same embedding model,
  same metadata contract, same vector-id strategy.
* Idempotent — repeated runs upsert (overwrite) rather than duplicate.
* Modular — callable from ``run_pipeline.py`` or any other ingestion path.
* Fail-safe — Pinecone errors are logged and surfaced without crashing the
  caller; partial batches are handled gracefully.

**Vector ID strategy:**

The Pinecone vector ID is set to the article's ``doc_id`` (a SHA-256 content
hash from :mod:`src.utils.hashing`).  Using the same stable ``doc_id`` means:

* Reruns produce upserts, not duplicate inserts.
* If article content changes, the caller can re-index and the old vector is
  cleanly replaced.
* The vector ID is directly queryable against ``news_articles.doc_id`` in
  Supabase.

**Chunking strategy:**

The existing corpus uses one vector per article (no intra-article chunking).
RSS articles follow the same convention.  The embedding text is::

    f"{title}\\n\\n{body}"

truncated to 8,000 characters to stay within model token limits.

**Metadata contract (``politics-news`` index):**

::

    {
        "doc_id":             str,   # Supabase primary key
        "title":              str,   # headline (≤ 200 chars)
        "text":               str,   # body preview (≤ 500 chars)
        "date":               str,   # ISO 8601 publication date or ""
        "media_name":         str,   # publication name
        "media_type":         str,   # e.g. "rss_news"
        "state":              str,   # US state or ""
        "link":               str,   # canonical article URL
        "speakers_mentioned": list[str],
        "type":               "news_article",
    }

This matches the shape returned by :func:`src.agent_tools.news_search.news_search`
and expected by the News Agent retrieval layer.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Heavy dependencies are imported at module level with a graceful fallback
# so that the module remains importable in test/CI environments where
# pinecone/openai may not be installed.
try:
    from openai import OpenAI as _OpenAI
    from pinecone import Pinecone as _Pinecone
    _DEPS_AVAILABLE = True
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]
    _Pinecone = None  # type: ignore[assignment,misc]
    _DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match the rest of the repo
# ---------------------------------------------------------------------------

PINECONE_NEWS_INDEX: str = "politics-news"
EMBEDDING_MODEL: str = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1024
DEFAULT_BATCH_SIZE: int = 50
EMBED_TEXT_MAX_CHARS: int = 8_000  # ~2,000 tokens — well within model limit
METADATA_TITLE_MAX: int = 200
METADATA_TEXT_PREVIEW_MAX: int = 500


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class PineconeIndexResult:
    """Outcome of a single :func:`index_articles` call.

    Attributes:
        upserted: Number of vectors successfully upserted.
        skipped: Number of records skipped (e.g. empty body).
        errors: Number of records that could not be indexed due to errors.
        error_messages: Human-readable description of each error.
    """

    upserted: int = 0
    skipped: int = 0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """``True`` if all records were upserted without errors."""
        return self.errors == 0


# ---------------------------------------------------------------------------
# Article payload type
# ---------------------------------------------------------------------------


@dataclass
class IndexableArticle:
    """Normalised article representation consumed by the Pinecone indexer.

    This is the intermediate document form that decouples the indexer from
    any specific pipeline input type (``SupabaseRecord``, ``ExtractedArticle``,
    raw dicts, etc.).

    Attributes:
        doc_id: Stable, unique content hash — used as the Pinecone vector ID.
        title: Article headline.
        body: Full article body text.
        date: Publication date string (ISO 8601) or empty string.
        media_name: Publisher / site name.
        media_type: Media category label (e.g. ``"rss_news"``).
        state: US state if known, otherwise empty string.
        link: Canonical article URL.
        speakers_mentioned: List of politician names mentioned in the article.
    """

    doc_id: str
    title: str
    body: str
    date: str
    media_name: str
    media_type: str
    state: str
    link: str
    speakers_mentioned: list[str]


# ---------------------------------------------------------------------------
# SupabaseRecord → IndexableArticle adapter
# ---------------------------------------------------------------------------


def supabase_record_to_indexable(record: Any) -> IndexableArticle:
    """Convert a :class:`~src.adapters.supabase_export.SupabaseRecord` to an
    :class:`IndexableArticle`.

    The adapter normalises the ``speakers_mentioned`` field from the
    comma-separated string stored in ``SupabaseRecord`` to a proper
    ``list[str]`` required by Pinecone metadata.

    Args:
        record: A ``SupabaseRecord`` instance.

    Returns:
        An :class:`IndexableArticle` ready for indexing.
    """
    # Parse comma-separated speakers string into a list
    speakers_raw: str = record.speakers_mentioned or ""
    speakers: list[str] = (
        [s.strip() for s in speakers_raw.split(",") if s.strip()]
        if speakers_raw
        else []
    )

    return IndexableArticle(
        doc_id=record.doc_id,
        title=record.title or "",
        body=record.text or "",
        date=record.date or "",
        media_name=record.media_name or "",
        media_type=record.media_type or "rss_news",
        state=record.state or "",
        link=record.link or "",
        speakers_mentioned=speakers,
    )


# ---------------------------------------------------------------------------
# Core indexing function
# ---------------------------------------------------------------------------


def index_articles(
    articles: list[IndexableArticle],
    *,
    pinecone_api_key: str | None = None,
    openai_api_key: str | None = None,
    base_url: str | None = None,
    index_name: str = PINECONE_NEWS_INDEX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rate_limit_delay: float = 0.2,
) -> PineconeIndexResult:
    """Embed and upsert a list of articles into the Pinecone news index.

    Follows the same embedding and upsert pattern as
    ``src/load_news_to_supabase_and_pinecone.py``:

    * Text is ``f"{title}\\n\\n{body}"`` truncated to
      :data:`EMBED_TEXT_MAX_CHARS` characters.
    * Vectors are upserted in batches of ``batch_size``.
    * Vector IDs are ``article.doc_id`` — stable, deterministic, idempotent.
    * A short delay between batches avoids hitting API rate limits.

    Args:
        articles: List of :class:`IndexableArticle` objects to index.
        pinecone_api_key: Pinecone API key.  Falls back to the
            ``PINECONE_API_KEY`` environment variable when ``None``.
        openai_api_key: OpenAI-compatible API key for the embedding model.
            Falls back to ``OPENAI_API_KEY`` env var when ``None``.
        base_url: Base URL for the OpenAI-compatible embedding API.
            Falls back to ``BASE_URL`` env var or ``https://api.llmod.ai/v1``.
        index_name: Pinecone index name.  Defaults to ``"politics-news"``.
        batch_size: Number of articles to embed and upsert per API call.
        rate_limit_delay: Seconds to sleep between batches to avoid rate
            limiting.  Defaults to ``0.2``.

    Returns:
        A :class:`PineconeIndexResult` summarising the outcome.
    """
    result = PineconeIndexResult()

    if not articles:
        logger.debug("pinecone_indexer: no articles to index, returning early.")
        return result

    # Resolve credentials
    _pinecone_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY", "")
    _openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    _base_url = (
        base_url
        or os.environ.get("BASE_URL", "")
        or "https://api.llmod.ai/v1"
    )

    if not _pinecone_key or not _openai_key:
        msg = "PINECONE_API_KEY and OPENAI_API_KEY must be set to index articles."
        logger.error("pinecone_indexer: %s", msg)
        result.errors += len(articles)
        result.error_messages.append(msg)
        return result

    # Verify heavy dependencies are available
    if not _DEPS_AVAILABLE or _Pinecone is None or _OpenAI is None:
        msg = "Required packages 'pinecone' and 'openai' must be installed."
        logger.error("pinecone_indexer: %s", msg)
        result.errors += len(articles)
        result.error_messages.append(msg)
        return result

    # Initialise clients
    pc = _Pinecone(api_key=_pinecone_key)
    openai_client = _OpenAI(api_key=_openai_key, base_url=_base_url)
    index = pc.Index(index_name)

    logger.info(
        "pinecone_indexer: starting indexing of %d article(s) into '%s'.",
        len(articles),
        index_name,
    )

    # Filter out articles with no content
    indexable: list[IndexableArticle] = []
    for art in articles:
        if not art.body:
            logger.warning(
                "pinecone_indexer: skipping %s — empty body.", art.doc_id
            )
            result.skipped += 1
        else:
            indexable.append(art)

    # Process in batches
    for batch_start in range(0, len(indexable), batch_size):
        batch = indexable[batch_start : batch_start + batch_size]

        # Build embedding texts
        texts: list[str] = []
        for art in batch:
            combined = f"{art.title}\n\n{art.body}"
            texts.append(combined[:EMBED_TEXT_MAX_CHARS])

        # Embed batch
        try:
            logger.debug(
                "pinecone_indexer: embedding batch %d (size=%d).",
                batch_start // batch_size + 1,
                len(texts),
            )
            emb_response = openai_client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
            )
        except Exception as exc:
            msg = (
                f"Embedding error at batch {batch_start // batch_size + 1}: {exc}"
            )
            logger.error("pinecone_indexer: %s", msg)
            result.errors += len(batch)
            result.error_messages.append(msg)
            continue

        # Build vector records
        vectors: list[dict[str, Any]] = []
        for i, art in enumerate(batch):
            vectors.append(
                {
                    "id": art.doc_id,
                    "values": emb_response.data[i].embedding,
                    "metadata": _build_metadata(art),
                }
            )

        # Upsert batch
        try:
            index.upsert(vectors=vectors)
            result.upserted += len(vectors)
            logger.info(
                "pinecone_indexer: upserted %d vector(s) (running total: %d).",
                len(vectors),
                result.upserted,
            )
        except Exception as exc:
            msg = (
                f"Upsert error at batch {batch_start // batch_size + 1}: {exc}"
            )
            logger.error("pinecone_indexer: %s", msg)
            result.errors += len(vectors)
            result.error_messages.append(msg)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    logger.info(
        "pinecone_indexer: finished — upserted=%d, skipped=%d, errors=%d.",
        result.upserted,
        result.skipped,
        result.errors,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_metadata(art: IndexableArticle) -> dict[str, Any]:
    """Build the Pinecone vector metadata dict for *art*.

    The shape matches the ``politics-news`` contract used by
    :func:`src.agent_tools.news_search.news_search` and
    ``src/load_news_to_supabase_and_pinecone.py``.

    Args:
        art: The article to build metadata for.

    Returns:
        A dict of Pinecone-compatible scalar/list metadata values.
    """
    return {
        "doc_id": art.doc_id,
        "title": art.title[:METADATA_TITLE_MAX],
        "text": art.body[:METADATA_TEXT_PREVIEW_MAX],
        "date": art.date or "",
        "media_name": art.media_name,
        "media_type": art.media_type,
        "state": art.state,
        "link": art.link,
        "speakers_mentioned": art.speakers_mentioned,
        "type": "news_article",
    }
