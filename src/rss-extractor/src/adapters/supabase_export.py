"""
adapters.supabase_export
========================
Schema adapter for transforming Scout/Extractor pipeline outputs into the
team's Supabase-compatible record format.

Target schema::

    id, doc_id, title, text, date, media_name, media_type, source_platform,
    state, city, link, speakers_mentioned, created_at

This module provides:

- :class:`SupabaseRecord` — typed dataclass matching the target header exactly.
- :func:`to_supabase_record` — maps an ``ExtractedArticle`` (plus optional
  mentions) into a single ``SupabaseRecord``.
- :func:`to_supabase_records` — batch conversion for multiple articles.
- :func:`records_to_dicts` — serialise records to plain dicts.
- :func:`records_to_csv` — write records as a CSV string.
"""

from __future__ import annotations

import csv
import io
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.extractor.models import ExtractedArticle, PoliticianMention

# The exact column order required by the Supabase target table.
SUPABASE_COLUMNS: list[str] = [
    "id",
    "doc_id",
    "title",
    "text",
    "date",
    "media_name",
    "media_type",
    "source_platform",
    "state",
    "city",
    "link",
    "speakers_mentioned",
    "created_at",
]


@dataclass
class SupabaseRecord:
    """A single record in the team's Supabase schema.

    Every field maps directly to a column in the target table.

    Field mapping rules:

    * ``id`` — a UUID-4 string generated per record.  Callers may supply a
      stable ``record_id`` to override the default.
    * ``doc_id`` — the pipeline's internal ``article_id`` (SHA-256 fingerprint).
    * ``title`` — extracted article headline.
    * ``text`` — cleaned article body text.
    * ``date`` — article publication date (ISO 8601) or ``None``.
    * ``media_name`` — publisher / site name, or ``""`` if unavailable.
    * ``media_type`` — normalised media category; defaults to ``"rss_news"``.
    * ``source_platform`` — fixed to ``"rss"`` for this pipeline.
    * ``state`` — US state, or ``""`` (not currently extracted).
    * ``city`` — city, or ``""`` (not currently extracted).
    * ``link`` — canonical article URL.
    * ``speakers_mentioned`` — comma-separated politician names, or ``""``.
    * ``created_at`` — ISO 8601 timestamp when this record was created.
    """

    id: str
    doc_id: str
    title: str
    text: str
    date: str | None
    media_name: str
    media_type: str
    source_platform: str
    state: str
    city: str
    link: str
    speakers_mentioned: str
    created_at: str


def to_supabase_record(
    article: ExtractedArticle,
    mentions: list[PoliticianMention] | None = None,
    *,
    media_type: str = "rss_news",
    source_platform: str = "rss",
    state: str = "",
    city: str = "",
    record_id: str | None = None,
    created_at: datetime | None = None,
) -> SupabaseRecord:
    """Convert an extracted article into a Supabase-compatible record.

    Args:
        article: An ``ExtractedArticle`` from the Extractor pipeline.
        mentions: Optional list of ``PoliticianMention`` records associated
            with this article.  Used to populate ``speakers_mentioned``.
        media_type: Media category label.  Defaults to ``"rss_news"``.
        source_platform: Source platform identifier.  Defaults to ``"rss"``.
        state: US state if known; defaults to ``""`` (not currently derived).
        city: City if known; defaults to ``""`` (not currently derived).
        record_id: Optional pre-generated string id.  If ``None``, a UUID-4
            is generated automatically.
        created_at: Optional creation timestamp.  Defaults to UTC now.

    Returns:
        A :class:`SupabaseRecord` ready for serialisation or database insert.
    """
    if record_id is None:
        record_id = str(uuid.uuid4())

    if created_at is None:
        created_at = datetime.now(tz=timezone.utc)

    # Determine publication date string
    pub_date: str | None = None
    if article.metadata.published_at is not None:
        pub_date = article.metadata.published_at.isoformat()

    # Determine media name
    media_name = article.metadata.site_name or ""

    # Build canonical link
    link = article.metadata.canonical_url or article.url

    # Build speakers_mentioned from politician mentions as comma-separated string
    speakers: list[str] = []
    if mentions:
        seen: set[str] = set()
        for m in mentions:
            if m.politician_name not in seen:
                speakers.append(m.politician_name)
                seen.add(m.politician_name)
    speakers_str = ", ".join(speakers)

    return SupabaseRecord(
        id=record_id,
        doc_id=article.article_id,
        title=article.metadata.title,
        text=article.body,
        date=pub_date,
        media_name=media_name,
        media_type=media_type,
        source_platform=source_platform,
        state=state,
        city=city,
        link=link,
        speakers_mentioned=speakers_str,
        created_at=created_at.isoformat(),
    )


def to_supabase_records(
    articles: list[ExtractedArticle],
    mentions_by_article: dict[str, list[PoliticianMention]] | None = None,
    **kwargs: str,
) -> list[SupabaseRecord]:
    """Batch-convert extracted articles into Supabase records.

    Args:
        articles: List of extracted articles to convert.
        mentions_by_article: Optional mapping of ``article_id`` to its
            ``PoliticianMention`` list.
        **kwargs: Additional keyword arguments forwarded to
            :func:`to_supabase_record` (e.g. ``media_type``, ``state``).

    Returns:
        A list of :class:`SupabaseRecord` objects, one per article.
    """
    mentions_map = mentions_by_article or {}
    return [
        to_supabase_record(
            article,
            mentions=mentions_map.get(article.article_id),
            **kwargs,  # type: ignore[arg-type]
        )
        for article in articles
    ]


def record_to_dict(record: SupabaseRecord) -> dict[str, str | None]:
    """Serialise a single ``SupabaseRecord`` to a plain dict.

    The dict keys match :data:`SUPABASE_COLUMNS` exactly and values are
    strings (or ``None`` for nullable fields).

    Args:
        record: The record to serialise.

    Returns:
        An ordered dict suitable for JSON serialisation or DB insert.
    """
    return asdict(record)


def records_to_dicts(records: list[SupabaseRecord]) -> list[dict[str, str | None]]:
    """Serialise a list of records to a list of plain dicts.

    Args:
        records: Records to serialise.

    Returns:
        A list of dicts, one per record.
    """
    return [record_to_dict(r) for r in records]


def records_to_csv(records: list[SupabaseRecord]) -> str:
    """Serialise records to a CSV string with the target header.

    The output includes a header row matching :data:`SUPABASE_COLUMNS`
    followed by one data row per record.

    Args:
        records: Records to serialise.

    Returns:
        A CSV-formatted string.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=SUPABASE_COLUMNS)
    writer.writeheader()
    for record in records:
        writer.writerow(record_to_dict(record))
    return output.getvalue()
