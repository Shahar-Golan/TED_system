"""
services.speaker_profile_enrichment
=====================================
Post-ingestion enrichment stage that updates ``speaker_profiles`` in Supabase
based on newly ingested RSS articles.

After articles are pushed to Supabase's ``news_articles`` table, this module:

1. Identifies politicians mentioned in each article (via ``PoliticianMention``
   records produced by the extractor pipeline).
2. Maps each mention to a ``speaker_profiles`` row using a deterministic ID
   mapping and an optional name-based fallback.
3. Extracts explicit role information from article text using pattern matching.
4. Updates ``current_role`` and ``profile.bio.current_role`` when justified by
   strong article evidence.
5. Adds or updates ``profile.recent_news`` with provenance to the new article.

---

Sync policy — Option A (SQL is source of truth)
-----------------------------------------------
The canonical current role is computed from article evidence, then:

* Written to the SQL column ``current_role``.
* Mirrored to ``profile.bio.current_role`` in the same atomic update.

This guarantees the two fields never drift silently.

Role update rules
-----------------
* Only update when the article contains strong, explicit role evidence.
* Strong roles: specific office/title (President, Minister of X, Mayor of X, …).
* Weak roles: vague labels (politician, public figure, activist).
* Do **not** overwrite a precise existing role with a vague one.
* Do **not** downgrade a role due to ambiguous or historical wording.

``recent_news`` configuration
------------------------------
* Max retained items: 10
* Recency window: 90 days
* Dedup heuristic: normalised headline prefix (first 60 characters).
* Same-development refresh: newer article replaces older item if their
  normalised headline prefixes match.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.adapters.supabase_export import SupabaseRecord
from src.extractor.models import PoliticianMention, RelevanceLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Maximum number of ``recent_news.items`` retained per speaker profile.
MAX_RECENT_NEWS_ITEMS: int = 10

#: Rolling recency window for recent-news items.
RECENT_NEWS_WINDOW_DAYS: int = 90

#: Minimum relevance score for a mention to trigger an enrichment write.
MIN_ENRICHMENT_RELEVANCE_SCORE: float = 0.05

#: Minimum confidence threshold to write a speaker match.
MIN_MATCH_CONFIDENCE: float = 0.7

#: Number of leading characters used for headline dedup comparison.
HEADLINE_DEDUP_PREFIX_LEN: int = 60

# ---------------------------------------------------------------------------
# Typed domain models
# ---------------------------------------------------------------------------


@dataclass
class SpeakerMatchResult:
    """Result of matching a politician mention to a ``speaker_profiles`` row.

    Attributes:
        speaker_id: The ``speaker_id`` value in the ``speaker_profiles`` table.
        name: Canonical name as stored in ``speaker_profiles``.
        confidence: Match confidence in the range 0.0–1.0.
        match_reason: Human-readable explanation of how the match was made.
    """

    speaker_id: str
    name: str
    confidence: float
    match_reason: str


@dataclass
class ResolvedRoleUpdate:
    """Outcome of a role resolution comparison.

    Attributes:
        new_role: The role string to write, if ``should_update`` is ``True``.
        existing_role: The existing ``current_role`` value (may be ``None``).
        should_update: Whether the stored role should be replaced.
        reason: Human-readable explanation for the decision.
    """

    new_role: str
    existing_role: str | None
    should_update: bool
    reason: str


@dataclass
class RecentNewsItem:
    """A single item inside ``profile.recent_news.items``.

    Attributes:
        date: ISO 8601 date string for the article's publication date.
        headline: Article headline.
        summary: Short summary of the article (~200 chars).
        significance: Importance label derived from the relevance level.
        source_article_id: The ``doc_id`` of the backing ``news_articles`` row.
    """

    date: str
    headline: str
    summary: str
    significance: str
    source_article_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "date": self.date,
            "headline": self.headline,
            "summary": self.summary,
            "significance": self.significance,
            "source_article_id": self.source_article_id,
        }


@dataclass
class RecentNewsPayload:
    """The full ``profile.recent_news`` JSON object.

    Attributes:
        summary: A compact narrative summary of recent developments.
        last_updated: ISO 8601 timestamp of the last enrichment write.
        date_range: Human-readable range, e.g. ``"2025-01-01 – 2025-03-15"``.
        source_article_ids: Deduplicated list of all backing ``doc_id`` values.
        items: Ordered list of individual recent-news items (newest first,
            capped at :data:`MAX_RECENT_NEWS_ITEMS`).
    """

    summary: str
    last_updated: str
    date_range: str
    source_article_ids: list[str]
    items: list[RecentNewsItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "last_updated": self.last_updated,
            "date_range": self.date_range,
            "source_article_ids": self.source_article_ids,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class SpeakerProfileUpdate:
    """Computed update payload for a single speaker profile.

    Attributes:
        speaker_id: Target ``speaker_profiles.speaker_id``.
        role_update: Role resolution outcome.
        recent_news: Updated ``recent_news`` payload.
    """

    speaker_id: str
    role_update: ResolvedRoleUpdate
    recent_news: RecentNewsPayload


@dataclass
class EnrichmentStats:
    """Aggregate statistics for a single enrichment run.

    Attributes:
        articles_processed: Total articles examined.
        matches_found: Speaker-to-profile matches above confidence threshold.
        ambiguous_skipped: Matches skipped due to ambiguity or low confidence.
        role_updates: Number of ``current_role`` writes.
        recent_news_updates: Number of ``profile.recent_news`` writes.
        no_op_dedup: Writes skipped because content was already present.
        errors: Number of unexpected errors during writes.
    """

    articles_processed: int = 0
    matches_found: int = 0
    ambiguous_skipped: int = 0
    role_updates: int = 0
    recent_news_updates: int = 0
    no_op_dedup: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Role-strength tables
# ---------------------------------------------------------------------------

# Ordered list of (regex_pattern, canonical_label_or_None).
# When canonical_label is None the matched text itself is used as the role.
# These patterns are tried against the article title and surrounding context.
_STRONG_ROLE_PATTERNS: list[tuple[str, str | None]] = [
    (r"President of the United States", "President of the United States"),
    (r"Vice President of the United States", "Vice President of the United States"),
    (r"Prime Minister", "Prime Minister"),
    (r"Deputy Prime Minister", "Deputy Prime Minister"),
    (r"Chancellor", "Chancellor"),
    (r"President", "President"),
    (r"Vice President", "Vice President"),
    (r"Secretary of State", "Secretary of State"),
    (r"Secretary of Defense", "Secretary of Defense"),
    (r"Attorney General", "Attorney General"),
    (r"Chief Justice", "Chief Justice"),
    (r"Speaker of the House", "Speaker of the House"),
    (r"Majority Leader", "Majority Leader"),
    (r"Minority Leader", "Minority Leader"),
    (r"Opposition Leader", "Opposition Leader"),
    (r"Minister of [\w ]{2,40}", None),  # "Minister of Defense"
    (r"Secretary of [\w ]{2,40}", None),  # "Secretary of Energy"
    (r"Governor of [\w ]{2,40}", None),  # "Governor of Texas"
    (r"Governor", "Governor"),
    (r"Mayor of [\w ]{2,40}", None),  # "Mayor of New York"
    (r"Mayor", "Mayor"),
    (r"Senator", "Senator"),
    (r"Representative", "Representative"),
    (r"\bMK\b", "MK"),  # Member of Knesset
]

# Roles that are considered vague/weak and should not overwrite a specific role.
_WEAK_ROLE_TOKENS: frozenset[str] = frozenset(
    {
        "politician",
        "public figure",
        "activist",
        "businessman",
        "businesswoman",
        "entrepreneur",
        "celebrity",
        "figure",
        "person",
        "official",
    }
)


def _role_strength(role: str) -> int:
    """Return an integer strength score for *role*.

    Returns:
        ``0`` for empty / unknown, ``1`` for weak/vague, ``2`` for strong/specific.
    """
    if not role or not role.strip():
        return 0
    normalized = role.strip().lower()
    if normalized in _WEAK_ROLE_TOKENS:
        return 1
    # Anything matching a strong pattern has strength 2.
    for pattern, _ in _STRONG_ROLE_PATTERNS:
        if re.search(pattern, role, re.IGNORECASE):
            return 2
    # Generic non-empty roles get strength 1 (better than nothing, not strong).
    return 1


# ---------------------------------------------------------------------------
# Speaker matching
# ---------------------------------------------------------------------------

#: Deterministic mapping from politician_id (dashes) to speaker_id (underscores).
#: Covers all politicians tracked in config/politicians.yaml.
_POLITICIAN_ID_TO_SPEAKER_ID: dict[str, str] = {
    "donald-trump": "donald_trump",
    "joe-biden": "joe_biden",
    "hillary-clinton": "hillary_clinton",
    "barack-obama": "barack_obama",
    "kamala-harris": "kamala_harris",
    "elon-musk": "elon_musk",
    "bill-gates": "bill_gates",
    "mark-zuckerberg": "mark_zuckerberg",
}


def _politician_id_to_speaker_id(politician_id: str) -> str:
    """Convert a politician slug (dashes) to a speaker_profiles id (underscores).

    Falls back to a generic ``str.replace`` conversion for politicians not
    listed in the hardcoded map.

    Args:
        politician_id: Slug from ``config/politicians.yaml``, e.g. ``"donald-trump"``.

    Returns:
        The corresponding ``speaker_profiles.speaker_id``, e.g. ``"donald_trump"``.
    """
    return _POLITICIAN_ID_TO_SPEAKER_ID.get(
        politician_id, politician_id.replace("-", "_")
    )


def match_speaker(
    politician_id: str,
    politician_name: str,
    supabase_client: Any,
) -> SpeakerMatchResult | None:
    """Attempt to match a politician mention to a ``speaker_profiles`` row.

    Strategy:
    1. Convert ``politician_id`` to a candidate ``speaker_id`` using the
       deterministic dash-to-underscore mapping.
    2. Query ``speaker_profiles`` for that ``speaker_id``.
    3. If found, return a high-confidence match (``confidence=1.0``).
    4. If not found, fall back to a name-normalised lookup (``confidence=0.8``).
    5. If still not found, return ``None``.

    Ambiguous matches (multiple rows for a name lookup) are logged and skipped.

    Args:
        politician_id: Slug identifier from the extractor pipeline.
        politician_name: Canonical name of the politician.
        supabase_client: An initialised Supabase Python client.

    Returns:
        A :class:`SpeakerMatchResult` if a confident match was found,
        otherwise ``None``.
    """
    candidate_id = _politician_id_to_speaker_id(politician_id)

    try:
        resp = (
            supabase_client.table("speaker_profiles")
            .select("speaker_id, name")
            .eq("speaker_id", candidate_id)
            .execute()
        )
        rows = resp.data or []
    except Exception:
        logger.exception(
            "DB error while matching politician %s (speaker_id=%s).",
            politician_name,
            candidate_id,
        )
        return None

    if rows:
        row = rows[0]
        logger.debug(
            "Matched politician %s → speaker_id=%s (exact id match).",
            politician_name,
            row["speaker_id"],
        )
        return SpeakerMatchResult(
            speaker_id=row["speaker_id"],
            name=row["name"],
            confidence=1.0,
            match_reason="exact_id",
        )

    # Fallback: normalised name match
    try:
        resp = (
            supabase_client.table("speaker_profiles")
            .select("speaker_id, name")
            .ilike("name", politician_name)
            .execute()
        )
        rows = resp.data or []
    except Exception:
        logger.exception(
            "DB error during name fallback for politician %s.", politician_name
        )
        return None

    if len(rows) == 1:
        row = rows[0]
        logger.debug(
            "Matched politician %s → speaker_id=%s (name fallback).",
            politician_name,
            row["speaker_id"],
        )
        return SpeakerMatchResult(
            speaker_id=row["speaker_id"],
            name=row["name"],
            confidence=0.8,
            match_reason="name_match",
        )

    if len(rows) > 1:
        matched_names = [r["name"] for r in rows]
        logger.warning(
            "Ambiguous name match for politician %s: found %d candidates (%s). "
            "Skipping enrichment to avoid unsafe write.",
            politician_name,
            len(rows),
            ", ".join(matched_names),
        )
        return None

    logger.debug(
        "No speaker_profiles row found for politician %s (speaker_id=%s).",
        politician_name,
        candidate_id,
    )
    return None


# ---------------------------------------------------------------------------
# Profile fetch
# ---------------------------------------------------------------------------


def fetch_speaker_row(
    speaker_id: str,
    supabase_client: Any,
) -> dict[str, Any] | None:
    """Fetch a full ``speaker_profiles`` row by ``speaker_id``.

    Args:
        speaker_id: The ``speaker_profiles.speaker_id`` to look up.
        supabase_client: An initialised Supabase Python client.

    Returns:
        A dict with keys ``speaker_id``, ``name``, ``current_role``,
        ``profile``, or ``None`` if the row is not found or an error occurs.
    """
    try:
        resp = (
            supabase_client.table("speaker_profiles")
            .select("speaker_id, name, current_role, profile")
            .eq("speaker_id", speaker_id)
            .execute()
        )
        rows = resp.data or []
    except Exception:
        logger.exception("DB error fetching profile for speaker_id=%s.", speaker_id)
        return None

    if not rows:
        logger.warning("No speaker_profiles row for speaker_id=%s.", speaker_id)
        return None

    return rows[0]


# ---------------------------------------------------------------------------
# Role extraction and resolution
# ---------------------------------------------------------------------------


def extract_role_from_article(
    body: str,
    title: str,
    politician_name: str,
    aliases: list[str] | None = None,
) -> str | None:
    """Scan article title and body for explicit role mentions near *politician_name*.

    Searches for ``ROLE NAME`` and ``NAME, ROLE`` patterns within a window of
    text around each politician name/alias occurrence.

    Args:
        body: Clean extracted article body text.
        title: Article headline.
        politician_name: Canonical name to search for.
        aliases: Optional list of name aliases for additional matching.

    Returns:
        The strongest extracted role string, or ``None`` if no strong role was
        found nearby.
    """
    search_names = [politician_name] + (aliases or [])
    # Combine title and body; title matches are especially reliable
    full_text = f"{title} {body}"

    best_role: str | None = None
    best_strength = 0

    for name in search_names:
        escaped_name = re.escape(name)
        # Extract a ±300-char window around each occurrence of the politician's name
        for match in re.finditer(escaped_name, full_text, re.IGNORECASE):
            start = max(0, match.start() - 300)
            end = min(len(full_text), match.end() + 300)
            window = full_text[start:end]

            # Pattern 1: ROLE [A/THE] NAME  (title-before-name)
            for role_pattern, canonical in _STRONG_ROLE_PATTERNS:
                pat = r"\b(" + role_pattern + r")\s+(?:the\s+|a\s+)?" + escaped_name
                m = re.search(pat, window, re.IGNORECASE)
                if m:
                    role = canonical if canonical else m.group(1).strip()
                    # Trim any trailing junk
                    role = _trim_role(role)
                    strength = _role_strength(role)
                    if strength > best_strength:
                        best_role = role
                        best_strength = strength

            # Pattern 2: NAME, [the/a] ROLE  (appositive)
            pat2 = escaped_name + r"\s*,\s*(?:the\s+|a\s+)?([A-Z][a-z]+(?:\s+[a-zA-Z]+){0,4})"
            m2 = re.search(pat2, window)
            if m2:
                role = _trim_role(m2.group(1).strip())
                strength = _role_strength(role)
                if strength > best_strength:
                    best_role = role
                    best_strength = strength

    return best_role if best_strength >= 2 else None


def _trim_role(role: str) -> str:
    """Strip trailing punctuation and excess whitespace from a role string."""
    return re.sub(r"[\s,.:;]+$", "", role).strip()


def resolve_role_update(
    article_role: str | None,
    existing_role: str | None,
) -> ResolvedRoleUpdate:
    """Decide whether to replace the stored ``current_role`` with *article_role*.

    Rules:

    * If no role was extracted from the article → no update.
    * If the extracted role is weak (strength ≤ 1) → no update.
    * If the existing role is stronger or equal → no update (no downgrade).
    * Otherwise → update.

    Args:
        article_role: Role extracted from the article, or ``None``.
        existing_role: Current value of ``speaker_profiles.current_role``.

    Returns:
        A :class:`ResolvedRoleUpdate` with a decision and explanation.
    """
    if not article_role:
        return ResolvedRoleUpdate(
            new_role="",
            existing_role=existing_role,
            should_update=False,
            reason="No role evidence found in article.",
        )

    article_strength = _role_strength(article_role)
    if article_strength < 2:
        return ResolvedRoleUpdate(
            new_role=article_role,
            existing_role=existing_role,
            should_update=False,
            reason=f"Article role '{article_role}' is too vague (strength={article_strength}).",
        )

    existing_strength = _role_strength(existing_role or "")
    if existing_strength >= article_strength and existing_role:
        return ResolvedRoleUpdate(
            new_role=article_role,
            existing_role=existing_role,
            should_update=False,
            reason=(
                f"Existing role '{existing_role}' (strength={existing_strength}) is "
                f"at least as specific as article role '{article_role}' "
                f"(strength={article_strength}). No downgrade."
            ),
        )

    return ResolvedRoleUpdate(
        new_role=article_role,
        existing_role=existing_role,
        should_update=True,
        reason=(
            f"Article role '{article_role}' (strength={article_strength}) is more "
            f"specific than existing role '{existing_role}' (strength={existing_strength})."
        ),
    )


# ---------------------------------------------------------------------------
# Recent-news construction and merge
# ---------------------------------------------------------------------------


def _significance_label(relevance: RelevanceLevel) -> str:
    """Map a :class:`~extractor.models.RelevanceLevel` to a significance label."""
    mapping = {
        RelevanceLevel.PRIMARY: "primary subject",
        RelevanceLevel.SECONDARY: "significant mention",
        RelevanceLevel.INCIDENTAL: "brief mention",
        RelevanceLevel.IRRELEVANT: "tangential",
    }
    return mapping.get(relevance, "mentioned")


def _normalize_headline_for_dedup(headline: str) -> str:
    """Return a normalised headline prefix used for near-duplicate detection.

    Lowercases, strips punctuation, and truncates to
    :data:`HEADLINE_DEDUP_PREFIX_LEN` characters.
    """
    normalized = re.sub(r"[^\w\s]", "", headline.lower()).strip()
    return normalized[:HEADLINE_DEDUP_PREFIX_LEN]


def build_recent_news_item(
    record: SupabaseRecord,
    mention: PoliticianMention,
) -> RecentNewsItem:
    """Construct a :class:`RecentNewsItem` from a Supabase record and mention.

    Args:
        record: The ingested news article record.
        mention: The politician mention record for this article.

    Returns:
        A :class:`RecentNewsItem` ready for insertion into ``recent_news.items``.
    """
    # Build a short summary from the article body (first 200 chars)
    summary = (record.text or "")[:200].strip()
    if len(record.text or "") > 200:
        summary += "…"

    return RecentNewsItem(
        date=record.date or "",
        headline=record.title or "",
        summary=summary,
        significance=_significance_label(mention.relevance),
        source_article_id=record.doc_id,
    )


def merge_recent_news(
    existing: dict[str, Any] | None,
    new_item: RecentNewsItem,
) -> RecentNewsPayload:
    """Merge *new_item* into the existing ``profile.recent_news`` payload.

    Merge rules:

    * If an existing item has the same normalised headline prefix (same
      development), the older item is **replaced** with the new one.
    * Otherwise the new item is **prepended** (newest-first order).
    * Items outside the :data:`RECENT_NEWS_WINDOW_DAYS` recency window are
      dropped.
    * The retained list is capped at :data:`MAX_RECENT_NEWS_ITEMS`.

    Args:
        existing: The current ``profile.recent_news`` dict, or ``None`` if
            the field does not yet exist.
        new_item: The newly constructed :class:`RecentNewsItem`.

    Returns:
        An updated :class:`RecentNewsPayload` ready to be serialised back into
        the ``profile`` JSONB column.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=RECENT_NEWS_WINDOW_DAYS)
    cutoff_str = cutoff.date().isoformat()

    # Parse existing items, dropping those outside the recency window.
    existing_items: list[RecentNewsItem] = []
    existing_source_ids: list[str] = []

    if existing and isinstance(existing, dict):
        raw_items = existing.get("items") or []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            item_date = raw.get("date", "")
            if item_date and item_date < cutoff_str:
                continue  # expired — drop
            existing_items.append(
                RecentNewsItem(
                    date=raw.get("date", ""),
                    headline=raw.get("headline", ""),
                    summary=raw.get("summary", ""),
                    significance=raw.get("significance", ""),
                    source_article_id=raw.get("source_article_id", ""),
                )
            )
        existing_source_ids = list(existing.get("source_article_ids") or [])

    new_dedup_key = _normalize_headline_for_dedup(new_item.headline)

    # Check for same-development match (replace) or duplicate article (no-op)
    replaced = False
    merged_items: list[RecentNewsItem] = []
    for item in existing_items:
        # Exact article id match → already ingested, replace to refresh metadata
        if item.source_article_id == new_item.source_article_id:
            merged_items.append(new_item)
            replaced = True
            continue
        # Same development (near-duplicate headline) → replace with newer item
        if _normalize_headline_for_dedup(item.headline) == new_dedup_key:
            merged_items.append(new_item)
            replaced = True
            continue
        merged_items.append(item)

    if not replaced:
        # Prepend — newest items first
        merged_items = [new_item] + merged_items

    # Enforce recency cap
    merged_items = merged_items[:MAX_RECENT_NEWS_ITEMS]

    # Rebuild source_article_ids (deduplicated, preserving order)
    seen_ids: set[str] = set()
    source_ids: list[str] = []
    for item in merged_items:
        if item.source_article_id not in seen_ids:
            source_ids.append(item.source_article_id)
            seen_ids.add(item.source_article_id)
    # Preserve any ids from existing payload not present in current items
    for sid in existing_source_ids:
        if sid not in seen_ids:
            source_ids.append(sid)
            seen_ids.add(sid)

    # Compute date range
    dates = [item.date for item in merged_items if item.date]
    if dates:
        date_range = f"{min(dates)} – {max(dates)}"
    else:
        date_range = ""

    # Build compact summary (most recent headline)
    summary = merged_items[0].headline if merged_items else ""

    return RecentNewsPayload(
        summary=summary,
        last_updated=now.isoformat(),
        date_range=date_range,
        source_article_ids=source_ids,
        items=merged_items,
    )


# ---------------------------------------------------------------------------
# Safe profile JSON merge
# ---------------------------------------------------------------------------


def merge_profile_update(
    existing_profile: dict[str, Any],
    update: SpeakerProfileUpdate,
) -> dict[str, Any]:
    """Return a new profile dict with enrichment updates applied.

    Only the following keys are touched:

    * ``profile.recent_news`` — always updated.
    * ``profile.bio.current_role`` — updated only when
      ``update.role_update.should_update`` is ``True``.

    All other top-level keys (``bio``, ``controversies``, ``media_profile``,
    ``relationships``, ``notable_topics``, ``dataset_insights``,
    ``public_perception``, ``timeline_highlights``) are preserved verbatim.

    Args:
        existing_profile: The current ``profile`` JSONB value (a Python dict).
        update: The computed update payload.

    Returns:
        A new dict representing the merged profile.

    Raises:
        ValueError: If *existing_profile* is not a dict.
    """
    if not isinstance(existing_profile, dict):
        raise ValueError(
            f"existing_profile must be a dict, got {type(existing_profile).__name__}"
        )

    # Shallow copy to avoid mutating the input
    merged = dict(existing_profile)

    # Update bio.current_role when role resolution says to
    if update.role_update.should_update and update.role_update.new_role:
        bio = dict(merged.get("bio") or {})
        bio["current_role"] = update.role_update.new_role
        merged["bio"] = bio
        logger.info(
            "speaker_id=%s | profile.bio.current_role updated: '%s' → '%s'",
            update.speaker_id,
            update.role_update.existing_role,
            update.role_update.new_role,
        )

    # Always update recent_news
    merged["recent_news"] = update.recent_news.to_dict()

    return merged


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_speaker_update(
    update: SpeakerProfileUpdate,
    existing_row: dict[str, Any],
    supabase_client: Any,
) -> bool:
    """Write the enrichment update for one speaker profile to Supabase.

    Builds a merged profile, then issues a single ``UPDATE`` covering:

    * ``current_role`` — only if :attr:`~ResolvedRoleUpdate.should_update`.
    * ``profile`` — always (contains the updated ``recent_news`` and possibly
      updated ``bio.current_role``).
    * ``updated_at`` — always set to the current UTC timestamp.

    Args:
        update: The computed update payload.
        existing_row: The current ``speaker_profiles`` row dict (with ``profile``
            and ``current_role`` keys).
        supabase_client: An initialised Supabase Python client.

    Returns:
        ``True`` if the update was persisted successfully, ``False`` otherwise.
    """
    existing_profile = existing_row.get("profile") or {}
    if not isinstance(existing_profile, dict):
        # Guard: profile column might be a JSON string in some clients
        import json as _json

        try:
            existing_profile = _json.loads(existing_profile)
        except Exception:
            logger.error(
                "speaker_id=%s | Cannot parse existing profile JSON. Skipping.",
                update.speaker_id,
            )
            return False

    try:
        merged_profile = merge_profile_update(existing_profile, update)
    except ValueError as exc:
        logger.error(
            "speaker_id=%s | Profile merge failed: %s. Skipping.",
            update.speaker_id,
            exc,
        )
        return False

    patch: dict[str, Any] = {
        "profile": merged_profile,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if update.role_update.should_update and update.role_update.new_role:
        patch["current_role"] = update.role_update.new_role
        logger.info(
            "speaker_id=%s | current_role updated: '%s' → '%s'",
            update.speaker_id,
            update.role_update.existing_role,
            update.role_update.new_role,
        )

    try:
        supabase_client.table("speaker_profiles").update(patch).eq(
            "speaker_id", update.speaker_id
        ).execute()
        return True
    except Exception:
        logger.exception(
            "speaker_id=%s | Supabase update failed.", update.speaker_id
        )
        return False


# ---------------------------------------------------------------------------
# Per-article enrichment
# ---------------------------------------------------------------------------


def enrich_from_article(
    record: SupabaseRecord,
    mentions: list[PoliticianMention],
    supabase_client: Any,
    stats: EnrichmentStats,
    politician_aliases: dict[str, list[str]] | None = None,
) -> None:
    """Process one article and update all matching speaker profiles.

    For each mention that passes the relevance threshold:

    1. Match the politician to a ``speaker_profiles`` row.
    2. Fetch the current row.
    3. Extract role evidence from the article.
    4. Resolve the role update decision.
    5. Build the new ``recent_news`` item.
    6. Merge the item into the existing ``profile.recent_news``.
    7. Persist the merged update.

    Args:
        record: The ingested ``SupabaseRecord`` for this article.
        mentions: Politician mentions extracted during ingestion.
        supabase_client: An initialised Supabase Python client.
        stats: Mutable stats object to increment counters.
        politician_aliases: Optional mapping of politician_id → aliases list,
            used to improve role extraction accuracy.
    """
    stats.articles_processed += 1
    aliases_map = politician_aliases or {}

    for mention in mentions:
        if mention.relevance_score < MIN_ENRICHMENT_RELEVANCE_SCORE:
            continue
        if mention.relevance == RelevanceLevel.IRRELEVANT:
            continue

        # --- 1. Match ---
        match = match_speaker(
            mention.politician_id,
            mention.politician_name,
            supabase_client,
        )
        if match is None:
            logger.debug(
                "article_id=%s | No match for politician %s — skipping.",
                record.doc_id,
                mention.politician_name,
            )
            continue

        if match.confidence < MIN_MATCH_CONFIDENCE:
            logger.warning(
                "article_id=%s | Low-confidence match for %s → %s (%.2f). Skipping.",
                record.doc_id,
                mention.politician_name,
                match.speaker_id,
                match.confidence,
            )
            stats.ambiguous_skipped += 1
            continue

        stats.matches_found += 1
        logger.info(
            "article_id=%s | Matched %s → speaker_id=%s (%s, confidence=%.2f).",
            record.doc_id,
            mention.politician_name,
            match.speaker_id,
            match.match_reason,
            match.confidence,
        )

        # --- 2. Fetch current row ---
        row = fetch_speaker_row(match.speaker_id, supabase_client)
        if row is None:
            logger.warning(
                "article_id=%s | speaker_id=%s row not found after match. Skipping.",
                record.doc_id,
                match.speaker_id,
            )
            stats.errors += 1
            continue

        # --- 3 & 4. Role extraction + resolution ---
        aliases = aliases_map.get(mention.politician_id, [])
        article_role = extract_role_from_article(
            body=record.text or "",
            title=record.title or "",
            politician_name=mention.politician_name,
            aliases=aliases,
        )
        role_update = resolve_role_update(article_role, row.get("current_role"))
        logger.debug(
            "article_id=%s | speaker_id=%s | role resolution: %s",
            record.doc_id,
            match.speaker_id,
            role_update.reason,
        )

        # --- 5 & 6. Recent-news construction and merge ---
        news_item = build_recent_news_item(record, mention)
        existing_profile = row.get("profile") or {}
        if not isinstance(existing_profile, dict):
            existing_profile = {}
        existing_recent_news: dict[str, Any] | None = existing_profile.get(
            "recent_news"
        )

        # Check for true no-op: same article already in recent_news
        if existing_recent_news:
            existing_ids: list[str] = existing_recent_news.get(
                "source_article_ids", []
            )
            if record.doc_id in existing_ids and not role_update.should_update:
                logger.debug(
                    "article_id=%s | speaker_id=%s | already in recent_news; no role update needed — no-op.",
                    record.doc_id,
                    match.speaker_id,
                )
                stats.no_op_dedup += 1
                continue

        recent_news = merge_recent_news(existing_recent_news, news_item)

        # --- 7. Persist ---
        update = SpeakerProfileUpdate(
            speaker_id=match.speaker_id,
            role_update=role_update,
            recent_news=recent_news,
        )

        ok = persist_speaker_update(update, row, supabase_client)
        if ok:
            if role_update.should_update:
                stats.role_updates += 1
            stats.recent_news_updates += 1
            logger.info(
                "article_id=%s | speaker_id=%s | profile updated "
                "(role_updated=%s, recent_news_items=%d).",
                record.doc_id,
                match.speaker_id,
                role_update.should_update,
                len(recent_news.items),
            )
        else:
            stats.errors += 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def enrich_speaker_profiles(
    records: list[SupabaseRecord],
    mentions_by_article: dict[str, list[PoliticianMention]],
    supabase_url: str,
    supabase_key: str,
    politician_aliases: dict[str, list[str]] | None = None,
) -> EnrichmentStats:
    """Run the full speaker-profile enrichment stage for a batch of articles.

    This is the primary entry point called from the RSS pipeline after articles
    have been pushed to Supabase.

    For each article in *records* that has politician mentions, this function:

    * Resolves speaker-profile matches.
    * Updates ``current_role`` when strong evidence is present.
    * Synchronises ``profile.bio.current_role`` with the SQL column.
    * Builds and merges ``profile.recent_news`` items.
    * Deduplicates near-duplicate news entries.
    * Logs all decisions for auditability.

    Args:
        records: List of ingested ``SupabaseRecord`` objects (from Stage 3).
        mentions_by_article: Mapping of ``doc_id`` → list of
            ``PoliticianMention`` records produced by the extractor pipeline.
        supabase_url: Supabase project URL (``SUPABASE_URL`` env var).
        supabase_key: Supabase service role key (``SUPABASE_KEY`` env var).
        politician_aliases: Optional mapping of politician_id → aliases, used
            to improve role extraction. Sourced from ``politicians.yaml``.

    Returns:
        An :class:`EnrichmentStats` summary of the run.
    """
    try:
        from supabase import create_client  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "The 'supabase' package is required. "
            "Install it with: pip install supabase"
        )

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set for enrichment."
        )

    client = create_client(supabase_url, supabase_key)
    stats = EnrichmentStats()

    for record in records:
        mentions = mentions_by_article.get(record.doc_id, [])
        if not mentions:
            logger.debug(
                "article_id=%s | No politician mentions — skipping enrichment.",
                record.doc_id,
            )
            stats.articles_processed += 1
            continue

        enrich_from_article(
            record=record,
            mentions=mentions,
            supabase_client=client,
            stats=stats,
            politician_aliases=politician_aliases,
        )

    logger.info(
        "Enrichment complete: articles=%d, matches=%d, ambiguous_skipped=%d, "
        "role_updates=%d, recent_news_updates=%d, no_op_dedup=%d, errors=%d.",
        stats.articles_processed,
        stats.matches_found,
        stats.ambiguous_skipped,
        stats.role_updates,
        stats.recent_news_updates,
        stats.no_op_dedup,
        stats.errors,
    )

    return stats
