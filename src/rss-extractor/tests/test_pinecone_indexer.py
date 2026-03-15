"""
test_pinecone_indexer
=====================
Tests for the Pinecone indexing service (:mod:`src.services.pinecone_indexer`).

Coverage:

Unit tests:
- RSS article adapter (supabase_record_to_indexable)
- Metadata shaping (_build_metadata)
- Chunk / ID generation (stable across reruns)

Integration tests (fully mocked — no live Pinecone or OpenAI calls):
- New article is embedded and upserted
- Duplicate/rerun produces an upsert (no error, idempotent)
- Multiple articles are indexed in a single call
- Pinecone failure is logged and captured without raising
- Changed article content is handled by replacing the existing vector
- Missing credentials return an error result without raising
- Empty body articles are skipped
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from services.pinecone_indexer import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBED_TEXT_MAX_CHARS,
    METADATA_TEXT_PREVIEW_MAX,
    METADATA_TITLE_MAX,
    PINECONE_NEWS_INDEX,
    IndexableArticle,
    PineconeIndexResult,
    _build_metadata,
    index_articles,
    supabase_record_to_indexable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(
    doc_id: str = "abc123",
    title: str = "Senate passes infrastructure bill",
    body: str = "The Senate passed a landmark bill today.",
    date: str = "2025-10-15T14:30:00+00:00",
    media_name: str = "Reuters",
    media_type: str = "rss_news",
    state: str = "",
    link: str = "https://reuters.com/example",
    speakers_mentioned: list[str] | None = None,
) -> IndexableArticle:
    return IndexableArticle(
        doc_id=doc_id,
        title=title,
        body=body,
        date=date,
        media_name=media_name,
        media_type=media_type,
        state=state,
        link=link,
        speakers_mentioned=speakers_mentioned or [],
    )


def _make_supabase_record(
    doc_id: str = "abc123",
    title: str = "Senate passes infrastructure bill",
    text: str = "The Senate passed a landmark bill today.",
    date: str = "2025-10-15T14:30:00+00:00",
    media_name: str = "Reuters",
    media_type: str = "rss_news",
    state: str = "",
    link: str = "https://reuters.com/example",
    speakers_mentioned: str = "Joe Biden, Chuck Schumer",
) -> MagicMock:
    rec = MagicMock()
    rec.doc_id = doc_id
    rec.title = title
    rec.text = text
    rec.date = date
    rec.media_name = media_name
    rec.media_type = media_type
    rec.state = state
    rec.link = link
    rec.speakers_mentioned = speakers_mentioned
    return rec


def _make_embedding_response(count: int = 1) -> MagicMock:
    """Build a fake OpenAI embeddings response with ``count`` embeddings."""
    resp = MagicMock()
    resp.data = [
        MagicMock(embedding=[0.1] * EMBEDDING_DIMENSIONS) for _ in range(count)
    ]
    return resp


# ---------------------------------------------------------------------------
# Unit tests — supabase_record_to_indexable (adapter)
# ---------------------------------------------------------------------------


class TestSupabaseRecordToIndexable:
    """Tests for the SupabaseRecord → IndexableArticle adapter."""

    def test_doc_id_preserved(self) -> None:
        rec = _make_supabase_record(doc_id="deadbeef")
        art = supabase_record_to_indexable(rec)
        assert art.doc_id == "deadbeef"

    def test_title_preserved(self) -> None:
        rec = _make_supabase_record(title="My Headline")
        art = supabase_record_to_indexable(rec)
        assert art.title == "My Headline"

    def test_body_from_text(self) -> None:
        rec = _make_supabase_record(text="Article body here.")
        art = supabase_record_to_indexable(rec)
        assert art.body == "Article body here."

    def test_date_preserved(self) -> None:
        rec = _make_supabase_record(date="2025-01-15")
        art = supabase_record_to_indexable(rec)
        assert art.date == "2025-01-15"

    def test_media_name_preserved(self) -> None:
        rec = _make_supabase_record(media_name="CNN")
        art = supabase_record_to_indexable(rec)
        assert art.media_name == "CNN"

    def test_media_type_preserved(self) -> None:
        rec = _make_supabase_record(media_type="rss_news")
        art = supabase_record_to_indexable(rec)
        assert art.media_type == "rss_news"

    def test_link_preserved(self) -> None:
        rec = _make_supabase_record(link="https://cnn.com/article")
        art = supabase_record_to_indexable(rec)
        assert art.link == "https://cnn.com/article"

    def test_speakers_parsed_from_comma_string(self) -> None:
        rec = _make_supabase_record(speakers_mentioned="Joe Biden, Chuck Schumer")
        art = supabase_record_to_indexable(rec)
        assert art.speakers_mentioned == ["Joe Biden", "Chuck Schumer"]

    def test_speakers_empty_string_gives_empty_list(self) -> None:
        rec = _make_supabase_record(speakers_mentioned="")
        art = supabase_record_to_indexable(rec)
        assert art.speakers_mentioned == []

    def test_speakers_single_name(self) -> None:
        rec = _make_supabase_record(speakers_mentioned="Donald Trump")
        art = supabase_record_to_indexable(rec)
        assert art.speakers_mentioned == ["Donald Trump"]

    def test_speakers_strips_whitespace(self) -> None:
        rec = _make_supabase_record(speakers_mentioned="  Joe Biden ,  Kamala Harris  ")
        art = supabase_record_to_indexable(rec)
        assert art.speakers_mentioned == ["Joe Biden", "Kamala Harris"]

    def test_null_speakers_gives_empty_list(self) -> None:
        rec = _make_supabase_record()
        rec.speakers_mentioned = None
        art = supabase_record_to_indexable(rec)
        assert art.speakers_mentioned == []

    def test_null_text_gives_empty_body(self) -> None:
        rec = _make_supabase_record()
        rec.text = None
        art = supabase_record_to_indexable(rec)
        assert art.body == ""

    def test_null_date_gives_empty_string(self) -> None:
        rec = _make_supabase_record()
        rec.date = None
        art = supabase_record_to_indexable(rec)
        assert art.date == ""


# ---------------------------------------------------------------------------
# Unit tests — _build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Tests for metadata shaping (_build_metadata)."""

    def test_metadata_contains_all_required_fields(self) -> None:
        """Metadata must contain all fields expected by the news_search tool."""
        art = _make_article()
        meta = _build_metadata(art)
        required_fields = [
            "doc_id", "title", "text", "date", "media_name",
            "media_type", "state", "link", "speakers_mentioned", "type",
        ]
        for field in required_fields:
            assert field in meta, f"Missing metadata field: {field}"

    def test_type_is_news_article(self) -> None:
        art = _make_article()
        meta = _build_metadata(art)
        assert meta["type"] == "news_article"

    def test_doc_id_in_metadata(self) -> None:
        art = _make_article(doc_id="cafebabe")
        meta = _build_metadata(art)
        assert meta["doc_id"] == "cafebabe"

    def test_title_truncated_to_200_chars(self) -> None:
        long_title = "A" * 300
        art = _make_article(title=long_title)
        meta = _build_metadata(art)
        assert len(meta["title"]) == METADATA_TITLE_MAX

    def test_text_preview_truncated_to_500_chars(self) -> None:
        long_body = "B" * 1000
        art = _make_article(body=long_body)
        meta = _build_metadata(art)
        assert len(meta["text"]) == METADATA_TEXT_PREVIEW_MAX

    def test_speakers_is_list(self) -> None:
        art = _make_article(speakers_mentioned=["Joe Biden", "Chuck Schumer"])
        meta = _build_metadata(art)
        assert isinstance(meta["speakers_mentioned"], list)
        assert meta["speakers_mentioned"] == ["Joe Biden", "Chuck Schumer"]

    def test_empty_date_stored_as_empty_string(self) -> None:
        art = _make_article(date="")
        meta = _build_metadata(art)
        assert meta["date"] == ""

    def test_state_preserved(self) -> None:
        art = _make_article(state="CA")
        meta = _build_metadata(art)
        assert meta["state"] == "CA"


# ---------------------------------------------------------------------------
# Unit tests — ID stability
# ---------------------------------------------------------------------------


class TestIdStability:
    """Tests for stable, deterministic vector IDs."""

    def test_vector_id_equals_doc_id(self) -> None:
        """The vector ID must be the doc_id for idempotent upserts."""
        article = _make_article(doc_id="stable-doc-id")
        # The vector id is set to article.doc_id inside index_articles;
        # verify via direct inspection of the upsert call.
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            index_articles(
                [article],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        upserted_vectors = mock_index.upsert.call_args[1]["vectors"]
        assert upserted_vectors[0]["id"] == "stable-doc-id"

    def test_same_doc_id_on_rerun_replaces_vector(self) -> None:
        """Upserting the same doc_id twice must not raise and must call upsert twice."""
        article = _make_article(doc_id="stable-doc-id")
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            res1 = index_articles(
                [article],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )
            res2 = index_articles(
                [article],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        assert res1.upserted == 1
        assert res2.upserted == 1
        assert mock_index.upsert.call_count == 2


# ---------------------------------------------------------------------------
# Integration tests (mocked Pinecone + OpenAI)
# ---------------------------------------------------------------------------


class TestIndexArticles:
    """Integration tests for index_articles()."""

    def _run(
        self,
        articles: list[IndexableArticle],
        upsert_error: Exception | None = None,
        embed_error: Exception | None = None,
    ) -> tuple[PineconeIndexResult, MagicMock]:
        """Helper: run index_articles with mocked clients, return result + mock index."""
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            if embed_error:
                mock_openai.return_value.embeddings.create.side_effect = embed_error
            else:
                mock_openai.return_value.embeddings.create.return_value = (
                    _make_embedding_response(len(articles))
                )
            mock_index = MagicMock()
            if upsert_error:
                mock_index.upsert.side_effect = upsert_error
            mock_pc.return_value.Index.return_value = mock_index

            result = index_articles(
                articles,
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )
        return result, mock_index

    # ------------------------------------------------------------------
    # New article indexing
    # ------------------------------------------------------------------

    def test_new_article_is_upserted(self) -> None:
        """A single new article should be embedded and upserted."""
        result, mock_index = self._run([_make_article()])
        assert result.upserted == 1
        assert result.errors == 0
        assert result.skipped == 0
        mock_index.upsert.assert_called_once()

    def test_upserted_vector_has_correct_id(self) -> None:
        """The upserted vector id must equal the article doc_id."""
        art = _make_article(doc_id="my-doc-id")
        _, mock_index = self._run([art])
        vectors = mock_index.upsert.call_args[1]["vectors"]
        assert vectors[0]["id"] == "my-doc-id"

    def test_upserted_vector_has_correct_metadata(self) -> None:
        """The upserted vector metadata must match the article's fields."""
        art = _make_article(
            doc_id="doc1",
            title="My Title",
            body="My Body",
            date="2025-01-01",
            media_name="BBC",
            media_type="rss_news",
            link="https://bbc.com",
            speakers_mentioned=["Joe Biden"],
        )
        _, mock_index = self._run([art])
        vectors = mock_index.upsert.call_args[1]["vectors"]
        meta = vectors[0]["metadata"]
        assert meta["doc_id"] == "doc1"
        assert meta["title"] == "My Title"
        assert meta["media_name"] == "BBC"
        assert meta["speakers_mentioned"] == ["Joe Biden"]
        assert meta["type"] == "news_article"

    def test_upserted_vector_has_correct_embedding_length(self) -> None:
        """The embedding must have the standard 1024 dimensions."""
        _, mock_index = self._run([_make_article()])
        vectors = mock_index.upsert.call_args[1]["vectors"]
        assert len(vectors[0]["values"]) == EMBEDDING_DIMENSIONS

    # ------------------------------------------------------------------
    # Multiple articles
    # ------------------------------------------------------------------

    def test_multiple_articles_all_upserted(self) -> None:
        """All articles in a batch must be upserted."""
        arts = [
            _make_article(doc_id=f"doc-{i}", body=f"Body {i}")
            for i in range(5)
        ]
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(5)
            )
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            result = index_articles(
                arts,
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        assert result.upserted == 5
        assert result.errors == 0

    def test_batch_splitting(self) -> None:
        """Articles exceeding batch_size must be split across multiple upsert calls."""
        arts = [_make_article(doc_id=f"doc-{i}") for i in range(10)]
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            # Two batches of 5
            mock_openai.return_value.embeddings.create.side_effect = [
                _make_embedding_response(5),
                _make_embedding_response(5),
            ]
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            result = index_articles(
                arts,
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
                batch_size=5,
                rate_limit_delay=0,
            )

        assert result.upserted == 10
        assert mock_index.upsert.call_count == 2

    # ------------------------------------------------------------------
    # Empty body
    # ------------------------------------------------------------------

    def test_empty_body_article_is_skipped(self) -> None:
        """Articles with no body must be skipped, not passed to the embedder."""
        art = _make_article(body="")
        result, mock_index = self._run([art])
        assert result.skipped == 1
        assert result.upserted == 0
        mock_index.upsert.assert_not_called()

    def test_mixed_empty_and_valid_articles(self) -> None:
        """Empty-body articles are skipped; valid ones are still upserted."""
        arts = [_make_article(doc_id="empty", body=""), _make_article(doc_id="valid")]
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            result = index_articles(
                arts,
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        assert result.skipped == 1
        assert result.upserted == 1

    # ------------------------------------------------------------------
    # Pinecone failure
    # ------------------------------------------------------------------

    def test_pinecone_upsert_error_is_captured(self) -> None:
        """A Pinecone upsert error must be captured in the result, not raised."""
        result, _ = self._run(
            [_make_article()],
            upsert_error=Exception("Pinecone 500"),
        )
        assert result.errors == 1
        assert result.upserted == 0
        assert any("Pinecone 500" in m for m in result.error_messages)

    def test_pinecone_upsert_error_does_not_raise(self) -> None:
        """Pinecone errors must not propagate as exceptions to the caller."""
        try:
            result, _ = self._run(
                [_make_article()],
                upsert_error=Exception("Pinecone error"),
            )
        except Exception as exc:
            pytest.fail(f"index_articles raised unexpectedly: {exc}")
        assert result.errors > 0

    def test_embed_error_is_captured(self) -> None:
        """An embedding API error must be captured in the result, not raised."""
        result, mock_index = self._run(
            [_make_article()],
            embed_error=Exception("OpenAI timeout"),
        )
        assert result.errors == 1
        assert result.upserted == 0
        mock_index.upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Missing credentials
    # ------------------------------------------------------------------

    def test_missing_pinecone_key_returns_error(self) -> None:
        """Missing PINECONE_API_KEY must return an error result immediately."""
        # Ensure env var is absent
        env = {k: v for k, v in os.environ.items() if k != "PINECONE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = index_articles(
                [_make_article()],
                pinecone_api_key="",
                openai_api_key="ok-test",
            )
        assert result.errors > 0
        assert result.upserted == 0

    def test_missing_openai_key_returns_error(self) -> None:
        """Missing OPENAI_API_KEY must return an error result immediately."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            result = index_articles(
                [_make_article()],
                pinecone_api_key="pk-test",
                openai_api_key="",
            )
        assert result.errors > 0
        assert result.upserted == 0

    # ------------------------------------------------------------------
    # Empty input
    # ------------------------------------------------------------------

    def test_empty_articles_list_returns_zero_counts(self) -> None:
        """Passing an empty list must return a zero-count result immediately."""
        result, _ = self._run([])
        assert result.upserted == 0
        assert result.errors == 0
        assert result.skipped == 0

    # ------------------------------------------------------------------
    # Changed article content (reindexing)
    # ------------------------------------------------------------------

    def test_changed_article_content_replaces_existing_vector(self) -> None:
        """Indexing an article with the same doc_id but different content must
        succeed (upsert replaces the old vector)."""
        art_v1 = _make_article(doc_id="stable", body="Original content.")
        art_v2 = _make_article(doc_id="stable", body="Updated content.")
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.side_effect = [
                _make_embedding_response(1),
                _make_embedding_response(1),
            ]
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            res1 = index_articles(
                [art_v1],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )
            res2 = index_articles(
                [art_v2],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        assert res1.upserted == 1
        assert res2.upserted == 1
        # Both calls used the same vector ID
        id_v1 = mock_index.upsert.call_args_list[0][1]["vectors"][0]["id"]
        id_v2 = mock_index.upsert.call_args_list[1][1]["vectors"][0]["id"]
        assert id_v1 == id_v2 == "stable"

    # ------------------------------------------------------------------
    # Embed text construction
    # ------------------------------------------------------------------

    def test_embed_text_is_title_plus_body(self) -> None:
        """The text passed to the embedding model must be title + body."""
        art = _make_article(title="My Title", body="My Body")
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_pc.return_value.Index.return_value = MagicMock()

            index_articles(
                [art],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        call_args = mock_openai.return_value.embeddings.create.call_args
        input_texts = call_args[1]["input"]
        assert input_texts[0] == "My Title\n\nMy Body"

    def test_embed_text_truncated_to_max_chars(self) -> None:
        """Embed text must be truncated to EMBED_TEXT_MAX_CHARS."""
        art = _make_article(title="T", body="B" * (EMBED_TEXT_MAX_CHARS + 1000))
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_pc.return_value.Index.return_value = MagicMock()

            index_articles(
                [art],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        call_args = mock_openai.return_value.embeddings.create.call_args
        input_texts = call_args[1]["input"]
        assert len(input_texts[0]) == EMBED_TEXT_MAX_CHARS

    # ------------------------------------------------------------------
    # Correct embedding model and index name
    # ------------------------------------------------------------------

    def test_correct_embedding_model_used(self) -> None:
        """The embedding call must use the standard repo embedding model."""
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_pc.return_value.Index.return_value = MagicMock()

            index_articles(
                [_make_article()],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        call_kwargs = mock_openai.return_value.embeddings.create.call_args[1]
        assert call_kwargs["model"] == EMBEDDING_MODEL
        assert call_kwargs["dimensions"] == EMBEDDING_DIMENSIONS

    def test_correct_index_name_used(self) -> None:
        """The Pinecone Index() call must use the standard news index name."""
        with (
            patch("services.pinecone_indexer._Pinecone") as mock_pc,
            patch("services.pinecone_indexer._OpenAI") as mock_openai,
        ):
            mock_openai.return_value.embeddings.create.return_value = (
                _make_embedding_response(1)
            )
            mock_index = MagicMock()
            mock_pc.return_value.Index.return_value = mock_index

            index_articles(
                [_make_article()],
                pinecone_api_key="pk-test",
                openai_api_key="ok-test",
            )

        mock_pc.return_value.Index.assert_called_with(PINECONE_NEWS_INDEX)


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImportSmoke:
    """Smoke test verifying the pinecone_indexer module can be imported."""

    def test_import_pinecone_indexer(self) -> None:
        from services.pinecone_indexer import (  # noqa: F401
            IndexableArticle,
            PineconeIndexResult,
            index_articles,
            supabase_record_to_indexable,
        )
