#!/usr/bin/env python3
"""
run_pipeline.py
===============
Full RSS extraction pipeline: poll feeds → fetch articles → extract text →
export CSV → push to Supabase.

Designed to run unattended in GitHub Actions, but works locally too.

Usage — local
-------------
::

    cd src/rss-extractor
    python run_pipeline.py
    python run_pipeline.py --dry-run          # skip Supabase push
    python run_pipeline.py --skip-poll        # skip feed polling

Usage — GitHub Actions
----------------------
Set ``SUPABASE_URL`` and ``SUPABASE_KEY`` as repository secrets, then::

    - name: Run RSS extraction pipeline
      working-directory: src/rss-extractor
      run: python run_pipeline.py
      env:
        SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
        SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}

Exit codes
----------
0   All stages completed without fatal errors.
1   A required credential is missing or a stage raised an unrecoverable error.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import textwrap
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the rss-extractor directory is importable as the package root so
# that ``from src.X import ...`` resolves correctly, whether the script is
# invoked from the project root or from inside src/rss-extractor/.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Pipeline-stage imports (after sys.path fixup)
# ---------------------------------------------------------------------------
from src.adapters.supabase_export import records_to_csv, to_supabase_record  # noqa: E402
from src.pipelines.ingest_article import ingest_article  # noqa: E402
from src.pipelines.ingest_feed import ingest_feed  # noqa: E402
from src.scout.fetcher import fetch_article  # noqa: E402
from src.storage.document_store import load_raw_html, save_raw_html  # noqa: E402
from src.storage.sql import (  # noqa: E402
    get_feed_item,
    get_feed_items_pending_fetch,
    get_raw_articles_pending_extraction,
    init_schema,
    insert_raw_article,
)
from src.utils.config import (  # noqa: E402
    load_feeds,
    load_politicians,
    load_settings,
    load_topics,
)
from src.utils.logging import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# GitHub Actions workflow command helpers
# ---------------------------------------------------------------------------

_IN_GITHUB_ACTIONS: bool = os.environ.get("GITHUB_ACTIONS") == "true"


def _gha(command: str, message: str, **params: str) -> None:
    """Emit a GitHub Actions workflow command to stdout."""
    if not _IN_GITHUB_ACTIONS:
        return
    param_str = ",".join(f"{k}={v}" for k, v in params.items())
    prefix = f"::{command} {param_str}::" if param_str else f"::{command}::"
    print(f"{prefix}{message}", flush=True)


def gha_error(message: str) -> None:
    _gha("error", message)


def gha_warning(message: str) -> None:
    _gha("warning", message)


def gha_notice(message: str) -> None:
    _gha("notice", message)


def gha_group(title: str) -> None:
    _gha("group", title)


def gha_endgroup() -> None:
    if _IN_GITHUB_ACTIONS:
        print("::endgroup::", flush=True)


def gha_set_output(name: str, value: str) -> None:
    """Write a step output to $GITHUB_OUTPUT (Actions ≥ 2022-10)."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def _write_step_summary(markdown: str) -> None:
    """Append *markdown* to $GITHUB_STEP_SUMMARY if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")


# ---------------------------------------------------------------------------
# .env loader (for local runs; GitHub Actions injects secrets as env vars)
# ---------------------------------------------------------------------------

def _load_env(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Skips blank lines and comments. Does not override variables already set
    in the environment, so GitHub Actions secrets always take precedence.
    """
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Supabase push (inlined to avoid subprocess overhead in CI)
# ---------------------------------------------------------------------------

def _push_to_supabase(
    csv_text: str,
    table: str,
    batch_size: int,
) -> tuple[int, int, int]:
    """Insert rows from *csv_text* into Supabase, skipping duplicates.

    Returns ``(uploaded, skipped, errors)`` counts.
    """
    try:
        from supabase import create_client  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "The 'supabase' package is required. "
            "Install it with: pip install supabase"
        )

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set."
        )

    rows: list[dict[str, str]] = list(csv.DictReader(StringIO(csv_text)))
    if not rows:
        return 0, 0, 0

    client = create_client(supabase_url, supabase_key)

    # Fetch existing doc_ids for deduplication
    existing_doc_ids: set[str] = set()
    offset = 0
    page_size = 1000
    while True:
        response = (
            client.table(table)
            .select("doc_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        for record in batch:
            existing_doc_ids.add(record["doc_id"])
        if len(batch) < page_size:
            break
        offset += page_size

    new_rows = [r for r in rows if r.get("doc_id") not in existing_doc_ids]
    skipped = len(rows) - len(new_rows)

    uploaded = 0
    errors = 0
    for i in range(0, len(new_rows), batch_size):
        batch = new_rows[i : i + batch_size]
        try:
            client.table(table).insert(batch).execute()
            uploaded += len(batch)
        except Exception as exc:
            errors += len(batch)
            gha_error(f"Supabase insert error on batch {i // batch_size + 1}: {exc}")
            print(f"  ERROR on batch {i // batch_size + 1}: {exc}", file=sys.stderr)

    return uploaded, skipped, errors


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Full RSS extraction pipeline.
            Stages: poll → fetch → extract → push to Supabase.
        """),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all stages but skip the final Supabase push.",
    )
    parser.add_argument(
        "--table",
        default="news_articles",
        help="Supabase table name (default: news_articles).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Records per Supabase insert batch (default: 100).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file with credentials (default: .env).",
    )
    parser.add_argument(
        "--csv-out",
        default="output.csv",
        help="Path to write the exported CSV (default: output.csv).",
    )
    parser.add_argument(
        "--skip-poll",
        action="store_true",
        help="Skip the feed-polling stage.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the article-fetching stage.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip the article-extraction stage.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (complexity is acceptable for a pipeline runner)
    args = _parse_args()

    # Credentials: .env file for local runs; env vars take precedence in CI
    _load_env(Path(args.env_file))

    configure_logging()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    gha_group("Setup")
    print("=== Politics-Contradictor RSS Pipeline ===", flush=True)

    Path("data").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect("data/tracker.db")
    conn.row_factory = sqlite3.Row
    init_schema(conn)

    settings = load_settings("config/settings.yaml")
    feeds = load_feeds("config/feeds.yaml")
    politicians = load_politicians("config/politicians.yaml")
    topics = load_topics("config/topics.yaml")
    data_dir = Path(settings.storage.data_dir)

    enabled_feeds = [f for f in feeds if f.enabled]
    print(f"Loaded {len(enabled_feeds)} enabled feed(s).", flush=True)
    print(f"Tracking {len(politicians)} politician(s).", flush=True)
    gha_endgroup()

    # -----------------------------------------------------------------------
    # Stage 1: Poll feeds
    # -----------------------------------------------------------------------
    gha_group("Stage 1 — Poll feeds")
    polled = 0
    new_items_total = 0
    if args.skip_poll:
        print("Stage 1 skipped (--skip-poll).", flush=True)
    else:
        for source in enabled_feeds:
            result = ingest_feed(source, conn, settings)
            polled += 1
            new_items_total += result.items_new
            print(
                f"  [{source.id}] {result.items_new} new item(s) "
                f"(status={result.status.value})",
                flush=True,
            )
        print(
            f"Stage 1 complete: polled {polled} feed(s), "
            f"{new_items_total} new item(s).",
            flush=True,
        )
    gha_endgroup()

    # -----------------------------------------------------------------------
    # Stage 2: Fetch articles
    # -----------------------------------------------------------------------
    gha_group("Stage 2 — Fetch articles")
    fetched_total = 0
    fetched_success = 0
    if args.skip_fetch:
        print("Stage 2 skipped (--skip-fetch).", flush=True)
    else:
        for item in get_feed_items_pending_fetch(conn):
            raw = fetch_article(item, settings)
            html_path = save_raw_html(raw.article_id, raw.html, data_dir)
            insert_raw_article(conn, raw, str(html_path))
            fetched_total += 1
            if raw.status.value == "success":
                fetched_success += 1
        failed_fetch = fetched_total - fetched_success
        if failed_fetch:
            gha_warning(f"{failed_fetch} article fetch(es) failed.")
        print(
            f"Stage 2 complete: fetched {fetched_total} article(s), "
            f"{fetched_success} succeeded.",
            flush=True,
        )
    gha_endgroup()

    # -----------------------------------------------------------------------
    # Stage 3: Extract articles
    # -----------------------------------------------------------------------
    gha_group("Stage 3 — Extract articles")
    supabase_records = []
    extracted_success = 0
    skipped_no_html = 0
    if args.skip_extract:
        print("Stage 3 skipped (--skip-extract).", flush=True)
    else:
        for raw_article in get_raw_articles_pending_extraction(conn):
            html = load_raw_html(raw_article.article_id, data_dir) or ""
            if not html:
                # Attempt one re-fetch before giving up
                feed_item = get_feed_item(conn, raw_article.feed_item_id)
                if feed_item is not None:
                    refetched = fetch_article(feed_item, settings)
                    if refetched.status.value == "success" and refetched.html:
                        save_raw_html(raw_article.article_id, refetched.html, data_dir)
                        html = refetched.html

            if not html:
                skipped_no_html += 1
                continue

            raw_article.html = html
            result = ingest_article(raw_article, conn, politicians, topics, settings)
            if result.extracted_article and result.extracted_article.body:
                record = to_supabase_record(
                    result.extracted_article, mentions=result.mentions
                )
                supabase_records.append(record)
                extracted_success += 1

        if skipped_no_html:
            gha_warning(f"{skipped_no_html} article(s) skipped (no HTML body).")
        print(
            f"Stage 3 complete: extracted {extracted_success} article(s), "
            f"{skipped_no_html} skipped.",
            flush=True,
        )
    gha_endgroup()

    # -----------------------------------------------------------------------
    # Stage 4: Export CSV
    # -----------------------------------------------------------------------
    gha_group("Stage 4 — Export CSV")
    csv_text = records_to_csv(supabase_records)
    csv_path = Path(args.csv_out)
    csv_path.write_text(csv_text, encoding="utf-8")
    print(
        f"Stage 4 complete: wrote {len(supabase_records)} record(s) → {csv_path}",
        flush=True,
    )
    gha_endgroup()

    # -----------------------------------------------------------------------
    # Stage 5: Push to Supabase
    # -----------------------------------------------------------------------
    gha_group("Stage 5 — Push to Supabase")
    uploaded = skipped_dup = push_errors = 0
    if args.dry_run:
        print(
            f"Stage 5 skipped (--dry-run). "
            f"Would have pushed {len(supabase_records)} record(s) to "
            f"'{args.table}'.",
            flush=True,
        )
        gha_notice(f"Dry-run: {len(supabase_records)} record(s) ready for Supabase.")
    elif not supabase_records:
        print("Stage 5: nothing to push (0 new records).", flush=True)
    else:
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            gha_error(
                "SUPABASE_URL and SUPABASE_KEY are not set. "
                "Add them as repository secrets."
            )
            print(
                "ERROR: SUPABASE_URL and SUPABASE_KEY must be set.\n"
                "  • GitHub Actions: add them as repository secrets.\n"
                "  • Local: create a .env file (see .env.example).",
                file=sys.stderr,
            )
            conn.close()
            sys.exit(1)

        try:
            uploaded, skipped_dup, push_errors = _push_to_supabase(
                csv_text, args.table, args.batch_size
            )
        except RuntimeError as exc:
            gha_error(str(exc))
            print(f"ERROR: {exc}", file=sys.stderr)
            conn.close()
            sys.exit(1)

        if push_errors:
            gha_error(f"{push_errors} record(s) failed to upload.")
        print(
            f"Stage 5 complete: uploaded={uploaded}, "
            f"duplicates_skipped={skipped_dup}, errors={push_errors}.",
            flush=True,
        )
    gha_endgroup()

    # -----------------------------------------------------------------------
    # Write GitHub Actions step summary
    # -----------------------------------------------------------------------
    summary_lines = [
        "## RSS Pipeline Summary",
        "",
        "| Stage | Result |",
        "|-------|--------|",
        f"| 1 · Poll feeds | {polled} feed(s), {new_items_total} new item(s) |",
        f"| 2 · Fetch articles | {fetched_success}/{fetched_total} succeeded |",
        f"| 3 · Extract articles | {extracted_success} extracted, {skipped_no_html} skipped |",
        f"| 4 · Export CSV | {len(supabase_records)} record(s) → `{csv_path}` |",
    ]
    if args.dry_run:
        summary_lines.append(
            f"| 5 · Push to Supabase | ⏭ dry-run ({len(supabase_records)} ready) |"
        )
    else:
        summary_lines.append(
            f"| 5 · Push to Supabase | {uploaded} uploaded, "
            f"{skipped_dup} duplicates, {push_errors} errors |"
        )
    _write_step_summary("\n".join(summary_lines))

    # Expose key counts as step outputs for downstream jobs
    gha_set_output("new_feed_items", str(new_items_total))
    gha_set_output("extracted_articles", str(extracted_success))
    gha_set_output("uploaded_records", str(uploaded))

    # -----------------------------------------------------------------------
    # Exit
    # -----------------------------------------------------------------------
    conn.close()
    if push_errors:
        sys.exit(1)
    print("\nPipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
