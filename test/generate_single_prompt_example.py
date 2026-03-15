"""Generate one prompt example JSON from the real System B multi-agent flow.

Usage:
    python test/generate_single_prompt_example.py
    python test/generate_single_prompt_example.py --prompt "What does Joe Biden policy about Palestine?"
    python test/generate_single_prompt_example.py --output test/joe_biden_palestine_example.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.news_agent import run_news_agent
from agents.page_lookup import lookup_page
from agents.router import route_query
from agents.tweet_agent import run_tweet_agent


def build_prompt_example(query: str) -> dict[str, Any]:
    """Run the System B pipeline and return one organized prompt example."""
    steps: list[dict[str, Any]] = []

    page_result = lookup_page(query)
    steps.append(
        {
            "module": "Page Lookup",
            "prompt": {
                "query": query,
                "task": "Check if cached figure page can answer directly.",
            },
            "response": page_result,
        }
    )

    if page_result.get("found"):
        full_response = str(page_result.get("content") or "")
        return {
            "prompt": query,
            "full_response": full_response,
            "steps": steps,
        }

    page_context = str(page_result.get("content") or "")
    route_result = route_query(query, page_context=page_context)
    route = str(route_result.get("route") or "tweet_agent")

    steps.append(
        {
            "module": "Router",
            "prompt": {
                "query": query,
                "page_context_available": bool(page_context),
            },
            "response": route_result,
        }
    )

    if route == "news_agent":
        news_result = run_news_agent(query)
        steps.append(
            {
                "module": "News Agent",
                "prompt": {
                    "query": query,
                    "source": "politics-news index",
                },
                "response": {
                    "articles_count": len(news_result.get("articles") or []),
                    "agent": news_result.get("agent"),
                    "result": news_result.get("answer"),
                },
            }
        )
        full_response = str(news_result.get("answer") or "")

    elif route == "both":
        tweet_result = run_tweet_agent(query)
        news_result = run_news_agent(query)

        steps.append(
            {
                "module": "Tweet Agent",
                "prompt": {
                    "query": query,
                    "source": "politics tweet index",
                },
                "response": {
                    "tweets_count": len(tweet_result.get("tweets") or []),
                    "agent": tweet_result.get("agent"),
                    "result": tweet_result.get("answer"),
                },
            }
        )
        steps.append(
            {
                "module": "News Agent",
                "prompt": {
                    "query": query,
                    "source": "politics-news index",
                },
                "response": {
                    "articles_count": len(news_result.get("articles") or []),
                    "agent": news_result.get("agent"),
                    "result": news_result.get("answer"),
                },
            }
        )

        full_response = (
            "## From Tweets (direct statements)\n\n"
            f"{tweet_result.get('answer', '')}\n\n"
            "---\n\n"
            "## From News Coverage\n\n"
            f"{news_result.get('answer', '')}"
        )

    else:
        tweet_result = run_tweet_agent(query)
        steps.append(
            {
                "module": "Tweet Agent",
                "prompt": {
                    "query": query,
                    "source": "politics tweet index",
                },
                "response": {
                    "tweets_count": len(tweet_result.get("tweets") or []),
                    "agent": tweet_result.get("agent"),
                    "result": tweet_result.get("answer"),
                },
            }
        )
        full_response = str(tweet_result.get("answer") or "")

    return {
        "prompt": query,
        "full_response": full_response,
        "steps": steps,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate one prompt example JSON.")
    parser.add_argument(
        "--prompt",
        default="What does Joe Biden policy about Palestine?",
        help="Prompt text to execute.",
    )
    parser.add_argument(
        "--output",
        default="test/joe_biden_palestine_example.txt",
        help="Output .txt path for the generated JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    result = build_prompt_example(args.prompt)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    output_path.write_text(json_output + "\n", encoding="utf-8")

    print(json_output)
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
