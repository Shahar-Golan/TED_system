"""
News Agent
Expert in detailed opinions and comprehensive responses from news coverage.
Searches the politics-news Pinecone index for relevant articles.
"""

import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_tools.news_search import news_search

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

llm = ChatOpenAI(
    model=os.environ.get("GPT_MODEL", "RPRTHPB-gpt-5-mini"),
    base_url=os.environ.get("BASE_URL", "https://api.llmod.ai/v1"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=1,
    max_tokens=2000,
)

NEWS_SYSTEM_PROMPT = """You are a political news analyst. You specialize in analyzing
local US news coverage of public figures — how newspapers, radio, and TV report on
politicians' actions, policies, and controversies.

Given the user's question and relevant news articles retrieved from the database,
provide a detailed, comprehensive answer. Follow these rules:

- Start with a short direct answer in 1-2 sentences
- Keep total length concise (target 120-180 words)
- Use at most 4 bullet points after the short answer
- Cite only the most relevant 1-3 sources inline (outlet + state + date)
- Highlight regional differences only if clearly evident
- Use ONLY the provided article data — do not use external knowledge
- If articles are not relevant: "I don't have news coverage addressing this topic."
- Do NOT append sections like "Articles", "Sources", "Raw sources", or "X sources"
- Do NOT repeat full article snippets that are already shown in source cards"""


def _strip_duplicate_source_dump(answer: str) -> str:
    """Remove accidental trailing source/article dump from the textual answer."""
    if not answer:
        return answer

    patterns = [
        r"\n(?:Articles|Article Sources|Sources|Raw Sources?)\s*\n",
        r"\n\d+\s+sources\s*\n",
    ]

    cleaned = answer
    for pattern in patterns:
        parts = re.split(pattern, cleaned, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            cleaned = parts[0].rstrip()

    return cleaned.strip()


def _limit_answer_length(answer: str, max_chars: int = 900) -> str:
    """Apply a soft character budget to keep responses concise in the UI."""
    if not answer or len(answer) <= max_chars:
        return answer

    clipped = answer[:max_chars].rstrip()
    last_sentence_end = max(
        clipped.rfind(". "),
        clipped.rfind("! "),
        clipped.rfind("? "),
    )
    if last_sentence_end > 300:
        return clipped[: last_sentence_end + 1].strip()
    return clipped.strip() + "..."


def run_news_agent(query: str, top_k: int = 7, on_token=None) -> dict:
    """
    Search for news articles and synthesize a response.

    Returns:
        dict: {"answer": str, "articles": list, "agent": "news_agent"}
    """
    # Search for news articles
    search_result = news_search(query, top_k=top_k)

    if not search_result["success"]:
        return {
            "answer": f"News search failed: {search_result['error']}",
            "articles": [],
            "agent": "news_agent",
        }

    articles = search_result["results"]

    if not articles:
        return {
            "answer": "No relevant news articles found for this query.",
            "articles": [],
            "agent": "news_agent",
        }

    # Build context from articles (sorted by date)
    sorted_articles = sorted(articles, key=lambda a: a["metadata"].get("date", ""))
    context = ""
    for a in sorted_articles[:5]:
        meta = a["metadata"]
        context += (
            f"Title: {meta.get('title', 'N/A')}\n"
            f"Source: {meta.get('media_name', 'Unknown')} ({meta.get('state', '')})\n"
            f"Date: {meta.get('date', 'Unknown')}\n"
            f"Type: {meta.get('media_type', 'Unknown')}\n"
            f"Speakers mentioned: {', '.join(meta.get('speakers_mentioned', []))}\n"
            f"Text: {meta.get('text', '')}\n\n"
        )

    # Synthesize answer
    messages = [
        {"role": "system", "content": NEWS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    try:
        if on_token:
            answer = ""
            for chunk in llm.stream(messages):
                token = chunk.content or ""
                answer += token
                on_token(token)
            answer = answer.strip()
        else:
            response = llm.invoke(messages)
            answer = response.content.strip()

        answer = _strip_duplicate_source_dump(answer)
        answer = _limit_answer_length(answer)

        if not answer:
            answer = "I found relevant coverage, but I could not synthesize a concise answer for this run."
    except Exception as e:
        answer = f"Error generating news analysis: {e}"

    return {
        "answer": answer,
        "articles": sorted_articles[:5],
        "agent": "news_agent",
    }
