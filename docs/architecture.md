# Architecture — Politics-Contradictor

## Overview

Politics-Contradictor is an autonomous political intelligence system. It monitors a fixed set of public figures, cross-references their statements (tweets) with news coverage, detects contradictions, and exposes results through an interactive query interface.

The system is built with LangGraph, Flask, React (Vite), Pinecone (vector search), and Supabase (PostgreSQL).

---

## Major modules

| Folder | Role |
|---|---|
| `src/graphs/` | LangGraph `StateGraph` definitions — wire agent nodes together, define conditional edges |
| `src/agents/` | Agent node implementations — each file is one agent; calls tools and the LLM |
| `src/agent_tools/` | Shared, reusable tool functions — Pinecone search, web scraping, URL extraction |
| `src/rss-extractor/` | RSS ingestion module — scrapes news feeds and exports to CSV and Supabase |
| `api/` | Flask application — HTTP handlers and SSE streaming only |
| `frontend/` | React UI (Vite) — chat interface and pipeline flowchart |
| `test/` | All tests and debugging utilities |

---

## Separation of concerns

**Layer rules (enforce in every PR):**

1. **Graph files** (`src/graphs/`) wire nodes and define routing edges. They do not call the LLM or implement business logic directly.
2. **Agent files** (`src/agents/`) implement one agent each: receive graph state, call tools, call the LLM, return an updated state.
3. **Tool files** (`src/agent_tools/`) are stateless, reusable functions with no agent-level state. They should be importable independently.
4. **Flask handlers** (`api/index.py`) are thin: validate input, invoke the graph or agent, return JSON or SSE. No business logic.
5. **Frontend** (`frontend/`) handles display and user interaction only. It consumes the Flask API.

---

## Runtime systems

### System A — Background Pipeline (scheduled daily) [PLANNED]

Autonomous batch pipeline that ingests data, analyses it, and builds cached per-figure pages.

```
START
  ↓
Ingestion Agent       → loads new tweets / articles into Supabase + Pinecone
  ↓
Topic Extractor       → tags records with topics (healthcare, economy, climate, etc.)
  ↓
Contradiction Finder  → compares tweets vs news per figure/topic
  ↓
Page Builder          → generates per-figure summary pages, writes to Supabase
  ↓
END
```

Planned graph definition: `src/graphs/background_graph.py`

### System B — Interactive Query Graph (on-demand) [COMPLETE]

Routes user questions to specialist RAG agents via a cached-first strategy.

```
START
  ↓
Page Lookup           → searches figure_pages in Supabase for a cached answer
  ↓ (conditional)
  ├── SUFFICIENT      → synthesise answer from cached page → END
  └── INSUFFICIENT
        ↓
      Router           → LLM classifies query → "tweet" / "news" / "both"
        ↓ (conditional)
        ├── Tweet Agent   → RAG over politics Pinecone index → END
        ├── News Agent    → RAG over politics-news Pinecone index → END
        └── Both Agents   → parallel RAG over both indexes → END
```

Graph definition: `src/graphs/query_graph.py`

---

## Tracked public figures

Donald Trump, Hillary Clinton, Barack Obama, Joe Biden, Kamala Harris, Elon Musk, Bill Gates, Mark Zuckerberg

Configured in: `src/rss-extractor/config/politicians.yaml`

---

## Data layer

See `docs/data_model.md` for full schema details.

**Supabase (PostgreSQL):** `tweets`, `news_articles`, `topics`, `tweet_topics`, `article_topics`, `contradictions`, `figure_pages`, `agent_runs`

**Pinecone indexes:**
- `politics` — tweet embeddings (~52K vectors, 1024-dim, cosine)
- `politics-news` — news article embeddings (~400 vectors, 1024-dim, cosine)

Embedding model: `RPRTHPB-text-embedding-3-small` via `https://api.llmod.ai/v1`

---

## Tech stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | `RPRTHPB-gpt-5-mini` via `api.llmod.ai` |
| Embeddings | `RPRTHPB-text-embedding-3-small` (1024-dim) via `api.llmod.ai` |
| Vector DB | Pinecone (serverless, AWS us-east-1) |
| SQL DB | Supabase (PostgreSQL) |
| Backend API | Flask |
| Frontend | React (Vite) |
| Deployment | Render |

---

## Implementation phases

| Phase | Description | Status |
|---|---|---|
| 1 | System B — interactive query graph + UI | **COMPLETE** |
| 2 | Topic Extraction | NOT STARTED |
| 3 | Contradiction Detection | NOT STARTED |
| 4 | Figure Pages + full System A | NOT STARTED |
| 5 | Deployment and automation | NOT STARTED |

---

## Known gaps and TODOs

- `src/graphs/background_graph.py` does not yet exist — System A is fully planned but not implemented.
- `src/agents/page_lookup.py` is a stub that always returns `{"found": False}`. It will be upgraded in Phase 4.
- `src/agents/ingestion_agent.py`, `topic_extractor.py`, `contradiction_finder.py`, `page_builder.py` are all planned but not yet implemented.
- The `src/rss-extractor/` module is functional but not yet integrated into the main LangGraph pipeline.
- `src/agent/` contains a legacy ReAct agent kept for backward compatibility — it is not part of the current System B graph.
