from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from openai import OpenAI
import os
import sys
import json
import psycopg2
from dotenv import load_dotenv
from pathlib import Path
from collections import OrderedDict
from typing import Any

# Add src directory to path for agent_tools import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent_tools.vector_search import vector_search
from agent.react_agent import run_agent
from graphs.query_graph import run_query, run_query_stream

# Load .env locally; Render will use its own Environment Variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)  # Enable CORS for React frontend
app.config["JSON_SORT_KEYS"] = False
if hasattr(app, "json"):
    app.json.sort_keys = False

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("BASE_URL", "https://api.llmod.ai/v1")

GPT_MODEL = "RPRTHPB-gpt-5-mini"
TOP_K = 15
CHUNK_SIZE = 1024
OVERLAP = 0.2

# Initialize OpenAI Client (for GPT responses)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip('"')
ARCHITECTURE_IMAGE_PATH = Path(__file__).parent.parent / "system_architecture.png"


def _get_db():
    """Create a Supabase DB connection."""
    return psycopg2.connect(
        SUPABASE_URL,
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


SYSTEM_PROMPT = """You are a source of truth for what public figures have actually stated on social media. Your role is to provide accurate, concise information about public figures' opinions and statements based strictly on their tweets.

Response Format:
- Present the 3 most relevant tweets in chronological order (oldest to newest)
- For each tweet include: Author name, date, and direct quote or key statement
- Keep it concise - avoid repeating similar points
- omit urls that appear in the tweet's text
- If other public figures in the context have relevant perspectives, briefly mention them at the end
- Identify patterns or contradictions only if clearly evident
- Use bullet points or short paragraphs for clarity

Guidelines:
- Answer using ONLY the tweet content and metadata provided
- Provide direct quotes when available
- Clearly attribute statements to the public figure who made them
- Do NOT use external knowledge beyond the provided tweets
- If tweets are not relevant to the question: "I don't have tweets from public figures addressing this topic."

Keep responses focused and readable."""


def _build_agent_info_examples() -> list[dict[str, Any]]:
    """Return static examples aligned with the System B multi-agent architecture."""
    return [
        {
            "prompt": "What does Donald Trump policy have said about Iran nuclear weapon development?",
            "full_response": "No relevant news articles were synthesized into a final answer for this run.",
            "steps": [
                {
                    "module": "Page Lookup",
                    "prompt": {
                        "query": "What does Donald Trump policy have said about Iran nuclear weapon development?",
                        "task": "Check if cached figure page can answer directly."
                    },
                    "response": {
                        "found": False,
                        "content": None,
                        "figure": "donald_trump"
                    }
                },
                {
                    "module": "Router",
                    "prompt": {
                        "query": "What does Donald Trump policy have said about Iran nuclear weapon development?",
                        "page_context_available": False
                    },
                    "response": {
                        "route": "news_agent",
                        "reason": "This is a request about formal policy positions and actions (sanctions, diplomacy, military posture) rather than specific social-media remarks; news_agent specializes in detailed, contextual coverage of such policies."
                    }
                },
                {
                    "module": "News Agent",
                    "prompt": {
                        "query": "What does Donald Trump policy have said about Iran nuclear weapon development?",
                        "source": "politics-news index"
                    },
                    "response": {
                        "articles_count": 5,
                        "agent": "news_agent",
                        "result": "No relevant news synthesis was returned for this run."
                    }
                }
            ]
        },
        {
            "prompt": "What does Joe Biden policy about Palestine?",
            "full_response": "No relevant news articles were synthesized into a final answer for this run.",
            "steps": [
                {
                    "module": "Page Lookup",
                    "prompt": {
                        "query": "What does Joe Biden policy about Palestine?",
                        "task": "Check if cached figure page can answer directly."
                    },
                    "response": {
                        "found": False,
                        "content": None,
                        "figure": "joe_biden"
                    }
                },
                {
                    "module": "Router",
                    "prompt": {
                        "query": "What does Joe Biden policy about Palestine?",
                        "page_context_available": False
                    },
                    "response": {
                        "route": "news_agent",
                        "reason": "This requires a detailed, contextual summary of Biden administration policy, actions, and coverage on Palestine rather than just social media quotes."
                    }
                },
                {
                    "module": "News Agent",
                    "prompt": {
                        "query": "What does Joe Biden policy about Palestine?",
                        "source": "politics-news index"
                    },
                    "response": {
                        "articles_count": 5,
                        "agent": "news_agent",
                        "result": "No relevant news synthesis was returned for this run."
                    }
                }
            ]
        },
        {
            "prompt": "Compare what Elon Musk tweeted about free speech with how news outlets covered it.",
            "full_response": "The system would route this to both specialist agents, then merge the tweet-based answer and the news-based answer into one response that separates direct statements from media coverage.",
            "steps": [
                {
                    "module": "Page Lookup",
                    "prompt": {
                        "query": "Compare what Elon Musk tweeted about free speech with how news outlets covered it.",
                        "task": "Check whether the cached figure page already contains enough material to answer both sides of the comparison."
                    },
                    "response": {
                        "found": False,
                        "figure": "elon_musk",
                        "decision": "The cached profile was insufficient for a full tweet-versus-news comparison."
                    }
                },
                {
                    "module": "Router",
                    "prompt": {
                        "query": "Compare what Elon Musk tweeted about free speech with how news outlets covered it.",
                        "page_context_available": True
                    },
                    "response": {
                        "route": "both",
                        "reason": "The user explicitly wants both direct tweets and news coverage."
                    }
                },
                {
                    "module": "Tweet Agent",
                    "prompt": {
                        "query": "Compare what Elon Musk tweeted about free speech with how news outlets covered it.",
                        "source": "politics tweet index"
                    },
                    "response": {
                        "result": "Produced the direct-statement portion from tweets."
                    }
                },
                {
                    "module": "News Agent",
                    "prompt": {
                        "query": "Compare what Elon Musk tweeted about free speech with how news outlets covered it.",
                        "source": "politics-news index"
                    },
                    "response": {
                        "result": "Produced the coverage-analysis portion from news articles."
                    }
                }
            ]
        }
    ]


def _build_execute_steps(user_prompt: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a lecturer-compatible execution trace from graph output."""
    route = str(result.get("route", ""))
    route_reason = str(result.get("route_reason", ""))
    tweets = result.get("tweets", [])
    articles = result.get("articles", [])

    steps: list[dict[str, Any]] = [
        {
            "module": "Router",
            "prompt": {
                "query": user_prompt,
                "task": "Select the best agent path (tweet, news, or both)."
            },
            "response": {
                "route": route,
                "reason": route_reason
            }
        }
    ]

    if route in ("tweet_agent", "both"):
        steps.append(
            {
                "module": "Tweet Agent",
                "prompt": {
                    "query": user_prompt,
                    "source": "politics tweet index"
                },
                "response": {
                    "tweets_count": len(tweets) if isinstance(tweets, list) else 0,
                    "tweets": tweets if isinstance(tweets, list) else []
                }
            }
        )

    if route in ("news_agent", "both"):
        steps.append(
            {
                "module": "News Agent",
                "prompt": {
                    "query": user_prompt,
                    "source": "politics-news index"
                },
                "response": {
                    "articles_count": len(articles) if isinstance(articles, list) else 0,
                    "articles": articles if isinstance(articles, list) else []
                }
            }
        )

    return steps

# --- Routes ---

@app.route('/api/team_info', methods=['GET'])
def team_info():
    """Returns student details for the team."""
    return jsonify({
        "group_batch_order_number": "3_12",
        "team_name": "שחר+תומר+אייל",
        "students": [
            {"name": "שחר גולן", "email": "shahar.golan@campus.technion.ac.il"},
            {"name": "תומר פרץ", "email": "tomer.perez@campus.technion.ac.il"},
            {"name": "אייל קוטליק", "email": "eyal.kotlik@campus.technion.ac.il"}
        ]
    })

@app.route('/api/agent_info', methods=['GET'])
def agent_info() -> Response:
    """Returns agent meta information and usage guidelines."""

    return jsonify({
        "description": "Politics-Contradictor uses a multi-agent System B query graph with four interactive modules: Page Lookup, Router, Tweet Agent, and News Agent. The graph first checks for an existing figure-page answer, then routes the query to tweet evidence, news coverage, or both.",
        "purpose": "Provide structured answers about public figures while preserving the lecturer's required API format: direct statements come from Tweet Agent, coverage questions come from News Agent, and mixed questions can use both after Page Lookup and Router decide the path.",
        "prompt_template": {
            "template": "Ask about a public figure and a topic, for example: 'What did [Figure] say about [Topic]?' for direct statements, 'How did news cover [Figure] on [Topic]?' for media coverage, or 'Compare [Figure]'s tweets and news coverage about [Topic]' when both views are needed.",
            "examples": [
                "What does Donald Trump policy have said about Iran nuclear weapon development?",
                "What does Joe Biden policy about Palestine?",
                "Compare what Elon Musk tweeted about free speech with how news outlets covered it."
            ],
            "guidelines": [
                "Name the public figure explicitly so Page Lookup can identify the right profile.",
                "Ask for direct statements when you want Tweet Agent to search social-media evidence.",
                "Ask about coverage, reporting, or regional analysis when you want News Agent.",
                "Ask for a comparison when you want Router to choose both specialist agents."
            ]
        },
        "prompt_examples": _build_agent_info_examples()
    })


@app.route('/api/model_architecture', methods=['GET'])
def model_architecture() -> Response:
    """Returns the architecture diagram as a PNG image."""
    if not ARCHITECTURE_IMAGE_PATH.exists():
        return jsonify({"error": "Architecture image not found"}), 404

    return send_from_directory(
        str(ARCHITECTURE_IMAGE_PATH.parent),
        ARCHITECTURE_IMAGE_PATH.name,
        mimetype='image/png',
    )


@app.route('/api/stats', methods=['GET'])
def stats():
    """Returns system parameters for automated grading."""
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP,
        "top_k": TOP_K
    })

@app.route('/api/prompt', methods=['POST'])
def chat():
    """Main RAG endpoint. Returns a compliant JSON object."""
    data = request.json
    user_query = data.get("question", "")
    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    # 1. Search for relevant tweets using vector_search tool
    search_result = vector_search(user_query, top_k=TOP_K)
    
    if not search_result["success"]:
        return jsonify({"error": f"Search failed: {search_result['error']}"}), 500

    # 2. Process tweets into context list
    context_list = []
    for match in search_result['results']:
        meta = match['metadata']
        score = match['score']
        text = meta.get('text', '')
        context_list.append({
            "tweet_id": match['id'],
            "account_id": meta.get('account_id'),
            "author_name": meta.get('author_name'),
            "text": text,
            "text_len": len(text),
            "created_at": meta.get('created_at'),
            "score": score
        })

    # 3. Final Context List (Top 7 for better coverage)
    final_context_list = context_list[:7]
    
    # 4. Sort by date for chronological presentation (oldest first)
    final_context_list_sorted = sorted(
        final_context_list, 
        key=lambda x: x.get('created_at', ''), 
        reverse=False
    )

    # 5. Build Augmented Prompt (using chronologically sorted context)
    context_text = ""
    for item in final_context_list_sorted:
        context_text += f"Author: {item['author_name']}\nDate: {item['created_at']}\nTweet: {item['text']}\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
    ]
    
    # 6. Generate Answer (Non-streaming for JSON compliance)
    chat_res = client.chat.completions.create(model=GPT_MODEL, messages=messages)
    final_answer = chat_res.choices[0].message.content

    # 7. Ordered JSON Output (Required by assignment)
    response_data = OrderedDict([
        ("response", final_answer),
        ("context", final_context_list_sorted),
        ("Augmented_prompt", {
            "System": SYSTEM_PROMPT,
            "User": f"Context:\n{context_text}\n\nQuestion: {user_query}"
        })
    ])
    
    return jsonify(response_data)


@app.route('/api/agent/query', methods=['POST'])
def agent_query():
    """
    Agentic RAG endpoint - uses ReAct agent with LLM reasoning.
    Returns comprehensive response with thought process and sources.
    """
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Run ReAct agent with LLM mode
    result = run_agent(
        user_query, 
        max_iterations=5, 
        verbose=False, 
        use_llm=True
    )
    
    if not result['success']:
        return jsonify({"error": "Agent failed to process query"}), 500
    
    # Format response
    response_data = OrderedDict([
        ("answer", result['final_answer']),
        ("mode", result['mode']),
        ("iterations", result['iterations']),
        ("thought_process", result['thoughts']),
        ("actions_taken", result['actions']),
        ("tweets_found", result['tweets_found']),
        ("tweets_used", result['tweets']),
        ("urls_analyzed", result['scraped_content'])
    ])
    
    return jsonify(response_data)


@app.route('/api/v2/query', methods=['POST'])
def graph_query():
    """
    LangGraph multi-agent endpoint.
    Routes queries through: page_lookup → router → tweet_agent / news_agent / both.
    """
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = run_query(user_query)

        response_data = OrderedDict([
            ("answer", result.get("answer", "")),
            ("route", result.get("route", "")),
            ("route_reason", result.get("route_reason", "")),
            ("agent_used", result.get("agent_used", "")),
            ("tweets", result.get("tweets", [])),
            ("articles", result.get("articles", [])),
        ])

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Graph query failed: {str(e)}"}), 500


@app.route('/api/execute', methods=['POST'])
def execute() -> Response:
    """
    Lecturer-compatible entrypoint.
    Input: {"prompt": "..."}
    Output: {"status", "error", "response", "steps"}
    """
    data = request.get_json(silent=True) or {}
    user_prompt = str(data.get('prompt', '')).strip() if isinstance(data, dict) else ''

    if not user_prompt:
        response_data = OrderedDict([
            ("status", "error"),
            ("error", "Missing required field: prompt"),
            ("response", None),
            ("steps", []),
        ])
        return jsonify(response_data), 400

    try:
        result = run_query(user_prompt)
        steps = _build_execute_steps(user_prompt, result)

        response_data = OrderedDict([
            ("status", "ok"),
            ("error", None),
            ("response", result.get("answer", "")),
            ("steps", steps),
        ])
        return jsonify(response_data)
    except Exception as e:
        response_data = OrderedDict([
            ("status", "error"),
            ("error", f"Execution failed: {str(e)}"),
            ("response", None),
            ("steps", []),
        ])
        return jsonify(response_data), 500


@app.route('/api/v2/query/stream', methods=['POST'])
def graph_query_stream():
    """
    SSE streaming endpoint for the LangGraph multi-agent pipeline.
    Streams node transitions and LLM tokens in real-time.
    """
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    def generate():
        for event in run_query_stream(user_query):
            yield f"data: {event}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    """Returns list of all speaker profiles (summary only)."""
    try:
        conn = _get_db()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                SELECT sp.speaker_id,
                       sp.name,
                       sp.party,
                       sp."current_role",
                       sp.profile->'bio'->>'born' as born,
                       sp.profile->'dataset_insights'->>'total_articles' as total_articles,
                       COUNT(t.author_name) as total_tweets
                FROM speaker_profiles sp
                LEFT JOIN tweets t
                                    ON (
                                             LOWER(REGEXP_REPLACE(COALESCE(t.author_name, ''), '[^a-z0-9]', '', 'g')) = LOWER(REGEXP_REPLACE(COALESCE(sp.name, ''), '[^a-z0-9]', '', 'g'))
                                        OR LOWER(REGEXP_REPLACE(COALESCE(t.author_name, ''), '[^a-z0-9]', '', 'g')) LIKE '%' || LOWER(REGEXP_REPLACE(COALESCE(sp.name, ''), '[^a-z0-9]', '', 'g')) || '%'
                                        OR LOWER(REGEXP_REPLACE(COALESCE(sp.name, ''), '[^a-z0-9]', '', 'g')) LIKE '%' || LOWER(REGEXP_REPLACE(COALESCE(t.author_name, ''), '[^a-z0-9]', '', 'g')) || '%'
                                    )
                GROUP BY sp.speaker_id,
                         sp.name,
                         sp.party,
                         sp."current_role",
                         sp.profile->'bio'->>'born',
                         sp.profile->'dataset_insights'->>'total_articles'
                ORDER BY sp.name
            """)
            rows = cur.fetchall()
        conn.close()

        speakers = []
        for row in rows:
            speakers.append({
                "speaker_id": row[0],
                "name": row[1],
                "party": row[2],
                "current_role": row[3],
                "born": row[4] or "",
                "total_articles": int(row[5]) if row[5] else 0,
                "total_tweets": int(row[6]) if row[6] else 0,
            })
        return jsonify(speakers)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/speakers/<speaker_id>', methods=['GET'])
def get_speaker_profile(speaker_id):
    """Returns full profile for a specific speaker."""
    try:
        conn = _get_db()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT profile, updated_at FROM speaker_profiles WHERE speaker_id = %s",
                (speaker_id,)
            )
            row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Speaker not found"}), 404

        profile = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        profile["updated_at"] = row[1].isoformat() if row[1] else None
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve React frontend for all non-API routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)