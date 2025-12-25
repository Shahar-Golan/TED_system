from flask import Flask, request, jsonify, render_template_string
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
from collections import OrderedDict

# Load .env locally; Vercel will use its own Environment Variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

app = Flask(__name__)

# --- Configuration ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ted-rag")

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
GPT_MODEL = "RPRTHPB-gpt-5-mini"
TOP_K = 10
CHUNK_SIZE = 1024
OVERLAP = 0.2

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.llmod.ai/v1"
)

SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.
If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.
You may add additional clarifications (e.g., response style), but you must keep the above constraints."""

# --- Routes ---

@app.route('/')
def home():
    """Renders the landing page with the integrated Chat UI."""
    return render_template_string(CHAT_UI_TEMPLATE)

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

    # 1. Embed Question
    emb_res = client.embeddings.create(input=user_query, model=EMBEDDING_MODEL)
    query_vector = emb_res.data[0].embedding

    # 2. Retrieve from Pinecone
    search_results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    # 3. Deduplicate (Ensure distinct talk titles)
    unique_talks = {}
    for match in search_results['matches']:
        meta = match['metadata']
        tid = meta.get('talk_id')
        score = match['score']
        if tid not in unique_talks or score > unique_talks[tid]['score']:
            unique_talks[tid] = {
                "talk_id": tid,
                "title": meta.get('title'),
                "speakers": meta.get('speakers'),
                "chunk": meta.get('chunk_text'),
                "score": score
            }

    # 4. Final Context List (Top 5 Unique)
    final_context_list = sorted(unique_talks.values(), key=lambda x: x['score'], reverse=True)[:5]

    # 5. Build Augmented Prompt
    context_text = ""
    for item in final_context_list:
        context_text += f"Title: {item['title']}\nSpeaker: {item['speakers']}\nTranscript: {item['chunk']}\n\n"

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
        ("context", final_context_list),
        ("Augmented_prompt", {
            "System": SYSTEM_PROMPT,
            "User": f"Context:\n{context_text}\n\nQuestion: {user_query}"
        })
    ])
    
    return jsonify(response_data)

# --- Integrated UI Template ---
CHAT_UI_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TED Assistant</title>
    <style>
        body { font-family: sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .container { background: white; width: 100%; max-width: 700px; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #e62b1e; margin-top: 0; }
        #chat-box { border: 1px solid #ddd; border-radius: 8px; padding: 15px; min-height: 200px; margin: 20px 0; background: #fafafa; white-space: pre-wrap; line-height: 1.5; color: #333; }
        .input-group { display: flex; gap: 10px; }
        input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
        button { background: #e62b1e; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: bold; }
        button:disabled { background: #ccc; }
        .status { font-size: 0.9em; color: #666; margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TED Talk RAG Assistant</h1>
        <div class="status">System status: <a href="/api/stats" target="_blank">Active</a></div>
        <div id="chat-box">Enter your question below...</div>
        <div class="input-group">
            <input type="text" id="user-input" placeholder="e.g., Recommend a talk on climate change" onkeypress="if(event.key==='Enter') sendMessage()">
            <button id="send-btn" onclick="sendMessage()">Ask</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const box = document.getElementById('chat-box');
            const btn = document.getElementById('send-btn');
            const query = input.value.trim();
            if (!query) return;

            box.innerHTML = "<i>Searching TED dataset and generating answer...</i>";
            btn.disabled = true;

            try {
                const res = await fetch('/api/prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: query })
                });
                const data = await res.json();
                
                // We extract ONLY the "response" field for the UI
                box.innerText = data.response; 
            } catch (err) {
                box.innerText = "Error: Could not connect to the API.";
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=3000, use_reloader=False)