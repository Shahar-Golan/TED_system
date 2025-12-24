from flask import Flask, request, jsonify
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
from collections import OrderedDict

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

app = Flask(__name__)

# --- Configuration ---
# Note: Vercel will get these from the "Environment Variables" settings you add later
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ted-rag")

# Constants from your assignment
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
GPT_MODEL = "RPRTHPB-gpt-5-mini"
TOP_K = 5
CHUNK_SIZE = 2048
OVERLAP = 300

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.llmod.ai/v1"
)

# --- The System Prompt (MANDATORY from PDF) ---
SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.
If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.
You may add additional clarifications (e.g., response style), but you must keep the above constraints."""

@app.route('/api/stats', methods=['GET'])
def stats():
    # Strict JSON format required by assignment
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": round(OVERLAP / CHUNK_SIZE, 2), # e.g. 0.15
        "top_k": TOP_K
    })

@app.route('/api/prompt', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("question", "")

    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    # 1. Embed the User's Question
    emb_response = client.embeddings.create(
        input=user_query,
        model=EMBEDDING_MODEL
    )
    query_vector = emb_response.data[0].embedding

    # 2. Search Pinecone (Retrieval)
    search_results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )

    # 3. Build Context String & Context JSON list
    context_text = ""
    context_list_json = []

    for match in search_results['matches']:
        meta = match['metadata']
        score = match['score']
        
        # Format for the LLM to read
        snippet = f"Title: {meta.get('title')}\nTranscript: {meta.get('chunk_text')}\n\n"
        context_text += snippet
        
        # Format for the API response (Required by PDF)
        context_list_json.append({
            "talk_id": meta.get('talk_id'),
            "title": meta.get('title'),
            "chunk": meta.get('chunk_text'),
            "score": score
        })

    # 4. Construct the Final Prompt
    # We combine System Prompt + Retrieved Context + User Question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
    ]

    # 5. Call the LLM (Generation)
    chat_response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages
    )
    
    final_answer = chat_response.choices[0].message.content

    # 6. Return Final JSON (Strict format required by PDF)
    response_data = OrderedDict([
            ("response", final_answer),
            ("context", context_list_json),
            ("Augmented_prompt", {
                "System": SYSTEM_PROMPT,
                "User": f"Context:\n{context_text}\n\nQuestion: {user_query}"
            })
        ])
    app.json.sort_keys = False

    return jsonify(response_data)

# For local testing
if __name__ == '__main__':
    app.run(debug=True, port=3000, use_reloader=False)