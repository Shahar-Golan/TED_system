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
        "overlap_ratio": OVERLAP, # e.g. 0.15
        "top_k": TOP_K
    })

@app.route('/api/prompt', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("question", "")

    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    # 1. Embed the User's Question
    emb_response = client.embeddings.create(input=user_query, model=EMBEDDING_MODEL)
    query_vector = emb_response.data[0].embedding

# 2. Search Pinecone
    search_results = index.query(vector=query_vector, top_k=10, include_metadata=True)

    # 3. DEDUPLICATION: Keep only the best chunk for each unique talk_id
    unique_talks = {}
    for match in search_results['matches']:
        meta = match['metadata']
        talk_id = meta.get('talk_id')
        score = match['score']
        
        if talk_id not in unique_talks or score > unique_talks[talk_id]['score']:
            unique_talks[talk_id] = {
                "talk_id": talk_id,
                "title": meta.get('title'),
                "speakers": meta.get('speakers'), # <--- ADD THIS LINE
                "chunk": meta.get('chunk_text'),  # Matches your storage field
                "score": score
            }

    # 4. Get the Top 5 unique talks sorted by score
    final_context_list = sorted(unique_talks.values(), key=lambda x: x['score'], reverse=True)[:5]

    # 5. Build Context for the LLM
    context_text = ""
    for item in final_context_list:
        # Now item['speakers'] exists and won't cause a KeyError!
        context_text += f"Title: {item['title']}\nSpeaker: {item['speakers']}\nTranscript: {item['chunk']}\n\n"

    # 6. Call the LLM (gpt-5-mini)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
    ]
    
    chat_response = client.chat.completions.create(model=GPT_MODEL, messages=messages)
    final_answer = chat_response.choices[0].message.content

    # 7. Final Response (using your OrderedDict for strict formatting) [cite: 69-78]
    response_data = OrderedDict([
        ("response", final_answer),
        ("context", final_context_list), # Already contains talk_id, title, chunk, score
        ("Augmented_prompt", {
            "System": SYSTEM_PROMPT,
            "User": f"Context:\n{context_text}\n\nQuestion: {user_query}"
        })
    ])
    
    return jsonify(response_data)

# For local testing
if __name__ == '__main__':
    app.run(debug=True, port=3000, use_reloader=False)