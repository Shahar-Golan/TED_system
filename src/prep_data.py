import os
import pandas as pd
import tiktoken
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import time
from pathlib import Path

# 1. Load Environment Variables
# Load .env from project root (one level up from src)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.llmod.ai"
)
index = pc.Index(INDEX_NAME)

# 2. Configuration (Strictly following Assignment Constraints)
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHUNK_SIZE_LIMIT = 2048  # Max tokens [cite: 42]
OVERLAP = 300            # Approx 15-20% overlap (Max is 30%) [cite: 43]
BATCH_SIZE = 100         # Batch processing to save time/requests

def get_embedding(text):
    # Generates embedding for a single string
    print(f"DEBUG: Requesting embedding with model: {EMBEDDING_MODEL}")
    print(f"DEBUG: API Base URL: {client.base_url}")
    print(f"DEBUG: Text length: {len(text)} chars")
    try:
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        print(f"DEBUG: Embedding successful, vector size: {len(response.data[0].embedding)}")
        return response.data[0].embedding
    except Exception as e:
        print(f"DEBUG: Full error details: {type(e).__name__}: {e}")
        raise

def chunk_text(text, limit, overlap):
    # Uses tiktoken to ensure we don't exceed token limits accurately
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    
    chunks = []
    step = limit - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + limit]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def process_and_upload():
    print("Loading Dataset...")
    # Load only the necessary columns to save memory
    # CSV is stored in the data folder (one level up from src)
    csv_path = Path(__file__).parent.parent / "data" / "ted_talks_en.csv"
    print(f"DEBUG: CSV path: {csv_path}")
    print(f"DEBUG: CSV exists: {csv_path.exists()}")
    df = pd.read_csv(csv_path)
    print(f"DEBUG: Loaded {len(df)} talks from CSV")
    
    # ⚠️ COST SAVING TIP: Start with top 20 talks to test your pipeline! 
    # Remove the .head(20) when you are ready for the full dataset.
    df_subset = df.head(20) 
    
    vectors_to_upsert = []
    
    print(f"Processing {len(df_subset)} talks...")
    
    for i, row in df_subset.iterrows():
        talk_id = str(row['talk_id'])
        title = row['title']
        transcript = row['transcript']
        
        # Skip if transcript is missing
        if pd.isna(transcript):
            continue

        # Create Chunks
        chunks = chunk_text(transcript, CHUNK_SIZE_LIMIT, OVERLAP)
        
        for chunk_index, chunk_content in enumerate(chunks):
            # Create a unique ID for this chunk
            vector_id = f"{talk_id}_chk_{chunk_index}"
            
            # Generate Embedding
            # Note: For production, you'd batch these embedding calls too, 
            # but this is simpler for debugging.
            try:
                embedding = get_embedding(chunk_content)
                
                # Prepare Metadata (Crucial for the RAG later)
                metadata = {
                    "talk_id": talk_id,
                    "title": title,
                    "chunk_text": chunk_content,
                    "url": row['url'],
                    "speakers": row['speakers'] if 'speakers' in row else "Unknown"
                }
                
                vectors_to_upsert.append((vector_id, embedding, metadata))
                
            except Exception as e:
                print(f"Error embedding {vector_id}: {e}")

            # Upload in batches of 50 to avoid timeouts
            if len(vectors_to_upsert) >= 50:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Uploaded batch. Total progress: roughly talk {i}")
                vectors_to_upsert = []

    # Upload any remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        print("Final batch uploaded.")

if __name__ == "__main__":
    process_and_upload()
    print("Data preparation complete. Pinecone is ready.")