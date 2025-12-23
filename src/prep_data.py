import os
import pandas as pd
import tiktoken
from pinecone import Pinecone
from openai import OpenAI
from pathlib import Path
import ast
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Client Initialization (Assuming API keys are in .env)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.llmod.ai/v1")
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# 2. Assignment Configuration [cite: 150-152]
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHUNK_SIZE_LIMIT = 1000  # Within the 2048 token limit [cite: 150]
OVERLAP = 200            # Within the 30% overlap limit [cite: 151]
BATCH_SIZE = 50

def get_embedding(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding

import math

def get_balanced_chunks(text, limit=1024, residual_threshold=200, overlap_ratio=0.25):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    
    # 1. Condition: If size < 1024 or the extra part is tiny (< 100)
    # We create one single chunk to avoid small, low-value vectors.
    if total_tokens < (limit + residual_threshold):
        return [tokenizer.decode(tokens)]
    
    # 2. Condition: If size > 1124, divide logically
    # Calculate number of chunks needed
    # Math: Total = N * Size - (N-1) * (Overlap_Ratio * Size)
    # Solving for N: N = ceil((Total - Overlap_Size) / (Limit - Overlap_Size))
    
    overlap_size = int(limit * overlap_ratio)
    step_size = limit - overlap_size
    num_chunks = math.ceil((total_tokens - overlap_size) / step_size)
    
    # Recalculate balanced size for these N chunks
    # Size = Total / (N - (N-1) * Overlap_Ratio)
    balanced_size = math.ceil(total_tokens / (num_chunks - (num_chunks - 1) * overlap_ratio))
    balanced_overlap = int(balanced_size * overlap_ratio)
    balanced_step = balanced_size - balanced_overlap
    
    chunks = []
    for i in range(0, total_tokens, balanced_step):
        chunk_tokens = tokens[i : i + balanced_size]
        # Append chunk; the loop will handle the end naturally
        chunks.append(tokenizer.decode(chunk_tokens))
        
        # Stop if the next step would be empty or start too late
        if i + balanced_step >= total_tokens:
            break
            
    return chunks

def get_clean_speakers(row):
    """Merges speaker_1 and all_speakers while avoiding duplicates."""
    primary = str(row['speaker_1']).strip()
    try:
        all_sp_data = ast.literal_eval(str(row['all_speakers']))
        if isinstance(all_sp_data, dict):
            all_names = [str(n).strip() for n in all_sp_data.values()]
            unique_names = list(dict.fromkeys([primary] + all_names))
            return ", ".join(unique_names)
    except:
        return primary

def process_and_upload():
    csv_path = Path(__file__).parent.parent / "data" / "ted_talks_en.csv"
    df = pd.read_csv(csv_path)
    
    # TARGET ROWS 20-40 (iloc is exclusive at the end, so 41 is used)
    df_subset = df.iloc[20:41] 
    
    vectors_to_upsert = []
    print("Processing talks 20-40 with 'Narrative Header' strategy...")
    
    for _, row in df_subset.iterrows():
        speakers = get_clean_speakers(row)
        
        # 3. NARRATIVE HEADER (The Search Context)
        # This makes the vector searchable by Title, Speaker, and Topics [cite: 121, 125]
        embedding_header = (
            f"Title: {row['title']}; "
            f"Speaker: {speakers}; "
            f"Occupations: {row['occupations']}; "
            f"Topics: {row['topics']}; "
            f"Description: {row['description']}. "
            f"Transcript: "
        )
        
        transcript = str(row['transcript']) if not pd.isna(row['transcript']) else ""
        chunks = get_balanced_chunks(transcript, limit=CHUNK_SIZE_LIMIT, residual_threshold=200, overlap_ratio=OVERLAP/CHUNK_SIZE_LIMIT)
        
        for chunk_index, chunk_content in enumerate(chunks):
            # Composite String for Embedding
            text_to_embed = embedding_header + chunk_content
            
            try:
                embedding = get_embedding(text_to_embed)
                
                # 4. SELECTIVE METADATA (The Retrieval Context)
                # Maps directly to the PDF's required API Output format 
                metadata = {
                    "talk_id": str(row['talk_id']),
                    "title": str(row['title']),
                    "speakers": speakers,
                    "chunk_text": chunk_content   # Matches index.py metadata field
                }
                
                vectors_to_upsert.append((f"{row['talk_id']}_c{chunk_index}", embedding, metadata))
                
            except Exception as e:
                print(f"Error at talk {row['talk_id']}: {e}")

            if len(vectors_to_upsert) >= BATCH_SIZE:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
    print("Upload of rows 20-40 complete.")

if __name__ == "__main__":
    process_and_upload()