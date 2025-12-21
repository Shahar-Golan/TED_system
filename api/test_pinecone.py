from pinecone import Pinecone
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ted-vector-db")

print(f"Testing Pinecone connection...")
print(f"API Key loaded: {PINECONE_API_KEY[:20]}..." if PINECONE_API_KEY else "No API key found!")
print(f"Index Name: {PINECONE_INDEX_NAME}")

try:
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✅ Pinecone client initialized")
    
    # Try to connect to index
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to index: {PINECONE_INDEX_NAME}")
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"✅ Index stats retrieved: {stats}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
