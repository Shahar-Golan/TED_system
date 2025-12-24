import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Delete all vectors
print(f"Deleting all vectors from index: {os.getenv('PINECONE_INDEX_NAME')}")
index.delete(delete_all=True)
print("✅ All vectors deleted successfully!")
