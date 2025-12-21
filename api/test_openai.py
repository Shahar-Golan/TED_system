from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print(f"Testing OpenAI API connection...")
print(f"API Key loaded: {OPENAI_API_KEY[:20]}..." if OPENAI_API_KEY else "No API key found!")

# Initialize client with LLMod base URL
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.llmod.ai/v1"
)

try:
    # Simple test query
    response = client.chat.completions.create(
        model="RPRTHPB-gpt-5-mini",
        messages=[
            {"role": "user", "content": "who is the first president of USA"}
        ]
    )
    
    print("\n✅ SUCCESS! OpenAI API is working!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
