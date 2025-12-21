import requests
import json

# The URL of your local server
url = "http://127.0.0.1:3000/api/prompt"

# The test question
payload = {
    "question": "What talks discuss technology and education?"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        print("🤖 AI Answer:", data['response'])
        print("\n📄 Retrieval Evidence (Context):")
        for ctx in data['context']:
            print(f"- [Score: {ctx['score']:.4f}] {ctx['title']}")
    else:
        print(f"Error (Status {response.status_code}):", response.text)

except Exception as e:
    print(f"Failed to connect. Is the server running? Error: {e}")
    import traceback
    traceback.print_exc()