import requests
import json

url = "http://127.0.0.1:3000/api/prompt"
payload = {"question": "Which TED talk focuses on education or learning? Return a list of exactly 3 talk titles"}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        # This will print the FULL JSON structure required by your course
        print("\n--- FULL API RESPONSE ---")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Error (Status {response.status_code}):", response.text)

except Exception as e:
    print(f"Failed to connect: {e}")