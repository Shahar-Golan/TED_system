import requests
import json

url = "https://ted-system.vercel.app/api/prompt"
payload = {"question": "I’m looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
}
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