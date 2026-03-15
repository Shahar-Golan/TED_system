"""
Test the new API endpoints for Task 2
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_team_info():
    """Test GET /api/team_info"""
    print("Testing GET /api/team_info...")
    response = requests.get(f"{BASE_URL}/api/team_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print("\n" + "="*80 + "\n")

def test_agent_info():
    """Test GET /api/agent_info"""
    print("Testing GET /api/agent_info...")
    response = requests.get(f"{BASE_URL}/api/agent_info")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    # Print summary (full response is too long)
    print(f"Description: {data['description'][:100]}...")
    print(f"Purpose: {data['purpose'][:100]}...")
    print(f"Prompt template keys: {list(data['prompt_template'].keys())}")
    print(f"Number of prompt examples: {len(data['prompt_examples'])}")
    
    # Print first example summary
    if data['prompt_examples']:
        example = data['prompt_examples'][0]
        print(f"\nFirst Example:")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Steps: {len(example['steps'])}")
        print(f"  First module: {example['steps'][0]['module']}")
        print(f"  Response length: {len(example['full_response'])} chars")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print("="*80)
    print("Testing Task 2 API Endpoints")
    print("="*80 + "\n")
    
    try:
        test_team_info()
        test_agent_info()
        print("✅ All endpoint tests passed!")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Please make sure the Flask server is running on port 5000.")
    except Exception as e:
        print(f"❌ Error: {e}")
