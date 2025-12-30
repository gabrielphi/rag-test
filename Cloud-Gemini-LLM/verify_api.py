import sys
import os
import json

# Add current dir to path to find api.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi.testclient import TestClient
    from api import app
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

client = TestClient(app)

def test_streaming():
    print("Testing Streaming Request...")
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": "Quem é você?"}
        ],
        "stream": True
    }
    
    try:
        # Note: TestClient uses httpx internally usually.
        # We use a simple post request and iterate if possible, 
        # or stream=True if supported by the client wrapper.
        # Standard requests-based TestClient in older fastapi might just return response.
        
        response = client.post("/v1/chat/completions", json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"FAILED: Status {response.status_code}")
            print(response.text)
            return

        print("Response headers:", response.headers)
        
        # Manually iterating assuming it's an iterator or similar
        print("Iterating response...")
        for line in response.iter_lines():
            if line:
                print(f"Received: {line}")
                
        print("\nStreaming Test PASSED.")

    except Exception as e:
        print(f"FAILED with Exception: {e}")

if __name__ == "__main__":
    test_streaming()
