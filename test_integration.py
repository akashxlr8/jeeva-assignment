import subprocess
import time
import requests
import sys
import os
import signal

def run_tests():
    # Start the server
    print("Starting server...")
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    base_url = "http://localhost:8000"
    import uuid
    user_id = f"test_user_{uuid.uuid4()}"
    print(f"Testing with User ID: {user_id}")
    
    try:
        # Test 1: Initial Chat (Default Persona)
        print("\nTest 1: Initial Chat (Default Persona)")
        response = requests.post(f"{base_url}/chat", json={"user_id": user_id, "message": "Hello, who are you?"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert "Business Domain Expert" in response.json().get("persona", "")

        # Test 2: Switch to Mentor
        print("\nTest 2: Switch to Mentor")
        response = requests.post(f"{base_url}/chat", json={"user_id": user_id, "message": "Act like my mentor. How can I improve?"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert "Mentor" in response.json().get("persona", "")
        
        # Test 3: Switch to Investor
        print("\nTest 3: Switch to Investor")
        response = requests.post(f"{base_url}/chat", json={"user_id": user_id, "message": "Switch to investor. What is the ROI?"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert "Investor" in response.json().get("persona", "")
        
        # Test 4: Switch back to Mentor
        print("\nTest 4: Switch back to Mentor")
        response = requests.post(f"{base_url}/chat", json={"user_id": user_id, "message": "Back to mentor. What did we talk about?"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert "Mentor" in response.json().get("persona", "")
        
        # Test 5: Chat History
        print("\nTest 5: Chat History")
        response = requests.get(f"{base_url}/chat_history", params={"user_id": user_id})
        print(f"Status: {response.status_code}")
        history = response.json().get("history", {})
        print(f"History Keys: {list(history.keys())}")
        assert "Mentor" in history
        assert "Investor" in history
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        # Print server output if failed
        stdout, stderr = process.communicate()
        print("Server STDOUT:", stdout)
        print("Server STDERR:", stderr)
    finally:
        print("Stopping server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    run_tests()
