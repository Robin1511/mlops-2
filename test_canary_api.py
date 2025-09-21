import requests
import json
import time
from sklearn import datasets
from collections import Counter

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Health Response: {json.dumps(health_data, indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_root():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        root_data = response.json()
        print(f"Root Response: {json.dumps(root_data, indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_predict():
    """Test basic prediction functionality"""
    print("\n=== Testing Predict Endpoint ===")
    iris = datasets.load_iris()
    test_features = iris.data[:5].tolist()
    
    payload = {
        "features": test_features
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predictions: {result['predictions']}")
        print(f"Model used: {result['model_used']}")
        print(f"True classes: {iris.target[:5].tolist()}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_canary_probability():
    """Test setting canary probability"""
    print("\n=== Testing Canary Probability Configuration ===")
    
    # Test setting probability to 0.5 (50/50 split)
    payload = {"probability": 0.5}
    response = requests.post(
        f"{API_BASE_URL}/set-canary-probability",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Canary probability set: {json.dumps(result, indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_canary_behavior():
    """Test canary deployment behavior with multiple predictions"""
    print("\n=== Testing Canary Deployment Behavior ===")
    
    # Set probability to 0.5 for testing
    payload = {"probability": 0.5}
    requests.post(
        f"{API_BASE_URL}/set-canary-probability",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    iris = datasets.load_iris()
    test_features = iris.data[:1].tolist()  # Single sample
    
    # Make multiple predictions to test canary behavior
    model_usage = []
    for i in range(20):
        payload = {"features": test_features}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            model_used = "current" if "current" in result['model_used'] else "next"
            model_usage.append(model_used)
    
    # Count usage distribution
    usage_count = Counter(model_usage)
    print(f"Model usage distribution over 20 predictions:")
    print(f"Current model: {usage_count.get('current', 0)} times")
    print(f"Next model: {usage_count.get('next', 0)} times")
    
    return len(model_usage) == 20

def test_update_next_model():
    """Test updating the next model"""
    print("\n=== Testing Update Next Model ===")
    
    # Try to update to version 2 (if it exists, otherwise version 1)
    payload = {
        "model_name": "tracking-quickstart",
        "version": "2"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/update-model",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Update result: {json.dumps(result, indent=2)}")
        return True
    else:
        print(f"Error (expected if version 2 doesn't exist): {response.text}")
        # Try with version 1 instead
        payload["version"] = "1"
        response = requests.post(
            f"{API_BASE_URL}/update-model",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            print("Successfully updated next model to version 1")
            return True
        return False

def test_accept_next_model():
    """Test accepting the next model as current"""
    print("\n=== Testing Accept Next Model ===")
    
    response = requests.post(f"{API_BASE_URL}/accept-next-model")
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Accept result: {json.dumps(result, indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_full_canary_workflow():
    """Test the complete canary deployment workflow"""
    print("\n=== Testing Complete Canary Workflow ===")
    
    # Step 1: Check initial state
    print("1. Checking initial state...")
    health_response = requests.get(f"{API_BASE_URL}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"Initial state: Current={health_data['current_model']}, Next={health_data['next_model']}")
    
    # Step 2: Set canary probability to 100% current
    print("2. Setting canary probability to 100% current...")
    prob_payload = {"probability": 1.0}
    requests.post(
        f"{API_BASE_URL}/set-canary-probability",
        headers={"Content-Type": "application/json"},
        data=json.dumps(prob_payload)
    )
    
    # Step 3: Make some predictions (should all use current)
    print("3. Making predictions with 100% current...")
    iris = datasets.load_iris()
    test_features = iris.data[:1].tolist()
    for i in range(3):
        payload = {"features": test_features}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.json()
            print(f"   Prediction {i+1}: {result['model_used']}")
    
    # Step 4: Update next model (simulated by reloading same model)
    print("4. Updating next model...")
    update_payload = {
        "model_name": "tracking-quickstart",
        "version": "1"
    }
    requests.post(
        f"{API_BASE_URL}/update-model",
        headers={"Content-Type": "application/json"},
        data=json.dumps(update_payload)
    )
    
    # Step 5: Set canary probability to 50/50
    print("5. Setting canary probability to 50/50...")
    prob_payload = {"probability": 0.5}
    requests.post(
        f"{API_BASE_URL}/set-canary-probability",
        headers={"Content-Type": "application/json"},
        data=json.dumps(prob_payload)
    )
    
    # Step 6: Accept next model as current
    print("6. Accepting next model as current...")
    accept_response = requests.post(f"{API_BASE_URL}/accept-next-model")
    if accept_response.status_code == 200:
        result = accept_response.json()
        print(f"   Accept result: {result['message']}")
    
    # Step 7: Check final state
    print("7. Checking final state...")
    health_response = requests.get(f"{API_BASE_URL}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"Final state: Current={health_data['current_model']}, Next={health_data['next_model']}")
    
    return True

def wait_for_api(max_attempts=30):
    """Wait for API to be ready"""
    for i in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("API ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Waiting for API... Attempt {i+1}/{max_attempts}")
        time.sleep(2)
    return False

def run_tests():
    """Run all canary deployment tests"""
    if not wait_for_api():
        print("Could not connect to API")
        return
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Basic Prediction", test_predict),
        ("Canary Probability Configuration", test_canary_probability),
        ("Canary Behavior", test_canary_behavior),
        ("Update Next Model", test_update_next_model),
        ("Accept Next Model", test_accept_next_model),
        ("Full Canary Workflow", test_full_canary_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
    
    print("\n" + "="*60)
    print("CANARY DEPLOYMENT TEST RESULTS:")
    print("="*60)
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    passed = sum(1 for _, result in results if "PASS" in result)
    total = len(results)
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All canary deployment tests passed!")
    else:
        print("⚠️  Some tests failed or had errors")

if __name__ == "__main__":
    run_tests()