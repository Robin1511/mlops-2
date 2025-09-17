import requests
import json
import time
from sklearn import datasets

API_BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_predict():
    
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
        print(f"Prédictions: {result['predictions']}")
        print(f"Vraies classes: {iris.target[:5].tolist()}")
        return True
    else:
        print(f"Erreur: {response.text}")
        return False

def test_update_model():
    payload = {
        "model_name": "tracking-quickstart",
        "version": "1"
    }
    response = requests.post(
        f"{API_BASE_URL}/update-model",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def wait_for_api(max_attempts=30):
    for i in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("API prête !")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Tentative {i+1}/{max_attempts}")
        time.sleep(2)
    return False

def run_tests():    
    if not wait_for_api():
        print("Impossible de se connecter à l'API")
        return
    
    tests = [
        ("Health Check", test_health),
        ("Predict Endpoint", test_predict),
        ("Update Model Endpoint", test_update_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
    
    print("\n" + "="*50)
    print("RÉSULTATS DES TESTS:")
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    passed = sum(1 for _, result in results if "PASS" in result)
    total = len(results)
    print(f"\nTests réussis: {passed}/{total}")

if __name__ == "__main__":
    run_tests() 