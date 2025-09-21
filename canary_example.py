#!/usr/bin/env python3
"""
Canary Deployment Example

This script demonstrates how to use the canary deployment API
for gradually rolling out new model versions.
"""

import requests
import json
import time
from sklearn import datasets

API_BASE_URL = "http://localhost:8000"

def wait_for_api():
    """Wait for the API to be ready"""
    print("Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False

def print_status():
    """Print current system status"""
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 System Status:")
        print(f"   Current Model: {data['current_model']['name']} v{data['current_model']['version']}")
        print(f"   Next Model: {data['next_model']['name']} v{data['next_model']['version']}")
        print(f"   Canary Probability: {data['canary_probability']} (current model usage)")
        print(f"   Current Model Loaded: {data['current_model_loaded']}")
        print(f"   Next Model Loaded: {data['next_model_loaded']}")

def make_predictions(num_predictions=5):
    """Make sample predictions and show which model was used"""
    iris = datasets.load_iris()
    test_features = [iris.data[0].tolist()]  # Single sample
    
    print(f"\n🔮 Making {num_predictions} predictions:")
    model_usage = {"current": 0, "next": 0}
    
    for i in range(num_predictions):
        payload = {"features": test_features}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            model_type = "current" if "current" in result['model_used'] else "next"
            model_usage[model_type] += 1
            print(f"   Prediction {i+1}: {result['predictions'][0]} (using {result['model_used']})")
        else:
            print(f"   Prediction {i+1}: ERROR - {response.text}")
    
    print(f"\n📈 Usage Distribution:")
    print(f"   Current model: {model_usage['current']}/{num_predictions} ({model_usage['current']/num_predictions*100:.1f}%)")
    print(f"   Next model: {model_usage['next']}/{num_predictions} ({model_usage['next']/num_predictions*100:.1f}%)")

def set_canary_probability(probability):
    """Set the canary probability"""
    payload = {"probability": probability}
    response = requests.post(
        f"{API_BASE_URL}/set-canary-probability",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n⚙️  {result['message']}")
        print(f"   {result['description']}")
        return True
    else:
        print(f"❌ Error setting probability: {response.text}")
        return False

def update_next_model(model_name, version):
    """Update the next model"""
    payload = {"model_name": model_name, "version": version}
    response = requests.post(
        f"{API_BASE_URL}/update-model",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n🔄 {result['message']}")
        return True
    else:
        print(f"❌ Error updating model: {response.text}")
        return False

def accept_next_model():
    """Accept the next model as current"""
    response = requests.post(f"{API_BASE_URL}/accept-next-model")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ {result['message']}")
        return True
    else:
        print(f"❌ Error accepting model: {response.text}")
        return False

def canary_deployment_demo():
    """Demonstrate a complete canary deployment workflow"""
    print("🚀 Canary Deployment Demo")
    print("=" * 50)
    
    if not wait_for_api():
        print("❌ Could not connect to API. Make sure the server is running.")
        return
    
    # Step 1: Check initial status
    print("\n1️⃣  Initial System Status")
    print_status()
    
    # Step 2: Make some initial predictions (100% current)
    print("\n2️⃣  Initial Predictions (100% current model)")
    make_predictions(10)
    
    # Step 3: Update next model (simulate new version by reloading same model)
    print("\n3️⃣  Updating Next Model")
    update_next_model("tracking-quickstart", "1")
    print_status()
    
    # Step 4: Start canary with 10% traffic
    print("\n4️⃣  Starting Canary Deployment (10% new model)")
    set_canary_probability(0.9)  # 90% current, 10% next
    make_predictions(20)
    
    # Step 5: Increase canary to 50%
    print("\n5️⃣  Increasing Canary Traffic (50% new model)")
    set_canary_probability(0.5)  # 50/50 split
    make_predictions(20)
    
    # Step 6: Full rollout
    print("\n6️⃣  Full Rollout (100% new model)")
    set_canary_probability(0.0)  # 100% next model
    make_predictions(10)
    
    # Step 7: Accept new model as current
    print("\n7️⃣  Accepting New Model as Current")
    accept_next_model()
    print_status()
    
    # Step 8: Reset to normal operation
    print("\n8️⃣  Resetting to Normal Operation")
    set_canary_probability(1.0)  # 100% current
    make_predictions(5)
    
    print("\n🎉 Canary deployment demo completed!")
    print("\nKey takeaways:")
    print("• Models can be updated without downtime")
    print("• Traffic can be gradually shifted to new models")
    print("• Both models run simultaneously during canary phase")
    print("• Safe rollback is always possible")

if __name__ == "__main__":
    canary_deployment_demo()