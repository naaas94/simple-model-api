#!/usr/bin/env python3
"""
Example client script for the Simple Model API.
Demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
from typing import List, Dict, Any

class SimpleModelAPIClient:
    """Client for interacting with the Simple Model API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the API."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics from the API."""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.text
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make a prediction using the API."""
        data = {"features": features}
        response = self.session.post(
            f"{self.base_url}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get basic API information."""
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

def main():
    """Main function demonstrating API usage."""
    client = SimpleModelAPIClient()
    
    print("🚀 Simple Model API Client Example")
    print("=" * 50)
    
    try:
        # Check API health
        print("\n1. Health Check:")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        print(f"   Timestamp: {health['timestamp']}")
        
        # Get API info
        print("\n2. API Information:")
        info = client.get_api_info()
        print(f"   Message: {info['message']}")
        print(f"   Version: {info['version']}")
        print(f"   Docs URL: {info['docs']}")
        
        # Make predictions
        print("\n3. Making Predictions:")
        
        # Test case 1: Normal prediction
        features1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        print(f"   Test 1 - Features: {features1}")
        result1 = client.predict(features1)
        print(f"   Prediction: {result1['prediction']}")
        print(f"   Model Version: {result1['model_version']}")
        print(f"   Processing Time: {result1['processing_time']:.4f}s")
        
        # Test case 2: Different features
        features2 = [-1.0, -2.0, -3.0, 4.0, 5.0]
        print(f"\n   Test 2 - Features: {features2}")
        result2 = client.predict(features2)
        print(f"   Prediction: {result2['prediction']}")
        print(f"   Model Version: {result2['model_version']}")
        print(f"   Processing Time: {result2['processing_time']:.4f}s")
        
        # Test case 3: Edge case
        features3 = [0.0, 0.0, 0.0, 0.0, 0.0]
        print(f"\n   Test 3 - Features: {features3}")
        result3 = client.predict(features3)
        print(f"   Prediction: {result3['prediction']}")
        print(f"   Model Version: {result3['model_version']}")
        print(f"   Processing Time: {result3['processing_time']:.4f}s")
        
        # Get metrics
        print("\n4. Prometheus Metrics:")
        metrics = client.get_metrics()
        print("   Metrics endpoint accessible")
        print(f"   Metrics length: {len(metrics)} characters")
        
        # Performance test
        print("\n5. Performance Test:")
        test_features = [1.0, 2.0, 3.0, 4.0, 5.0]
        times = []
        
        for i in range(10):
            start_time = time.time()
            result = client.predict(test_features)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"   Request {i+1}: {times[-1]:.4f}s")
        
        avg_time = sum(times) / len(times)
        print(f"   Average response time: {avg_time:.4f}s")
        
        print("\n✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("   Make sure the API is running on http://localhost:8000")
        print("   Run: make run")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 