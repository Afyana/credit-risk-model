"""
Test script for Credit Risk API
Run this after starting the API
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction():
    """Test prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Prediction Endpoint")
    print("="*50)
    
    # Test customer data
    test_customers = [
        {
            "name": "Low Risk Customer",
            "data": {
                "Frequency": 100,
                "Monetary_Sum": 50000,
                "Monetary_Mean": 500,
                "Monetary_Std": 100,
                "Recency": 2,
                "Tenure_Days": 365,
                "Fraud_Count": 0,
                "Fraud_Rate": 0,
                "Monetary_CV": 0.2,
                "Monetary_Min": 50,
                "Monetary_Max": 1000,
                "Monetary_Range": 950,
                "Avg_Txn_Value": 500,
                "Frequency_Daily": 0.27
            }
        },
        {
            "name": "High Risk Customer",
            "data": {
                "Frequency": 3,
                "Monetary_Sum": 150,
                "Monetary_Mean": 50,
                "Monetary_Std": 25,
                "Recency": 90,
                "Tenure_Days": 180,
                "Fraud_Count": 2,
                "Fraud_Rate": 0.67,
                "Monetary_CV": 0.5,
                "Monetary_Min": 25,
                "Monetary_Max": 75,
                "Monetary_Range": 50,
                "Avg_Txn_Value": 50,
                "Frequency_Daily": 0.017
            }
        }
    ]
    
    for customer in test_customers:
        print(f"\n📊 Testing: {customer['name']}")
        print("-" * 40)
        
        response = requests.post(f"{BASE_URL}/predict", json=customer['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Risk Probability: {result['risk_probability']:.4f}")
            print(f"✓ Credit Score: {result['credit_score']}")
            print(f"✓ Risk Category: {result['risk_category']}")
            print(f"✓ Recommendation: {result['recommendation']}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  {response.text}")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Batch Prediction Endpoint")
    print("="*50)
    
    batch_data = [
        {
            "Frequency": 100,
            "Monetary_Sum": 50000,
            "Monetary_Mean": 500,
            "Monetary_Std": 100,
            "Recency": 2,
            "Tenure_Days": 365,
            "Fraud_Count": 0,
            "Fraud_Rate": 0,
            "Monetary_CV": 0.2,
            "Monetary_Min": 50,
            "Monetary_Max": 1000,
            "Monetary_Range": 950,
            "Avg_Txn_Value": 500,
            "Frequency_Daily": 0.27
        },
        {
            "Frequency": 3,
            "Monetary_Sum": 150,
            "Monetary_Mean": 50,
            "Monetary_Std": 25,
            "Recency": 90,
            "Tenure_Days": 180,
            "Fraud_Count": 2,
            "Fraud_Rate": 0.67,
            "Monetary_CV": 0.5,
            "Monetary_Min": 25,
            "Monetary_Max": 75,
            "Monetary_Range": 50,
            "Avg_Txn_Value": 50,
            "Frequency_Daily": 0.017
        }
    ]
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Batch prediction complete: {len(results)} predictions")
        for i, result in enumerate(results):
            print(f"\n  Customer {i+1}:")
            print(f"    Risk: {result['risk_probability']:.4f}")
            print(f"    Score: {result['credit_score']}")
            print(f"    Category: {result['risk_category']}")
    else:
        print(f"✗ Error: {response.status_code}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING CREDIT RISK API")
    print("="*60)
    
    # First check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
    except requests.exceptions.ConnectionError:
        print("\n❌ API is not running!")
        print("Please start the API first:")
        print("  python src/api/main.py")
        exit(1)
    
    # Run tests
    test_health()
    test_prediction()
    test_batch_prediction()
    
    print("\n" + "="*60)
    print("✅ API TESTS COMPLETE!")
    print("="*60)