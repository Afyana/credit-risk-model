import os
from pathlib import Path

print("="*60)
print("FILE LOCATION DIAGNOSTIC")
print("="*60)

# Check all possible locations
locations_to_check = [
    "C:/Users/hp/Desktop/credit_risk_data/data/processed/",
    "C:/Users/hp/credit-risk-model/data/processed/",
    "C:/Users/hp/Desktop/credit_risk_data/",
    "C:/Users/hp/credit-risk-model/data/",
]

print("\n📁 Checking for processed files:\n")

for location in locations_to_check:
    path = Path(location)
    if path.exists():
        print(f"✓ Directory exists: {location}")
        print(f"  Files found:")
        for file in path.glob("*.csv"):
            print(f"    - {file.name} ({file.stat().st_size} bytes)")
        print()
    else:
        print(f"✗ Directory does not exist: {location}")
        print()

# Specifically check for final_training_data.csv
print("="*60)
print("CHECKING FOR final_training_data.csv")
print("="*60)

possible_full_paths = [
    "C:/Users/hp/Desktop/credit_risk_data/data/processed/final_training_data.csv",
    "C:/Users/hp/credit-risk-model/data/processed/final_training_data.csv",
    "C:/Users/hp/Desktop/credit_risk_data/final_training_data.csv",
]

for path in possible_full_paths:
    if Path(path).exists():
        print(f"✓ FOUND: {path}")
    else:
        print(f"✗ NOT FOUND: {path}")