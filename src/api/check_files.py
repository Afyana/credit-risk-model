import os
from pathlib import Path

print("Checking project structure...")
print(f"Current directory: {os.getcwd()}")

# Check for required directories
required_dirs = ['data/raw', 'data/processed', 'artifacts', 'src']
for dir_name in required_dirs:
    path = Path(dir_name)
    if path.exists():
        print(f"✓ {dir_name} exists")
    else:
        print(f"✗ {dir_name} missing - creating...")
        path.mkdir(parents=True, exist_ok=True)

# Check for data file
data_file = Path('data/raw/data.csv')
if data_file.exists():
    print(f"✓ Data file found: {data_file}")
else:
    print(f"✗ Data file missing! Please place data.csv in data/raw/")

# Check for processed files
processed_files = ['processed_customers.csv', 'final_training_data.csv']
for file in processed_files:
    path = Path(f'data/processed/{file}')
    if path.exists():
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")