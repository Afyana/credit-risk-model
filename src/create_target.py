"""
Task 4: Create Target Variable - Fixed for NaN values
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

print("="*60)
print("TASK 4: CREATING TARGET VARIABLE")
print("="*60)

# Find processed data
processed_paths = [
    Path("data/processed/processed_customers.csv"),
    Path("C:/Users/hp/Desktop/credit_risk_data/data/processed/processed_customers.csv"),
]

data_path = None
for path in processed_paths:
    if path.exists():
        data_path = path
        print(f"\n✓ Found processed data at: {path}")
        break

if data_path is None:
    print("\n✗ Could not find processed_customers.csv!")
    exit(1)

# Load data
df = pd.read_csv(data_path)
print(f"\n📂 Loaded {len(df)} customer records")

# Check for NaN values
print(f"\n📊 Checking for NaN values:")
print(df.isnull().sum())

# Clean the data - remove rows with NaN in critical columns
critical_cols = ['Recency', 'Frequency', 'Monetary_Sum', 'Frequency_Daily']
df = df.dropna(subset=critical_cols)
print(f"   After dropping NaN rows: {len(df)} customers")

# Also replace any remaining NaN or infinite values
df = df.replace([np.inf, -np.inf], 0).fillna(0)

print("\n🎯 Creating target variable using simple business rules...")

# Method 1: Simple business rules (most reliable)
# High risk = low frequency (< 10) AND high recency (> 30 days) AND low monetary (< 1000)
df['is_high_risk_rule'] = (
    (df['Frequency'] < 10) & 
    (df['Recency'] > 30) & 
    (df['Monetary_Sum'] < 1000)
).astype(int)

rule_risk_count = df['is_high_risk_rule'].sum()
print(f"\n   Business Rules Method:")
print(f"   - High Risk (1): {rule_risk_count} ({rule_risk_count/len(df)*100:.1f}%)")
print(f"   - Low Risk (0): {len(df)-rule_risk_count} ({(len(df)-rule_risk_count)/len(df)*100:.1f}%)")

# Method 2: Try K-Means clustering (with cleaned data)
print(f"\n🎯 Attempting K-Means clustering method...")

try:
    # Prepare RFM data for clustering
    rfm_data = df[['Recency', 'Frequency', 'Monetary_Sum']].copy()
    rfm_data.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Log transform monetary to handle skewness
    rfm_data['Monetary_Log'] = np.log1p(rfm_data['Monetary'])
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary_Log']])
    
    # Check for any remaining NaN
    if np.isnan(scaled_features).any():
        print("   Warning: NaN values still present, using business rules instead")
        df['is_high_risk'] = df['is_high_risk_rule']
    else:
        # K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        df['Cluster'] = clusters
        
        # Identify high-risk cluster (highest recency, lowest frequency)
        cluster_stats = df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary_Sum': 'mean'
        })
        
        print("\n   Cluster Profiles:")
        print(cluster_stats.to_string())
        
        # Find high risk cluster (highest recency, lowest frequency)
        high_risk_cluster = cluster_stats['Recency'].idxmax()
        print(f"\n   High-risk cluster identified: Cluster {high_risk_cluster}")
        
        df['is_high_risk'] = (df['Cluster'] == high_risk_cluster).astype(int)
        cluster_risk_count = df['is_high_risk'].sum()
        print(f"\n   K-Means Method:")
        print(f"   - High Risk (1): {cluster_risk_count} ({cluster_risk_count/len(df)*100:.1f}%)")
        print(f"   - Low Risk (0): {len(df)-cluster_risk_count} ({(len(df)-cluster_risk_count)/len(df)*100:.1f}%)")
        
except Exception as e:
    print(f"   K-Means failed: {e}")
    print("   Falling back to business rules method")
    df['is_high_risk'] = df['is_high_risk_rule']

# Final target column
print(f"\n📊 Final Target Distribution:")
print(df['is_high_risk'].value_counts())
print(f"   Risk rate: {df['is_high_risk'].mean()*100:.1f}%")

# Save to multiple locations
output_paths = [
    Path("data/processed/final_training_data.csv"),
    Path("C:/Users/hp/Desktop/credit_risk_data/data/processed/final_training_data.csv"),
]

for output_path in output_paths:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")

print("\n" + "="*60)
print("✅ TASK 4 COMPLETE!")
print("="*60)

# Show sample
print("\n📊 Sample of data with target:")
sample_cols = ['CustomerId', 'Frequency', 'Monetary_Sum', 'Recency', 'is_high_risk']
if 'Cluster' in df.columns:
    sample_cols.insert(4, 'Cluster')
available_cols = [col for col in sample_cols if col in df.columns]
print(df[available_cols].head(10))