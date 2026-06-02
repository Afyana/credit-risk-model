"""
Task 5: Model Training - Simplified (No MLflow required)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime
from pathlib import Path

print("="*70)
print("TASK 5: MODEL TRAINING")
print("="*70)

# Create directories
Path("artifacts").mkdir(parents=True, exist_ok=True)

# ============================================
# STEP 1: LOAD THE DATA
# ============================================
print("\n📂 STEP 1: Loading training data...")

# Try multiple possible locations
possible_paths = [
    Path("data/processed/final_training_data.csv"),
    Path("C:/Users/hp/Desktop/credit_risk_data/data/processed/final_training_data.csv"),
]

data_path = None
for path in possible_paths:
    if path.exists():
        data_path = path
        print(f"   ✓ Found data at: {path}")
        break

if data_path is None:
    print("   ✗ Could not find final_training_data.csv!")
    exit(1)

# Load the data
df = pd.read_csv(data_path)
print(f"   ✓ Loaded {len(df)} customer records")
print(f"   ✓ Columns: {len(df.columns)} features")

# ============================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================
print("\n🔧 STEP 2: Preparing features and target...")

# Columns to exclude from features
exclude_cols = ['CustomerId', 'is_high_risk', 'Cluster', 'First_Transaction_Date', 'Last_Transaction_Date', 'is_high_risk_rule']

# Get feature columns (all columns not in exclude list)
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"   ✓ Using {len(feature_cols)} features")

# Prepare X and y
X = df[feature_cols]
y = df['is_high_risk']

# Handle any missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

print(f"   ✓ Feature shape: {X.shape}")
print(f"   ✓ Target distribution:")
print(f"     0 (Low Risk): {(y==0).sum()} customers")
print(f"     1 (High Risk): {(y==1).sum()} customers")
print(f"   ✓ Risk rate: {y.mean()*100:.2f}%")

# ============================================
# STEP 3: SPLIT DATA
# ============================================
print("\n📊 STEP 3: Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")
print(f"   ✓ Train risk rate: {y_train.mean()*100:.2f}%")
print(f"   ✓ Test risk rate: {y_test.mean()*100:.2f}%")

# Save feature columns for later use (API needs this)
np.save('artifacts/feature_columns.npy', feature_cols)
print(f"   ✓ Saved feature columns to artifacts/feature_columns.npy")

# ============================================
# STEP 4: TRAIN MODELS
# ============================================
print("\n🤖 STEP 4: Training models...")

# Define models to train
models = {
    'Logistic_Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'Random_Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None]
        }
    },
    'Gradient_Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    }
}

results = {}
best_model = None
best_model_name = None
best_roc_auc = 0

for model_name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    
    # Hyperparameter tuning
    print("   🔍 Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model_instance = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"   ✓ Best parameters: {best_params}")
    
    # Make predictions
    y_pred = best_model_instance.predict(X_test)
    y_pred_proba = best_model_instance.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n   📊 Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"      {metric_name}: {metric_value:.4f}")
    
    # Save results
    results[model_name] = {
        'model': best_model_instance,
        'metrics': metrics,
        'best_params': best_params
    }
    
    # Track best model
    if metrics['roc_auc'] > best_roc_auc:
        best_roc_auc = metrics['roc_auc']
        best_model = best_model_instance
        best_model_name = model_name

# ============================================
# STEP 5: DISPLAY MODEL COMPARISON
# ============================================
print("\n" + "="*70)
print("📊 MODEL COMPARISON RESULTS")
print("="*70)

comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': f"{result['metrics']['accuracy']:.4f}",
        'Precision': f"{result['metrics']['precision']:.4f}",
        'Recall': f"{result['metrics']['recall']:.4f}",
        'F1 Score': f"{result['metrics']['f1_score']:.4f}",
        'ROC-AUC': f"{result['metrics']['roc_auc']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC Score: {best_roc_auc:.4f}")

# ============================================
# STEP 6: SAVE BEST MODEL
# ============================================
print("\n💾 STEP 6: Saving best model...")

# Save with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = Path(f"artifacts/best_model_{timestamp}.pkl")
joblib.dump(best_model, model_path)
print(f"   ✓ Saved to: {model_path}")

# Save as best_model.pkl (simpler name for API)
simple_path = Path("artifacts/best_model.pkl")
joblib.dump(best_model, simple_path)
print(f"   ✓ Saved to: {simple_path}")

# Save model metadata
metadata = {
    'best_model_name': best_model_name,
    'best_roc_auc': best_roc_auc,
    'training_date': timestamp,
    'number_of_features': len(feature_cols),
    'feature_list': feature_cols,
    'model_performance': results[best_model_name]['metrics']
}

import json
with open('artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved metadata to: artifacts/model_metadata.json")

# Save classification report for best model
y_pred_best = best_model.predict(X_test)
report = classification_report(y_test, y_pred_best, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('artifacts/classification_report.csv')
print(f"   ✓ Saved classification report to: artifacts/classification_report.csv")

# ============================================
# STEP 7: FEATURE IMPORTANCE (for tree-based models)
# ============================================
if hasattr(best_model, 'feature_importances_'):
    print("\n📊 Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv('artifacts/feature_importance.csv', index=False)
    print(f"   ✓ Saved to: artifacts/feature_importance.csv")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ TASK 5 COMPLETE!")
print("="*70)

print("\n📁 Artifacts Created:")
print("   📊 artifacts/best_model.pkl - Trained model (for API)")
print("   📊 artifacts/feature_columns.npy - Feature list (for API)")
print("   📊 artifacts/model_metadata.json - Model performance summary")
print("   📊 artifacts/classification_report.csv - Detailed metrics")

print("\n📊 Model Performance Summary:")
print(f"   Best Model: {best_model_name}")
print(f"   ROC-AUC: {best_roc_auc:.4f}")
for metric, value in results[best_model_name]['metrics'].items():
    print(f"   {metric}: {value:.4f}")

print("\n🚀 Next Steps:")
print("   1. Start the API: python src/api/main.py")
print("   2. Or start minimal API: python api_minimal.py")
print("   3. Test the API: python test_api.py")

print("\n" + "="*70)
print("READY FOR TASK 6: API DEPLOYMENT!")
print("="*70)