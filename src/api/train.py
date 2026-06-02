"""
Model Training Pipeline with MLflow Tracking
Trains and compares multiple models for credit risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CreditRiskTrainer:
    """
    Credit risk model training pipeline with MLflow integration
    """
    
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(random_state=random_state)
        }
        
        self.hyperparameters = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
    def prepare_features(self, df, target_col='is_high_risk'):
        """Prepare features and target for modeling"""
        # Define feature columns (exclude ID columns and target)
        exclude_cols = ['CustomerId', target_col, 'Cluster', 'First_Transaction_Date', 
                       'Last_Transaction_Date', 'TransactionId']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Features: {feature_cols}")
        
        return X, y
    
    def train_with_mlflow(self, model_name, X_train, X_test, y_train, y_test, hyperparameters=None):
        """Train a single model and log to MLflow"""
        
        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", self.random_state)
            
            # Get model instance
            model = self.models[model_name]
            
            # Apply hyperparameter tuning if specified
            if hyperparameters and hyperparameters != {}:
                mlflow.log_param("hyperparameter_tuning", "True")
                logger.info(f"Performing grid search for {model_name}...")
                
                grid_search = GridSearchCV(
                    model, hyperparameters, cv=5, 
                    scoring='roc_auc', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                mlflow.log_params(best_params)
                logger.info(f"Best parameters: {best_params}")
            else:
                # Train with default parameters
                model.fit(X_train, y_train)
                best_model = model
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            signature = infer_signature(X_train, y_pred_proba)
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
            # Log feature importance for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log as artifact
                feature_importance_df.to_csv('feature_importance.csv', index=False)
                mlflow.log_artifact('feature_importance.csv')
            
            logger.info(f"Model {model_name} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return best_model, metrics, run.info.run_id
    
    def run_training_pipeline(self, data_path):
        """Execute complete training pipeline"""
        
        # Load data
        logger.info("Loading training data...")
        df = pd.read_csv(data_path)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Set MLflow tracking
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("Bati_Bank_Credit_Risk")
        
        # Train and compare models
        results = {}
        best_model = None
        best_score = 0
        
        for model_name in self.models.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}...")
            logger.info(f"{'='*50}")
            
            # Use hyperparameters for tuning (or None for baseline)
            params = self.hyperparameters.get(model_name, None)
            
            model, metrics, run_id = self.train_with_mlflow(
                model_name, X_train, X_test, y_train, y_test, params
            )
            
            results[model_name] = {
                'metrics': metrics,
                'model': model,
                'run_id': run_id
            }
            
            # Track best model by ROC-AUC
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model
                best_model_name = model_name
        
        # Log best model to registry
        logger.info(f"\n{'='*50}")
        logger.info(f"Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        logger.info(f"{'='*50}")
        
        # Save best model
        model_path = f"C:/Users/hp/Desktop/credit_risk_data/artifacts/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Display comparison results
        comparison_df = pd.DataFrame({
            model_name: results[model_name]['metrics']
            for model_name in results.keys()
        }).T
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison_df.round(4))
        
        return results, best_model
    
    def register_best_model(self, best_model, model_name, metrics):
        """Register the best model in MLflow Model Registry"""
        with mlflow.start_run(run_name=f"REGISTER_{model_name}"):
            # Log the model with registry
            mlflow.sklearn.log_model(
                best_model,
                "credit_risk_model",
                registered_model_name="BatiBank_CreditRisk_Best"
            )
            
            # Log final metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Model registered as 'BatiBank_CreditRisk_Best'")
            return True


if __name__ == "__main__":
    # Initialize trainer
    trainer = CreditRiskTrainer(random_state=42, test_size=0.2)
    
    # Run training pipeline
    results, best_model = trainer.run_training_pipeline('C:/Users/hp/Desktop/credit_risk_data/data/processed/final_training_data.csv')
    
    # Register best model (will need to identify which one is best)
    # This is handled within run_training_pipeline