"""
FastAPI Application for Credit Risk Prediction
"""
from fastapi import FastAPI, HTTPException, status
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting credit risk probability using transaction data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== PYDANTIC MODELS (Defined Here) ====================

class CreditRiskRequest(BaseModel):
    """Request model for credit risk prediction"""
    
    CustomerId: Optional[str] = Field(None, description="Customer identifier")
    
    # RFM Features
    Frequency: float = Field(..., ge=0, description="Number of transactions")
    Monetary_Sum: float = Field(..., ge=0, description="Total transaction amount")
    Monetary_Mean: float = Field(..., ge=0, description="Average transaction amount")
    Monetary_Std: float = Field(..., ge=0, description="Standard deviation of transaction amounts")
    Monetary_Min: float = Field(0, ge=0, description="Minimum transaction amount")
    Monetary_Max: float = Field(0, ge=0, description="Maximum transaction amount")
    
    # Temporal Features
    Recency: int = Field(..., ge=0, description="Days since last transaction")
    Tenure_Days: int = Field(..., ge=0, description="Customer tenure in days")
    
    # Risk Indicators
    Fraud_Count: int = Field(0, ge=0, description="Number of fraud flags")
    Fraud_Rate: float = Field(0, ge=0, le=1, description="Fraud rate")
    Monetary_CV: float = Field(0, ge=0, description="Coefficient of variation")
    Monetary_Range: float = Field(0, ge=0, description="Range of transaction amounts")
    Avg_Txn_Value: float = Field(0, ge=0, description="Average transaction value")
    Frequency_Daily: float = Field(0, ge=0, description="Transactions per day")
    
    @validator('Recency')
    def validate_recency(cls, v):
        if v < 0:
            raise ValueError('Recency cannot be negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerId": "CUST001",
                "Frequency": 45,
                "Monetary_Sum": 12500.50,
                "Monetary_Mean": 277.79,
                "Monetary_Std": 345.67,
                "Monetary_Min": 10.50,
                "Monetary_Max": 2500.00,
                "Recency": 12,
                "Tenure_Days": 180,
                "Fraud_Count": 0,
                "Fraud_Rate": 0.0,
                "Monetary_CV": 1.24,
                "Monetary_Range": 2489.50,
                "Avg_Txn_Value": 277.79,
                "Frequency_Daily": 0.25
            }
        }


class CreditRiskResponse(BaseModel):
    """Response model for credit risk prediction"""
    
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of default (0-1)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (higher is better)")
    risk_category: str = Field(..., description="Risk category: Low/Medium/High Risk")
    recommendation: str = Field(..., description="Loan recommendation based on risk assessment")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "risk_probability": 0.2345,
                "credit_score": 721,
                "risk_category": "Low Risk",
                "recommendation": "Approve loan with standard terms",
                "timestamp": "2026-06-01T10:30:00"
            }
        }


# ==================== MODEL LOADING ====================

# Global model variable
model = None
feature_columns = None

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'


def load_model():
    """Load the trained model from artifacts directory"""
    global model, feature_columns
    
    try:
        if not ARTIFACTS_DIR.exists():
            logger.error(f"Artifacts directory not found at {ARTIFACTS_DIR}")
            return False
        
        # Find model files
        model_files = list(ARTIFACTS_DIR.glob("best_model_*.pkl"))
        
        if not model_files:
            logger.error("No model files found. Please train a model first.")
            logger.info("Run: python src/train.py")
            return False
        
        # Load most recent model
        latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        model = joblib.load(latest_model_path)
        logger.info(f"✓ Model loaded from {latest_model_path.name}")
        
        # Load feature columns
        feature_file = ARTIFACTS_DIR / "feature_columns.npy"
        if feature_file.exists():
            feature_columns = np.load(feature_file, allow_pickle=True)
            logger.info(f"✓ Loaded {len(feature_columns)} feature columns")
        else:
            logger.warning("Feature columns file not found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting Credit Risk API...")
    success = load_model()
    if success:
        logger.info("API ready for predictions")
    else:
        logger.warning("API started without model - predictions will fail")


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Bati Bank Credit Risk Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=CreditRiskResponse)
async def predict_risk(request: CreditRiskRequest):
    """
    Predict credit risk probability for a customer
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train a model first."
            )
        
        # Convert request to dictionary
        input_data = request.dict()
        customer_id = input_data.pop('CustomerId', None)
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        if feature_columns is not None:
            # Add missing columns with default values
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Ensure correct column order
            df = df[feature_columns]
        
        # Handle any infinite or NaN values
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Make prediction
        risk_probability = float(model.predict_proba(df)[0, 1])
        
        # Determine risk category and recommendation
        if risk_probability < 0.3:
            risk_category = "Low Risk"
            recommendation = "✓ Approve loan with standard terms"
        elif risk_probability < 0.7:
            risk_category = "Medium Risk"
            recommendation = "⚠ Approve with higher interest rate or reduced amount"
        else:
            risk_category = "High Risk"
            recommendation = "✗ Review manually or decline loan"
        
        # Calculate credit score (300-850 range, inverse of risk)
        credit_score = int(850 - (risk_probability * 550))
        
        logger.info(f"Prediction - Customer: {customer_id}, Risk: {risk_probability:.3f}, Score: {credit_score}")
        
        return CreditRiskResponse(
            customer_id=customer_id,
            risk_probability=round(risk_probability, 4),
            credit_score=credit_score,
            risk_category=risk_category,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting Bati Bank Credit Risk API")
    print("="*50)
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    print(f"Model loaded: {model is not None}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)