"""
Task 6: Credit Risk Prediction API
FastAPI application for serving the trained model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Predict credit risk probability for loan applicants",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
feature_columns = None

# ============================================
# PYDANTIC MODELS
# ============================================

class CreditRiskRequest(BaseModel):
    Frequency: float = Field(..., ge=0)
    Monetary_Sum: float = Field(..., ge=0)
    Monetary_Mean: float = Field(..., ge=0)
    Monetary_Std: float = Field(..., ge=0)
    Recency: int = Field(..., ge=0)
    Tenure_Days: int = Field(..., ge=1)
    Fraud_Count: int = Field(0, ge=0)
    Fraud_Rate: float = Field(0.0, ge=0, le=1)
    Monetary_CV: float = Field(0.0)
    Monetary_Min: float = Field(0.0)
    Monetary_Max: float = Field(0.0)
    Monetary_Range: float = Field(0.0)
    Avg_Txn_Value: float = Field(0.0)
    Frequency_Daily: float = Field(0.0)


class CreditRiskResponse(BaseModel):
    risk_probability: float
    credit_score: int
    risk_category: str
    recommendation: str
    prediction_timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# ============================================
# MODEL LOADING
# ============================================

def load_model():
    """Load the trained model from artifacts directory"""
    global model, feature_columns
    
    try:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # Try multiple possible locations
        possible_paths = [
            project_root / "artifacts" / "best_model.pkl",
            Path("artifacts/best_model.pkl"),
            Path("../artifacts/best_model.pkl"),
            Path("C:/Users/hp/credit-risk-model/artifacts/best_model.pkl"),
        ]
        
        model_path = None
        for path in possible_paths:
            if path and path.exists():
                model_path = path
                break
        
        if model_path is None:
            # Try to find any model file
            artifacts_dir = project_root / "artifacts"
            if artifacts_dir.exists():
                model_files = list(artifacts_dir.glob("best_model_*.pkl"))
                if model_files:
                    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            
            if model_path is None:
                logger.error("No model file found in artifacts/")
                return False
        
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        
        # Load feature columns
        feature_paths = [
            project_root / "artifacts" / "feature_columns.npy",
            Path("artifacts/feature_columns.npy"),
            Path("../artifacts/feature_columns.npy"),
        ]
        
        for path in feature_paths:
            if path and path.exists():
                feature_columns = np.load(path, allow_pickle=True)
                logger.info(f"✓ Loaded {len(feature_columns)} features")
                break
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


# ============================================
# API ENDPOINTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    logger.info("Starting Credit Risk API...")
    success = load_model()
    if success:
        logger.info("API ready for predictions")
    else:
        logger.warning("API started without model")


@app.get("/", response_model=dict)
async def root():
    return {
        "service": "Bati Bank Credit Risk API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=CreditRiskResponse)
async def predict_risk(request: CreditRiskRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_data = request.dict()
        df = pd.DataFrame([input_data])
        
        if feature_columns is not None:
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_columns]
        
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        risk_probability = float(model.predict_proba(df)[0, 1])
        credit_score = int(850 - (risk_probability * 550))
        
        if risk_probability < 0.3:
            risk_category = "Low Risk"
            recommendation = "✓ Approve loan with standard terms"
        elif risk_probability < 0.7:
            risk_category = "Medium Risk"
            recommendation = "⚠️ Approve with higher interest rate"
        else:
            risk_category = "High Risk"
            recommendation = "❌ Review manually or decline"
        
        logger.info(f"Prediction - Risk: {risk_probability:.3f}, Score: {credit_score}")
        
        return CreditRiskResponse(
            risk_probability=round(risk_probability, 4),
            credit_score=credit_score,
            risk_category=risk_category,
            recommendation=recommendation,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[CreditRiskResponse])
async def batch_predict(requests: List[CreditRiskRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    responses = []
    for request in requests:
        try:
            input_data = request.dict()
            df = pd.DataFrame([input_data])
            
            if feature_columns is not None:
                for col in feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                df = df[feature_columns]
            
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            risk_probability = float(model.predict_proba(df)[0, 1])
            credit_score = int(850 - (risk_probability * 550))
            
            if risk_probability < 0.3:
                risk_category = "Low Risk"
                recommendation = "✓ Approve loan with standard terms"
            elif risk_probability < 0.7:
                risk_category = "Medium Risk"
                recommendation = "⚠️ Approve with higher interest rate"
            else:
                risk_category = "High Risk"
                recommendation = "❌ Review manually or decline"
            
            responses.append(CreditRiskResponse(
                risk_probability=round(risk_probability, 4),
                credit_score=credit_score,
                risk_category=risk_category,
                recommendation=recommendation,
                prediction_timestamp=datetime.now().isoformat()
            ))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
    
    return responses


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 STARTING CREDIT RISK API")
    print("="*60)
    
    # Try to load model before starting
    success = load_model()
    
    print(f"📊 Model loaded: {success}")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print("="*60 + "\n")
    
    # Run without reload to avoid warning
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)