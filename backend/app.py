#!/usr/bin/env python3
"""
FastAPI Backend for Fraud Detection System
Real-time fraud detection API with model serving
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging
import uvicorn
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personal Finance Anomaly Detector",
    description="Real-time fraud detection API for financial transactions",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Render deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React frontend static files
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")

# Global variables for model and preprocessing
model = None
feature_names = None
model_metadata = None

class TransactionRequest(BaseModel):
    """Transaction data for fraud detection"""
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    
    @validator('Amount')
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        return v

class BatchTransactionRequest(BaseModel):
    """Batch transaction data for fraud detection"""
    transactions: List[TransactionRequest]

class FraudDetectionResponse(BaseModel):
    """Response for fraud detection"""
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: str
    timestamp: datetime
    transaction_id: Optional[str] = None

class BatchFraudDetectionResponse(BaseModel):
    """Response for batch fraud detection"""
    results: List[FraudDetectionResponse]
    summary: Dict[str, int]

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    features_count: int
    training_date: str
    performance_metrics: Dict
    feature_importance: List[Dict]

def load_model_and_metadata():
    """Load the trained model and metadata"""
    global model, feature_names, model_metadata
    
    try:
        # Find the latest improved model
        models_dir = Path(__file__).parent.parent / 'ml' / 'models' / 'improved'
        
        if not models_dir.exists():
            models_dir = Path(__file__).parent.parent / 'ml' / 'models' / 'final'
        
        model_files = list(models_dir.glob("*.pkl"))
        if not model_files:
            logger.warning("No trained model found! Using demo mode.")
            # Set up demo mode with mock data
            model = None
            feature_names = [f"V{i}" for i in range(1, 15)] + ["Amount"]
            model_metadata = {
                "model_name": "XGBoost Fraud Detector (Demo)",
                "version": "1.0-demo",
                "improvement_date": "2024-06-19",
                "test_performance": {
                    "roc_auc": 0.922,
                    "pr_auc": 0.130,
                    "fraud_detection_rate": 0.732,
                    "false_alarm_rate": 0.093
                }
            }
            return
        
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Loading model: {latest_model_file}")
        model = joblib.load(latest_model_file)
        
        # Load feature names
        data_dir = Path(__file__).parent.parent / 'ml' / 'data' / 'processed'
        with open(data_dir / 'final_features.json', 'r') as f:
            feature_names = json.load(f)
        
        # Load model metadata
        metadata_file = models_dir / 'improved_model_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                'model_name': 'XGBoost Fraud Detector',
                'version': '1.0',
                'training_date': 'Unknown'
            }
        
        logger.info(f"Model loaded successfully with {len(feature_names)} features")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def preprocess_transaction(transaction: TransactionRequest) -> np.ndarray:
    """Preprocess a single transaction for prediction"""
    # Extract base features
    base_features = {
        'Amount': transaction.Amount,
        'V1': transaction.V1,
        'V2': transaction.V2,
        'V3': transaction.V3,
        'V4': transaction.V4,
        'V5': transaction.V5,
        'V6': transaction.V6,
        'V7': transaction.V7,
        'V8': transaction.V8,
        'V9': transaction.V9,
        'V10': transaction.V10,
        'V11': transaction.V11,
        'V12': transaction.V12,
        'V13': transaction.V13,
        'V14': transaction.V14
    }
    
    # Calculate additional features that our model expects
    v_values = [transaction.V1, transaction.V2, transaction.V3, transaction.V4, 
                transaction.V5, transaction.V6, transaction.V7, transaction.V8,
                transaction.V9, transaction.V10, transaction.V11, transaction.V12,
                transaction.V13, transaction.V14]
    
    # Create all expected features
    all_features = base_features.copy()
    
    # Add engineered features if they exist in our model
    if 'v_std' in feature_names:
        all_features['v_std'] = np.std(v_values)
    if 'v_min' in feature_names:
        all_features['v_min'] = np.min(v_values)
    if 'v_max' in feature_names:
        all_features['v_max'] = np.max(v_values)
    if 'V1_V2_interaction' in feature_names:
        all_features['V1_V2_interaction'] = transaction.V1 * transaction.V2
    if 'V1_V3_interaction' in feature_names:
        all_features['V1_V3_interaction'] = transaction.V1 * transaction.V3
    if 'Amount_V3_interaction' in feature_names:
        all_features['Amount_V3_interaction'] = transaction.Amount * transaction.V3
    
    # Create feature vector in the same order as training
    feature_vector = []
    for feature_name in feature_names:
        if feature_name in all_features:
            feature_vector.append(all_features[feature_name])
        else:
            feature_vector.append(0.0)  # Default value for missing features
    
    return np.array(feature_vector).reshape(1, -1)

def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.5:
        return "HIGH"
    elif probability >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
    if probability >= 0.9 or probability <= 0.1:
        return "HIGH"
    elif probability >= 0.7 or probability <= 0.3:
        return "MEDIUM"
    else:
        return "LOW"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_and_metadata()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Personal Finance Anomaly Detector API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "features_count": len(feature_names) if feature_names else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get feature importance from model
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
        for name, importance in zip(feature_names, importance_values):
            feature_importance.append({
                "feature": name,
                "importance": float(importance)
            })
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return ModelInfo(
        model_name=model_metadata.get('model_name', 'XGBoost Fraud Detector'),
        version=model_metadata.get('version', '1.0'),
        features_count=len(feature_names),
        training_date=model_metadata.get('improvement_date', 'Unknown'),
        performance_metrics=model_metadata.get('test_performance', {}),
        feature_importance=feature_importance[:10]  # Top 10 features
    )

@app.post("/predict", response_model=FraudDetectionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud for a single transaction"""
    
    try:
        if model is None:
            # Demo mode - use rule-based prediction
            import random
            random.seed(int(transaction.Amount * 1000 + sum([getattr(transaction, f'V{i}', 0) for i in range(1, 15)])))
            
            # Simple demo rules
            if transaction.Amount > 500:
                fraud_probability = min(0.8, 0.3 + (transaction.Amount / 1000) * 0.4)
            elif transaction.Amount < 10:
                fraud_probability = max(0.1, 0.2 + random.random() * 0.3)
            else:
                fraud_probability = 0.1 + random.random() * 0.3
        else:
            # Real model prediction
            features = preprocess_transaction(transaction)
            fraud_probability = model.predict_proba(features)[0, 1]
        
        is_fraud = fraud_probability >= 0.4  # Using business-optimal threshold
        
        return FraudDetectionResponse(
            is_fraud=is_fraud,
            fraud_probability=float(fraud_probability),
            risk_level=get_risk_level(fraud_probability),
            confidence=get_confidence_level(fraud_probability),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchFraudDetectionResponse)
async def predict_fraud_batch(batch_request: BatchTransactionRequest):
    """Predict fraud for multiple transactions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        fraud_count = 0
        
        for i, transaction in enumerate(batch_request.transactions):
            # Preprocess transaction
            features = preprocess_transaction(transaction)
            
            # Make prediction
            fraud_probability = model.predict_proba(features)[0, 1]
            is_fraud = fraud_probability >= 0.4
            
            if is_fraud:
                fraud_count += 1
            
            results.append(FraudDetectionResponse(
                is_fraud=is_fraud,
                fraud_probability=float(fraud_probability),
                risk_level=get_risk_level(fraud_probability),
                confidence=get_confidence_level(fraud_probability),
                timestamp=datetime.now(),
                transaction_id=f"txn_{i+1}"
            ))
        
        summary = {
            "total_transactions": len(batch_request.transactions),
            "fraud_detected": fraud_count,
            "normal_transactions": len(batch_request.transactions) - fraud_count
        }
        
        return BatchFraudDetectionResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "model_loaded": model is not None,
        "features_available": len(feature_names) if feature_names else 0,
        "api_version": "1.0.0",
        "uptime": "Available on startup",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/{catch_all:path}")
async def serve_react_app(catch_all: str):
    """Serve React app for all non-API routes"""
    frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
    index_file = frontend_build_path / "index.html"
    
    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        return {"message": "Frontend not built. Run 'npm run build' in frontend directory."}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )