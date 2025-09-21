from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import uvicorn
from typing import List
import logging
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLFlow Model Serving with Canary Deployment", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[int]
    model_used: str  # Added to track which model was used

class UpdateModelRequest(BaseModel):
    model_name: str
    version: str

class CanaryConfigRequest(BaseModel):
    probability: float  # Probability of using current model (0.0 to 1.0)

class CanaryModelService:
    def __init__(self):
        self.current_model = None
        self.next_model = None
        self.current_model_name = None
        self.current_model_version = None
        self.next_model_name = None
        self.next_model_version = None
        self.canary_probability = 1.0  # Default: always use current model
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
    def load_model(self, model_name: str, version: str, target: str = "current"):
        """Load a model into either 'current' or 'next' slot"""
        try:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            if target == "current":
                self.current_model = model
                self.current_model_name = model_name
                self.current_model_version = version
                logger.info(f"Current model loaded: {model_name} version {version}")
            elif target == "next":
                self.next_model = model
                self.next_model_name = model_name
                self.next_model_version = version
                logger.info(f"Next model loaded: {model_name} version {version}")
            else:
                raise ValueError("Target must be 'current' or 'next'")
                
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_initial_models(self, model_name: str, version: str):
        """Load the same model into both current and next slots at startup"""
        success_current = self.load_model(model_name, version, "current")
        success_next = self.load_model(model_name, version, "next")
        return success_current and success_next
    
    def set_canary_probability(self, probability: float):
        """Set the probability of using the current model (0.0 to 1.0)"""
        if 0.0 <= probability <= 1.0:
            self.canary_probability = probability
            logger.info(f"Canary probability set to {probability}")
            return True
        return False
    
    def accept_next_model(self):
        """Set the next model as the current model (both models become the same)"""
        if self.next_model is None:
            return False
        
        self.current_model = self.next_model
        self.current_model_name = self.next_model_name
        self.current_model_version = self.next_model_version
        logger.info(f"Next model accepted as current: {self.current_model_name} version {self.current_model_version}")
        return True
            
    def predict(self, features: List[List[float]]) -> tuple[List[int], str]:
        """Make prediction using canary deployment logic"""
        if self.current_model is None:
            raise HTTPException(status_code=400, detail="No current model loaded")
        
        # Determine which model to use based on probability
        use_current = random.random() < self.canary_probability
        
        try:
            features_array = np.array(features)
            
            if use_current or self.next_model is None:
                predictions = self.current_model.predict(features_array)
                model_used = f"current ({self.current_model_name} v{self.current_model_version})"
            else:
                predictions = self.next_model.predict(features_array)
                model_used = f"next ({self.next_model_name} v{self.next_model_version})"
            
            return predictions.tolist(), model_used
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

model_service = CanaryModelService()

@app.on_event("startup")
async def startup_event():
    success = model_service.load_initial_models("tracking-quickstart", "1")
    if not success:
        logger.warning("Unable to load default models")

@app.get("/")
def root():
    return {
        "message": "MLFlow Model Serving API with Canary Deployment",
        "current_model": {
            "name": model_service.current_model_name,
            "version": model_service.current_model_version
        },
        "next_model": {
            "name": model_service.next_model_name,
            "version": model_service.next_model_version
        },
        "canary_probability": model_service.canary_probability,
        "endpoints": [
            "/predict", 
            "/update-model", 
            "/accept-next-model", 
            "/set-canary-probability",
            "/health"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    predictions, model_used = model_service.predict(request.features)
    return PredictionResponse(predictions=predictions, model_used=model_used)

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    """Update the next model (for canary deployment)"""
    success = model_service.load_model(request.model_name, request.version, "next")
    if success:
        return {
            "message": f"Next model updated to {request.model_name} version {request.version}",
            "current_model": {
                "name": model_service.current_model_name,
                "version": model_service.current_model_version
            },
            "next_model": {
                "name": model_service.next_model_name,
                "version": model_service.next_model_version
            }
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to update next model")

@app.post("/accept-next-model")
def accept_next_model():
    """Accept the next model as the current model"""
    success = model_service.accept_next_model()
    if success:
        return {
            "message": "Next model accepted as current model",
            "current_model": {
                "name": model_service.current_model_name,
                "version": model_service.current_model_version
            },
            "next_model": {
                "name": model_service.next_model_name,
                "version": model_service.next_model_version
            }
        }
    else:
        raise HTTPException(status_code=400, detail="No next model to accept")

@app.post("/set-canary-probability")
def set_canary_probability(request: CanaryConfigRequest):
    """Set the probability of using the current model (0.0 = always next, 1.0 = always current)"""
    success = model_service.set_canary_probability(request.probability)
    if success:
        return {
            "message": f"Canary probability set to {request.probability}",
            "canary_probability": model_service.canary_probability,
            "description": f"Current model will be used {request.probability*100:.1f}% of the time"
        }
    else:
        raise HTTPException(status_code=400, detail="Probability must be between 0.0 and 1.0")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "current_model_loaded": model_service.current_model is not None,
        "next_model_loaded": model_service.next_model is not None,
        "current_model": {
            "name": model_service.current_model_name,
            "version": model_service.current_model_version
        },
        "next_model": {
            "name": model_service.next_model_name,
            "version": model_service.next_model_version
        },
        "canary_probability": model_service.canary_probability
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 