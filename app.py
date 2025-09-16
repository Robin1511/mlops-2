from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import uvicorn
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLFlow Model Serving", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[int]

class UpdateModelRequest(BaseModel):
    model_name: str
    version: str

class ModelService:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_version = None
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
    def load_model(self, model_name: str, version: str):
        try:
            model_uri = f"models:/{model_name}/{version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_name = model_name
            self.model_version = version
            logger.info(f"Modèle chargé: {model_name} version {version}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
            
    def predict(self, features: List[List[float]]) -> List[int]:
        if self.model is None:
            raise HTTPException(status_code=400, detail="Aucun modèle chargé")
        
        try:
            features_array = np.array(features)
            predictions = self.model.predict(features_array)
            return predictions.tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    success = model_service.load_model("tracking-quickstart", "1")
    if not success:
        logger.warning("Impossible de charger le modèle par défaut")

@app.get("/")
def root():
    return {
        "message": "MLFlow Model Serving API",
        "model": model_service.model_name,
        "version": model_service.model_version,
        "endpoints": ["/predict", "/update-model", "/health"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    predictions = model_service.predict(request.features)
    return PredictionResponse(predictions=predictions)

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    success = model_service.load_model(request.model_name, request.version)
    if success:
        return {
            "message": f"Modèle mis à jour vers {request.model_name} version {request.version}",
            "model": model_service.model_name,
            "version": model_service.model_version
        }
    else:
        raise HTTPException(status_code=400, detail="Échec de la mise à jour du modèle")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "model": model_service.model_name,
        "version": model_service.model_version
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 