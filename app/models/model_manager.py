import joblib
import numpy as np
import logging
from typing import List, Union, Optional
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model loading and prediction."""
    
    def __init__(self, model_path: str = "models/sample_model.pkl"):
        self.model_path = Path(model_path)
        self.model: Optional[object] = None
        self.model_version: str = "1.0.0"
        self._model_loaded = False
    
    async def load_model(self) -> None:
        """Load the ML model asynchronously."""
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, self._load_model_sync)
            self._model_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_model_sync(self) -> object:
        """Synchronous model loading."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        return joblib.load(self.model_path)
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded and self.model is not None
    
    def get_model_version(self) -> str:
        """Get the current model version."""
        return self.model_version
    
    async def predict(self, features: List[float]) -> Union[float, int, str]:
        """
        Make a prediction using the loaded model.
        
        Args:
            features: List of input features
            
        Returns:
            Model prediction result
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert features to numpy array and reshape for sklearn
            features_array = np.array(features).reshape(1, -1)
            
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(None, self._predict_sync, features_array)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Failed to make prediction: {e}")
    
    def _predict_sync(self, features_array: np.ndarray) -> Union[float, int, str]:
        """Synchronous prediction."""
        return self.model.predict(features_array)[0]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.is_model_loaded():
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "model_type": type(self.model).__name__
        } 