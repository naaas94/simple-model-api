from pydantic import BaseModel, Field, field_validator
from typing import List, Union
import numpy as np

class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    
    features: List[float] = Field(
        description="Input features for prediction",
        min_length=1,
        max_length=100
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate that features are finite numbers."""
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in v):
            raise ValueError("All features must be finite numbers")
        return v

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    prediction: Union[float, int, str] = Field(
        description="Model prediction result"
    )
    model_version: str = Field(
        description="Version of the model used for prediction"
    )
    processing_time: float = Field(
        description="Time taken to process the prediction in seconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0.85,
                "model_version": "1.0.0",
                "processing_time": 0.0123
            }
        }
    } 