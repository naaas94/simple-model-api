from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    app_name: str = "Simple Model API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model settings
    model_path: str = "models/sample_model.pkl"
    model_version: str = "1.0.0"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS settings
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings() 