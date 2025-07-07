#!/usr/bin/env python3
"""
Training script to create a sample ML model for the API.
This script trains a simple Random Forest classifier on synthetic data.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 5) -> tuple:
    """
    Generate synthetic data for training.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create a simple target based on feature combinations
    # Target = 1 if sum of first 3 features > 0, else 0
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    
    return X, y

def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Args:
        X: Training features
        y: Training targets
        
    Returns:
        Trained Random Forest model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    return model

def save_model(model: RandomForestClassifier, model_path: str = "models/sample_model.pkl") -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        model_path: Path where to save the model
    """
    # Create models directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def main():
    """Main training function."""
    logger.info("Starting model training...")
    
    # Generate data
    logger.info("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=5)
    
    # Train model
    logger.info("Training Random Forest model...")
    model = train_model(X, y)
    
    # Save model
    logger.info("Saving model...")
    save_model(model)
    
    logger.info("Training completed successfully!")
    
    # Print model info
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Number of features: {model.n_features_in_}")
    logger.info(f"Number of classes: {len(model.classes_)}")

if __name__ == "__main__":
    main() 