import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from app.main import app
from app.models.model_manager import ModelManager
from app.schemas.prediction import PredictionRequest, PredictionResponse

# Test client
client = TestClient(app)

class TestHealthEndpoint:
    """Test cases for the health endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

class TestMetricsEndpoint:
    """Test cases for the metrics endpoint."""
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for expected metrics
        content = response.text
        assert "predictions_total" in content
        assert "prediction_duration_seconds" in content

class TestRootEndpoint:
    """Test cases for the root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Simple Model API"
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs"]
        assert "/health" in data["health"]
        assert "/predict" in data["predict"]

class TestPredictionEndpoint:
    """Test cases for the prediction endpoint."""
    
    @patch('app.main.model_manager')
    def test_prediction_success(self, mock_model_manager):
        """Test successful prediction request."""
        # Mock the model manager
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.predict = AsyncMock(return_value=0.85)
        mock_model_manager.get_model_version.return_value = "1.0.0"
        
        # Test data
        test_data = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert "processing_time" in data
        assert data["prediction"] == 0.85
        assert data["model_version"] == "1.0.0"
        assert isinstance(data["processing_time"], float)
    
    @patch('app.main.model_manager')
    def test_prediction_model_not_loaded(self, mock_model_manager):
        """Test prediction when model is not loaded."""
        mock_model_manager.is_model_loaded.return_value = False
        
        test_data = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    def test_prediction_invalid_input(self):
        """Test prediction with invalid input."""
        # Test with non-numeric features
        test_data = {
            "features": [1.0, "invalid", 3.0, 4.0, 5.0]
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_empty_features(self):
        """Test prediction with empty features list."""
        test_data = {
            "features": []
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_too_many_features(self):
        """Test prediction with too many features."""
        test_data = {
            "features": [1.0] * 101  # More than max_length=100
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error

class TestModelManager:
    """Test cases for the ModelManager class."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing."""
        return ModelManager("test_model.pkl")
    
    def test_model_manager_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager.model_path.name == "test_model.pkl"
        assert model_manager.model is None
        assert model_manager.model_version == "1.0.0"
        assert not model_manager.is_model_loaded()
    
    def test_get_model_version(self, model_manager):
        """Test getting model version."""
        assert model_manager.get_model_version() == "1.0.0"
    
    def test_get_model_info_not_loaded(self, model_manager):
        """Test getting model info when model is not loaded."""
        info = model_manager.get_model_info()
        assert info["status"] == "not_loaded"
    
    @patch('joblib.load')
    def test_load_model_success(self, mock_joblib_load, model_manager):
        """Test successful model loading."""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            asyncio.run(model_manager.load_model())
        
        assert model_manager.is_model_loaded()
        assert model_manager.model == mock_model
    
    @patch('pathlib.Path.exists')
    def test_load_model_file_not_found(self, mock_exists, model_manager):
        """Test model loading when file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            asyncio.run(model_manager.load_model())
    
    @patch('joblib.load')
    def test_predict_success(self, mock_joblib_load, model_manager):
        """Test successful prediction."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.85])
        mock_joblib_load.return_value = mock_model
        
        # Load model
        with patch('pathlib.Path.exists', return_value=True):
            asyncio.run(model_manager.load_model())
        
        # Test prediction
        features = [1.0, 2.0, 3.0, 4.0, 5.0]
        prediction = asyncio.run(model_manager.predict(features))
        
        assert prediction == 0.85
        mock_model.predict.assert_called_once()
    
    def test_predict_model_not_loaded(self, model_manager):
        """Test prediction when model is not loaded."""
        features = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            asyncio.run(model_manager.predict(features))

class TestSchemas:
    """Test cases for Pydantic schemas."""
    
    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        data = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        request = PredictionRequest(**data)
        assert request.features == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_prediction_request_invalid_features(self):
        """Test prediction request with invalid features."""
        data = {
            "features": [1.0, float('inf'), 3.0, 4.0, 5.0]
        }
        
        with pytest.raises(ValueError, match="All features must be finite numbers"):
            PredictionRequest(**data)
    
    def test_prediction_response_valid(self):
        """Test valid prediction response."""
        data = {
            "prediction": 0.85,
            "model_version": "1.0.0",
            "processing_time": 0.0123
        }
        
        response = PredictionResponse(**data)
        assert response.prediction == 0.85
        assert response.model_version == "1.0.0"
        assert response.processing_time == 0.0123

# Integration tests
class TestIntegration:
    """Integration tests for the full API flow."""
    
    @pytest.mark.asyncio
    async def test_full_prediction_flow(self):
        """Test the complete prediction flow with a real model."""
        # This test would require a real model file
        # For now, we'll test the API structure
        async with httpx.AsyncClient(base_url="http://test") as ac:
            response = await ac.get("/health")
            assert response.status_code == 200 