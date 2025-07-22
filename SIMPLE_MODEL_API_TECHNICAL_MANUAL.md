---
noteId: simple-model-api-technical-manual
tags:
  - SMA
---

# SIMPLE MODEL API TECHNICAL MANUAL
## Production-Ready ML Model Serving System - Complete System Reference

**Version:** 1.0.0  
**Author:** Alejandro Garay  
**Last Updated:** 2025-01-27

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Core Data Structures](#core-data-structures)
4. [Configuration Management](#configuration-management)
5. [API Layer](#api-layer)
6. [Model Management Layer](#model-management-layer)
7. [Validation Layer](#validation-layer)
8. [System Integration](#system-integration)
9. [Usage Examples](#usage-examples)
10. [Performance Characteristics](#performance-characteristics)
11. [Error Handling](#error-handling)
12. [Extension Points](#extension-points)

---

## SYSTEM OVERVIEW

Simple Model API is a production-ready FastAPI service for serving machine learning models with comprehensive containerization, monitoring, and CI/CD capabilities. The system operates on the principle of **asynchronous model serving** - providing high-performance, scalable ML inference with full observability.

### Key Design Principles

- **Production Ready**: Docker containerization, health checks, and monitoring
- **Asynchronous Processing**: Non-blocking model inference with thread pool execution
- **Comprehensive Validation**: Pydantic schemas for input/output validation
- **Observability**: Prometheus metrics and structured logging
- **Modular Architecture**: Clean separation of concerns with extensible components
- **Developer Experience**: Complete testing suite and development tooling
- **LLM Integration**: Daily word generation using an open-source LLM, enhancing dynamic content delivery

### Core Capabilities

✅ **Implemented in v1.0:**
- FastAPI-based REST API with automatic OpenAPI documentation
- Asynchronous ML model serving with scikit-learn support
- Docker containerization with multi-stage builds
- Comprehensive testing with pytest and coverage reporting
- Prometheus metrics for monitoring and observability
- Input validation with Pydantic schemas
- Health checks and readiness probes
- CI/CD pipeline with GitHub Actions
- Development tooling with Makefile automation
- **LLM Integration for daily word generation**

🚨 **Critical Gaps (Future Work):**
- Model versioning and A/B testing capabilities
- Model performance monitoring and drift detection
- Advanced caching strategies for high-throughput scenarios
- Multi-model serving and model routing
- Advanced authentication and authorization
- Model explainability and interpretability endpoints

---

## ARCHITECTURE & DATA FLOW

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API LAYER     │    │ MODEL MANAGER   │    │ VALIDATION      │
│                 │    │                 │    │                 │
│ • FastAPI       │───▶│ • Async Loading │───▶│ • Pydantic      │
│ • Endpoints     │    │ • Thread Pool   │    │ • Input/Output  │
│ • Middleware    │    │ • Model Cache   │    │ • Type Safety   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MONITORING    │    │   CONFIG        │    │   DEPLOYMENT    │
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • Environment   │    │ • Docker        │
│ • Health Checks │    │ • Settings      │    │ • Kubernetes    │
│ • Logging       │    │ • LLM Update    │    │ • CI/CD         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow Sequence

1. **Request Reception**: HTTP request arrives at FastAPI endpoint
2. **Input Validation**: Pydantic validates request schema and data types
3. **Model Check**: Verify model is loaded and ready for inference
4. **Async Processing**: Submit prediction to thread pool executor
5. **Model Inference**: Execute prediction in separate thread
6. **LLM Word Update**: Scheduled task updates the word using the LLM
7. **Response Validation**: Validate model output with Pydantic
8. **Metrics Recording**: Update Prometheus metrics and timing
9. **Response Return**: Return structured JSON response to client

---

## CORE DATA STRUCTURES

### PredictionRequest (app/schemas/prediction.py)

```python
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        description="Input features for prediction",
        min_length=1,
        max_length=100
    )
```

**Purpose**: Validates and structures incoming prediction requests.

**Key Methods**: 
- `validate_features()`: Ensures all features are finite numbers

**Usage**: Used by FastAPI endpoints to validate incoming JSON requests.

### PredictionResponse (app/schemas/prediction.py)

```python
class PredictionResponse(BaseModel):
    prediction: Union[float, int, str] = Field(
        description="Model prediction result"
    )
    model_version: str = Field(
        description="Version of the model used for prediction"
    )
    processing_time: float = Field(
        description="Time taken to process the prediction in seconds"
    )
```

**Purpose**: Structures and validates prediction responses.

**Key Methods**: None (data container)

**Usage**: Returned by prediction endpoints with standardized response format.

### Settings (app/config.py)

```python
class Settings(BaseSettings):
    app_name: str = "Simple Model API"
    app_version: str = "1.0.0"
    debug: bool = False
    model_path: str = "models/sample_model.pkl"
    model_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = ["*"]
```

**Purpose**: Centralized configuration management with environment variable support.

**Key Methods**: None (configuration container)

**Usage**: Loaded at application startup, used throughout the system.

---

## CONFIGURATION MANAGEMENT

### Settings (app/config.py)

**Purpose**: Manages application configuration with environment variable support.

**Environment Variables**:
- `MODEL_PATH`: Path to ML model file (default: "models/sample_model.pkl")
- `MODEL_VERSION`: Model version string (default: "1.0.0")
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: False)
- `UPDATE_INTERVAL_HOURS`: Interval in hours for updating the LLM-generated word (default: 24)
- `LLM_MODEL_PATH`: Path to the LLM model file

**Usage Example**:
```python
from app.config import settings

print(f"Model path: {settings.model_path}")
print(f"Server port: {settings.port}")
```

---

## API LAYER

### FastAPI Application (app/main.py)

**Purpose**: Main FastAPI application with endpoints, middleware, and monitoring.

**Class Methods**:

#### `__init__()`
- **Parameters**: None
- **Purpose**: Initialize FastAPI app with configuration
- **Returns**: None
- **Configuration**:
  - Title: "Simple Model API"
  - Description: "A production-ready FastAPI service for serving ML models"
  - Version: "1.0.0"
  - Documentation URLs: "/docs", "/redoc"

#### `add_middleware()`
- **Parameters**: None
- **Purpose**: Add CORS middleware for cross-origin requests
- **Returns**: None
- **Configuration**: Allow all origins, methods, and headers

#### `startup_event()`
- **Parameters**: None
- **Purpose**: Initialize model on application startup
- **Returns**: None
- **Operations**: Load model asynchronously, log success/failure

### API Endpoints

#### `GET /health`
**Purpose**: Health check endpoint for liveness/readiness probes.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1640995200.0
}
```

**Usage**: Used by container orchestrators and load balancers.

#### `GET /metrics`
**Purpose**: Prometheus metrics endpoint for monitoring.

**Response**: Prometheus-formatted metrics text

**Metrics**:
- `predictions_total`: Counter of total predictions
- `prediction_duration_seconds`: Histogram of prediction times

#### `POST /predict`
**Purpose**: Make predictions using the loaded ML model.

**Request**:
```json
{
  "features": [1.0, 2.0, 3.0, 4.0, 5.0]
}
```

**Response**:
```json
{
  "prediction": 0.85,
  "model_version": "1.0.0",
  "processing_time": 0.0123
}
```

**Error Handling**:
- 400: Invalid input features
- 503: Model not loaded
- 500: Internal server error

#### `GET /`
**Purpose**: Root endpoint with API information.

**Response**:
```json
{
  "message": "Simple Model API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

**Usage Example**:
```python
import httpx

async with httpx.AsyncClient() as client:
    # Health check
    response = await client.get("http://localhost:8000/health")
    health = response.json()
    
    # Make prediction
    data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
    response = await client.post("http://localhost:8000/predict", json=data)
    result = response.json()
```

#### `GET /today`
**Purpose**: Returns the current word generated by the LLM.

**Response**:
```json
{
  "word": "current_word"
}
```

**Usage**: Provides a daily updated word for display or other uses.

---

## MODEL MANAGEMENT LAYER

### ModelManager (app/models/model_manager.py)

**Purpose**: Manages ML model loading, caching, and asynchronous prediction.

**Class Methods**:

#### `__init__(model_path: str = "models/sample_model.pkl")`
- **Parameters**: `model_path` - Path to model file
- **Purpose**: Initialize model manager with model path
- **Returns**: None
- **State**: Sets up model path, version, and loading status

#### `load_model() -> None`
- **Parameters**: None
- **Purpose**: Load ML model asynchronously using thread pool
- **Returns**: None
- **Throws**: `FileNotFoundError` if model file doesn't exist
- **Operations**:
  1. Check if model file exists
  2. Load model using joblib in thread pool
  3. Set model loaded flag
  4. Log success/failure

#### `is_model_loaded() -> bool`
- **Parameters**: None
- **Purpose**: Check if model is loaded and ready
- **Returns**: True if model is loaded, False otherwise

#### `get_model_version() -> str`
- **Parameters**: None
- **Purpose**: Get current model version
- **Returns**: Model version string

#### `predict(features: List[float]) -> Union[float, int, str]`
- **Parameters**: `features` - Input features for prediction
- **Purpose**: Make prediction asynchronously
- **Returns**: Model prediction result
- **Throws**: `RuntimeError` if model not loaded, `ValueError` on prediction error
- **Operations**:
  1. Validate model is loaded
  2. Convert features to numpy array
  3. Execute prediction in thread pool
  4. Return prediction result

#### `get_model_info() -> dict`
- **Parameters**: None
- **Purpose**: Get information about loaded model
- **Returns**: Dictionary with model status and metadata
- **Response**: `{"status": "loaded", "model_path": "...", "model_version": "...", "model_type": "..."}`

**Usage Example**:
```python
from app.models.model_manager import ModelManager

# Initialize manager
manager = ModelManager("models/my_model.pkl")

# Load model
await manager.load_model()

# Check status
if manager.is_model_loaded():
    # Make prediction
    prediction = await manager.predict([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Prediction: {prediction}")

# Get model info
info = manager.get_model_info()
print(f"Model type: {info['model_type']}")
```

---

## VALIDATION LAYER

### Pydantic Schemas (app/schemas/prediction.py)

**Purpose**: Provides input/output validation and type safety.

**Class Methods**:

#### `PredictionRequest.validate_features()`
- **Parameters**: `v` - Features list to validate
- **Purpose**: Validate that all features are finite numbers
- **Returns**: Validated features list
- **Throws**: `ValueError` if any feature is invalid
- **Validation**:
  - All features must be int or float
  - All features must be finite (not inf, -inf, or NaN)
  - Features list length between 1 and 100

#### `PredictionRequest` Field Validators
- **min_length**: 1 (at least one feature required)
- **max_length**: 100 (maximum 100 features)
- **description**: "Input features for prediction"

#### `PredictionResponse` Field Validators
- **prediction**: Union[float, int, str] (flexible prediction type)
- **model_version**: str (version identifier)
- **processing_time**: float (timing information)

**Usage Example**:
```python
from app.schemas.prediction import PredictionRequest, PredictionResponse

# Validate request
try:
    request = PredictionRequest(features=[1.0, 2.0, 3.0, 4.0, 5.0])
    print("Valid request")
except ValueError as e:
    print(f"Invalid request: {e}")

# Create response
response = PredictionResponse(
    prediction=0.85,
    model_version="1.0.0",
    processing_time=0.0123
)
```

---

## SYSTEM INTEGRATION

### Complete Pipeline Flow

The Simple Model API integrates all components through a sequential pipeline:

1. **Application Startup**
   ```python
   app = FastAPI(title="Simple Model API", version="1.0.0")
   app.add_middleware(CORSMiddleware, allow_origins=["*"])
   model_manager = ModelManager()
   ```

2. **Model Initialization**
   ```python
   @app.on_event("startup")
   async def startup_event():
       await model_manager.load_model()
   ```

3. **Request Processing**
   ```python
   @app.post("/predict")
   async def predict(request: PredictionRequest):
       if not model_manager.is_model_loaded():
           raise HTTPException(status_code=503)
       
       prediction = await model_manager.predict(request.features)
       return PredictionResponse(prediction=prediction, ...)
   ```

4. **Response Generation**
   ```python
   return PredictionResponse(
       prediction=prediction,
       model_version=model_manager.get_model_version(),
       processing_time=duration
   )
   ```

### Data Transformation Chain

```
HTTP Request → Pydantic Validation → Model Check → Async Prediction → Response Validation → HTTP Response
```

### Error Handling Strategy

- **Validation Errors**: Pydantic automatically returns 422 for invalid input
- **Model Errors**: 503 when model not loaded, 500 for prediction failures
- **System Errors**: Comprehensive logging with structured error messages
- **Network Errors**: Graceful handling with appropriate HTTP status codes

---

## USAGE EXAMPLES

### Basic API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.json()['status']}")

# Make prediction
data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
```

### Async Client Usage

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        # Health check
        response = await client.get("http://localhost:8000/health")
        health = response.json()
        
        # Make prediction
        data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
        response = await client.post("http://localhost:8000/predict", json=data)
        result = response.json()
        
        print(f"Health: {health['status']}")
        print(f"Prediction: {result['prediction']}")

asyncio.run(main())
```

### Docker Deployment

```bash
# Build image
docker build -t simple-model-api .

# Run container
docker run -p 8000:8000 simple-model-api

# With custom model
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/models/custom_model.pkl \
  -v $(pwd)/models:/app/models \
  simple-model-api
```

### Docker Compose Deployment

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=models/sample_model.pkl
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: simple-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: simple-model-api
  template:
    metadata:
      labels:
        app: simple-model-api
    spec:
      containers:
      - name: api
        image: ghcr.io/yourusername/simple-model-api:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## PERFORMANCE CHARACTERISTICS

### API Performance

**Request Processing**:
- Average response time: 10-50ms (depending on model complexity)
- Throughput: 100-1000 requests/second (single instance)
- Memory usage: 100-500MB (depending on model size)

**Async Processing**:
- Thread pool size: Default system thread pool
- Non-blocking I/O: All endpoints are async
- Concurrent requests: Limited by thread pool size

### Model Performance

**Loading Time**:
- Small models (<100MB): 1-5 seconds
- Large models (>500MB): 10-30 seconds
- Memory usage: 2-4x model file size

**Inference Time**:
- Simple models: 1-10ms per prediction
- Complex models: 10-100ms per prediction
- Batch processing: Not currently supported

### Scalability

**Horizontal Scaling**:
- Stateless design enables easy scaling
- Load balancer can distribute requests
- Each instance loads model independently

**Vertical Scaling**:
- Limited by single-threaded model inference
- Memory usage scales with model size
- CPU usage depends on model complexity

### Monitoring Metrics

**Prometheus Metrics**:
- `predictions_total`: Total number of predictions
- `prediction_duration_seconds`: Prediction timing histogram
- Custom metrics can be added easily

**Health Check Metrics**:
- Response time: <100ms for health endpoint
- Model loading status: Boolean flag
- System uptime: Timestamp tracking

---

## ERROR HANDLING

### Common Error Scenarios

1. **Model File Not Found**
   ```python
   FileNotFoundError: Model file not found: models/sample_model.pkl
   ```
   **Solution**: Ensure model file exists or train model first

2. **Invalid Input Features**
   ```python
   ValueError: All features must be finite numbers
   ```
   **Solution**: Validate input data before sending request

3. **Model Not Loaded**
   ```python
   HTTPException: 503 - Model not loaded
   ```
   **Solution**: Wait for application startup or check model loading

4. **Memory Issues**
   ```python
   MemoryError: Unable to load model
   ```
   **Solution**: Increase container memory limits or use smaller model

5. **Network Timeouts**
   ```python
   TimeoutError: Request timed out
   ```
   **Solution**: Increase timeout settings or optimize model performance

### Error Recovery Strategies

- **Graceful Degradation**: Return appropriate HTTP status codes
- **Comprehensive Logging**: Log all errors with context
- **Health Checks**: Monitor system health and model status
- **Retry Logic**: Implement retry mechanisms for transient failures
- **Circuit Breaker**: Prevent cascading failures in distributed systems

### Error Response Format

```json
{
  "detail": "Model not loaded",
  "status_code": 503,
  "timestamp": "2025-01-27T10:00:00Z"
}
```

---

## EXTENSION POINTS

### Adding New Model Types

Extend ModelManager for different model formats:

```python
class CustomModelManager(ModelManager):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model_type = "custom"
    
    def _load_model_sync(self) -> object:
        """Load custom model format."""
        if self.model_path.suffix == ".onnx":
            return self._load_onnx_model()
        elif self.model_path.suffix == ".pb":
            return self._load_tensorflow_model()
        else:
            return super()._load_model_sync()
    
    def _load_onnx_model(self):
        import onnxruntime as ort
        return ort.InferenceSession(str(self.model_path))
```

### Adding New Endpoints

Extend FastAPI with custom endpoints:

```python
@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint."""
    results = []
    for request in requests:
        prediction = await model_manager.predict(request.features)
        results.append({
            "prediction": prediction,
            "model_version": model_manager.get_model_version()
        })
    return {"predictions": results}

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information."""
    return model_manager.get_model_info()
```

### Adding Authentication

Implement authentication middleware:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    token = credentials.credentials
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

@app.post("/predict")
async def predict(request: PredictionRequest, token: str = Depends(verify_token)):
    """Protected prediction endpoint."""
    # ... existing prediction logic
```

### Adding Caching

Implement prediction caching:

```python
from functools import lru_cache
import hashlib
import json

class CachedModelManager(ModelManager):
    def __init__(self, model_path: str, cache_size: int = 1000):
        super().__init__(model_path)
        self.cache_size = cache_size
    
    @lru_cache(maxsize=1000)
    def _cached_predict(self, features_hash: str):
        """Cached prediction using feature hash."""
        features = json.loads(features_hash)
        return self._predict_sync(np.array(features).reshape(1, -1))
    
    async def predict(self, features: List[float]) -> Union[float, int, str]:
        """Make cached prediction."""
        features_hash = json.dumps(features, sort_keys=True)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cached_predict, features_hash)
```

### Adding Model Versioning

Implement model version management:

```python
class VersionedModelManager(ModelManager):
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.current_version = None
        self.models = {}
    
    async def load_model_version(self, version: str):
        """Load specific model version."""
        model_path = self.model_dir / f"model_v{version}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        self.models[version] = await self._load_model_async(model_path)
        self.current_version = version
    
    async def predict_with_version(self, features: List[float], version: str):
        """Make prediction with specific model version."""
        if version not in self.models:
            await self.load_model_version(version)
        
        return await self._predict_with_model(features, self.models[version])
```

### Adding Model Monitoring

Implement model performance monitoring:

```python
class MonitoredModelManager(ModelManager):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.prediction_history = []
        self.error_count = 0
    
    async def predict(self, features: List[float]) -> Union[float, int, str]:
        """Make prediction with monitoring."""
        start_time = time.time()
        
        try:
            prediction = await super().predict(features)
            
            # Record successful prediction
            self.prediction_history.append({
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "features": features,
                "prediction": prediction,
                "success": True
            })
            
            return prediction
            
        except Exception as e:
            # Record failed prediction
            self.error_count += 1
            self.prediction_history.append({
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "features": features,
                "error": str(e),
                "success": False
            })
            raise
    
    def get_performance_stats(self):
        """Get model performance statistics."""
        if not self.prediction_history:
            return {"total_predictions": 0}
        
        successful = [p for p in self.prediction_history if p["success"]]
        failed = [p for p in self.prediction_history if not p["success"]]
        
        return {
            "total_predictions": len(self.prediction_history),
            "successful_predictions": len(successful),
            "failed_predictions": len(failed),
            "success_rate": len(successful) / len(self.prediction_history),
            "average_duration": sum(p["duration"] for p in successful) / len(successful) if successful else 0,
            "error_count": self.error_count
        }
```

### Adding LLM Features

Extend the system to support additional LLM features or models:

```python
class AdvancedLLMManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Additional initialization
    
    def generate_advanced_word(self):
        # Logic for advanced word generation
        return "advanced_word"
```

This allows for easy integration of new LLM capabilities or models.

---

## CONCLUSION

The Simple Model API provides a complete, production-ready framework for serving machine learning models with the following key strengths:

1. **Production Ready**: Docker containerization, health checks, monitoring, and CI/CD
2. **High Performance**: Asynchronous processing with thread pool execution
3. **Robust Validation**: Comprehensive input/output validation with Pydantic
4. **Observability**: Prometheus metrics, structured logging, and health endpoints
5. **Developer Experience**: Complete testing suite, documentation, and development tooling
6. **Extensible Architecture**: Clear extension points for custom functionality

The system is designed to be both immediately deployable with default settings and highly customizable for specific production requirements. The modular architecture ensures that improvements in one component (e.g., new model types, caching strategies) can be easily integrated without affecting other parts of the system.

For production deployment, consider implementing the missing features outlined in the roadmap section, particularly model versioning, advanced monitoring, and authentication capabilities. The system provides a solid foundation for building scalable, reliable ML model serving infrastructure.

---

## APPENDIX

### Development Commands

```bash
# Install dependencies
make install

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Train model
make train-model

# Run locally
make run

# Build Docker image
make build

# Run in Docker
make docker-run

# Full CI check
make ci
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/sample_model.pkl` | Path to ML model file |
| `MODEL_VERSION` | `1.0.0` | Model version string |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `False` | Debug mode |

### API Endpoints Summary

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| GET | `/` | API information | JSON |
| GET | `/health` | Health check | JSON |
| GET | `/metrics` | Prometheus metrics | Text |
| POST | `/predict` | Make prediction | JSON |

### Monitoring Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `predictions_total` | Counter | Total number of predictions |
| `prediction_duration_seconds` | Histogram | Prediction timing distribution |

### File Structure

```
simple-model-api/
├── app/                    # Application code
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration settings
│   ├── models/            # Model management
│   └── schemas/           # Pydantic schemas
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── models/                # Model files
├── monitoring/            # Monitoring configuration
├── examples/              # Usage examples
├── requirements.txt       # Dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service deployment
├── Makefile              # Development tasks
└── README.md             # Project documentation
``` 
noteId: "56b5b85061b511f09cda1b0c8520cb78"
tags: []

---

 