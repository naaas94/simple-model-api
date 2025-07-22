# Simple Model API

A production-ready FastAPI service for serving ML models with Docker containerization, comprehensive testing, and CI/CD pipeline.

[![CI/CD Pipeline](https://github.com/yourusername/simple-model-api/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/simple-model-api/actions)
[![Test Coverage](https://codecov.io/gh/yourusername/simple-model-api/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/simple-model-api)
[![Docker Image](https://img.shields.io/badge/docker-latest-blue.svg)](https://ghcr.io/yourusername/simple-model-api)

## Features

- **FastAPI-based REST API** with automatic OpenAPI documentation
- **ML Model Serving** with scikit-learn model support
- **Docker Containerization** with multi-stage builds
- **Comprehensive Testing** with pytest and coverage reporting
- **CI/CD Pipeline** with GitHub Actions
- **Input Validation** with Pydantic schemas
- **Health Checks** and monitoring endpoints
- **Prometheus Metrics** for observability
- **Modular Architecture** for easy extension

## Requirements

- Python 3.9+
- Docker (for containerized deployment)
- Git

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/simple-model-api.git
   cd simple-model-api
   ```

2. **Install dependencies**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

3. **Train the sample model**
   ```bash
   make train-model
   # or
   python scripts/train_model.py
   ```

4. **Run the service**
   ```bash
   make run
   # or
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics

### Docker Deployment

1. **Build the Docker image**
   ```bash
   make build
   # or
   docker build -t simple-model-api .
   ```

2. **Run the container**
   ```bash
   make docker-run
   # or
   docker run -p 8000:8000 simple-model-api
   ```

## API Documentation

### Endpoints

#### `GET /`
Returns basic API information.

**Response:**
```json
{
  "message": "Simple Model API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

#### `GET /health`
Health check endpoint for liveness/readiness probes.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1640995200.0
}
```

#### `GET /metrics`
Prometheus metrics endpoint for monitoring.

**Response:** Prometheus-formatted metrics

#### `POST /predict`
Make predictions using the loaded ML model.

**Request:**
```json
{
  "features": [1.0, 2.0, 3.0, 4.0, 5.0]
}
```

**Response:**
```json
{
  "prediction": 0.85,
  "model_version": "1.0.0",
  "processing_time": 0.0123
}
```

### Example Usage

#### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

#### Using Python requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Testing

### Run Tests
```bash
# Run all tests
make test

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Code Quality
```bash
# Linting
make lint

# Format code
make format

# Full CI check
make ci
```

## Project Structure

```
simple-model-api/
├── app/                    # Application code
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration settings
│   ├── models/            # Model management
│   │   ├── __init__.py
│   │   └── model_manager.py
│   └── schemas/           # Pydantic schemas
│       ├── __init__.py
│       └── prediction.py
├── tests/                 # Test suite
│   ├── __init__.py
│   └── test_api.py
├── scripts/               # Utility scripts
│   └── train_model.py
├── models/                # Model files (generated)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── Makefile              # Development tasks
├── .github/workflows/    # CI/CD pipeline
└── README.md
```

## Configuration

The application can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/sample_model.pkl` | Path to the ML model file |
| `MODEL_VERSION` | `1.0.0` | Model version string |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `False` | Debug mode |

## Deployment

### Docker Compose
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

### Kubernetes
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

## Monitoring

The service exposes Prometheus metrics at `/metrics` for monitoring:

- `predictions_total`: Total number of predictions made
- `prediction_duration_seconds`: Time spent processing predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for ML capabilities
- [Docker](https://www.docker.com/) for containerization
- [GitHub Actions](https://github.com/features/actions) for CI/CD

## Support

For support and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the test examples in `tests/`
