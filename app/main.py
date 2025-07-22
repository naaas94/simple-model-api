from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import logging

from app.models.model_manager import ModelManager
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.config import settings
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent processing prediction')

app = FastAPI(
    title="Simple Model API",
    description="A production-ready FastAPI service for serving ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Initialize the scheduler
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        await model_manager.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("startup")
async def start_scheduler():
    scheduler.start()

# Initialize the current state
current_state = {
    "prompt": "Continue this generation of a dystopian future:",
    "current_text": "",
    "today_word": "",
    "timestamp": ""
}

async def update_word():
    # Generate today's word using the LLM
    today_word = generate_word_llm()
    # Update the current state
    current_state["today_word"] = today_word
    current_state["current_text"] += " " + today_word
    current_state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    logger.info(f"Updated state: {current_state}")

scheduler.add_job(update_word, 'interval', hours=settings.UPDATE_INTERVAL_HOURS)

@app.get("/health")
async def health_check():
    """Health check endpoint for liveness/readiness probes."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_model_loaded(),
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction using the loaded ML model.
    
    Args:
        request: PredictionRequest containing the input features
        
    Returns:
        PredictionResponse containing the prediction result
    """
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Make prediction
        prediction = await model_manager.predict(request.features)
        
        # Record metrics
        duration = time.time() - start_time
        PREDICTION_COUNTER.inc()
        PREDICTION_DURATION.observe(duration)
        
        logger.info(f"Prediction completed in {duration:.4f}s")
        
        return PredictionResponse(
            prediction=prediction,
            model_version=model_manager.get_model_version(),
            processing_time=duration
        )
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/today")
async def get_today_word():
    # Return the current state
    return current_state

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Simple Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    } 