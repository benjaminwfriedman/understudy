#!/usr/bin/env python3
"""
Evaluation Service

Calculates semantic similarity between LLM and SLM responses using sentence transformers.
Processes evaluation jobs from Redis queue and returns similarity scores to Backend.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import redis.asyncio as redis
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis-service:6379")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend-service:8000")

# Global variables
similarity_model = None
redis_client = None


class EvaluationPair(BaseModel):
    llm_response: str
    slm_response: str


class EvaluationRequest(BaseModel):
    train_id: str
    evaluation_pairs: List[EvaluationPair]


class EvaluationResult(BaseModel):
    train_id: str
    semantic_similarity_score: float
    evaluation_count: int
    processing_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global similarity_model, redis_client
    
    # Startup
    logger.info("Starting Evaluation Service...")
    
    # Load sentence transformer model
    logger.info("Loading sentence transformer model...")
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Model loaded successfully")
    
    # Initialize Redis connection
    redis_client = redis.from_url(REDIS_URL)
    await redis_client.ping()
    logger.info("Connected to Redis")
    
    # Start evaluation queue worker
    asyncio.create_task(evaluation_queue_worker())
    logger.info("Evaluation queue worker started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Evaluation Service...")
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="Understudy Evaluation Service",
    description="Semantic similarity evaluation using sentence transformers",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_similarity(llm_response: str, slm_response: str) -> float:
    """Calculate semantic similarity between two responses using sentence transformers."""
    try:
        # Encode both responses separately to avoid batch operations
        embedding1 = similarity_model.encode(llm_response, convert_to_tensor=False)
        embedding2 = similarity_model.encode(slm_response, convert_to_tensor=False)
        
        # Calculate cosine similarity using numpy instead of torch
        import numpy as np
        from numpy.linalg import norm
        
        # Convert to numpy arrays if they aren't already
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Manual cosine similarity calculation
        dot_product = np.dot(emb1, emb2)
        norm1 = norm(emb1)
        norm2 = norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        similarity = max(0.0, min(1.0, float(similarity)))
        
        logger.debug(f"Calculated similarity: {similarity:.3f}")
        return similarity
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        # Fallback to simple text similarity
        return 0.0


async def process_evaluation_batch(evaluation_pairs: List[EvaluationPair]) -> float:
    """Process a batch of evaluation pairs and return average similarity."""
    similarities = []
    
    for pair in evaluation_pairs:
        similarity = calculate_similarity(pair.llm_response, pair.slm_response)
        similarities.append(similarity)
        
        # Log individual scores for debugging
        logger.debug(f"Similarity: {similarity:.3f} | LLM: '{pair.llm_response[:50]}...' | SLM: '{pair.slm_response[:50]}...'")
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    logger.info(f"Processed {len(similarities)} pairs, average similarity: {avg_similarity:.3f}")
    return float(avg_similarity)


async def evaluation_queue_worker():
    """Background worker to process evaluation queue."""
    logger.info("Evaluation queue worker started")
    
    while True:
        try:
            # Check for pending evaluation jobs
            job_data = await redis_client.brpop("evaluation_queue", timeout=30)
            
            if job_data:
                import json
                _, job_json = job_data
                job = json.loads(job_json)
                
                train_id = job["train_id"]
                evaluation_pairs = [EvaluationPair(**pair) for pair in job["evaluation_pairs"]]
                
                logger.info(f"Processing evaluation for {train_id} with {len(evaluation_pairs)} pairs")
                
                start_time = datetime.now()
                
                # Process evaluation
                avg_similarity = await process_evaluation_batch(evaluation_pairs)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Send results to backend
                await send_results_to_backend(
                    train_id=train_id,
                    similarity_score=avg_similarity,
                    evaluation_count=len(evaluation_pairs),
                    processing_time=processing_time
                )
                
                logger.info(f"Evaluation completed for {train_id}: {avg_similarity:.3f}")
            
        except Exception as e:
            logger.error(f"Error in evaluation queue worker: {e}")
            await asyncio.sleep(5)


async def send_results_to_backend(
    train_id: str,
    similarity_score: float,
    evaluation_count: int,
    processing_time: float
):
    """Send evaluation results back to the backend."""
    try:
        result = EvaluationResult(
            train_id=train_id,
            semantic_similarity_score=similarity_score,
            evaluation_count=evaluation_count,
            processing_time=processing_time
        )
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/lifecycle/models/{train_id}/similarity-score",
                json=result.dict()
            )
            
            if response.status_code == 200:
                logger.info(f"Results sent to backend for {train_id}")
            else:
                logger.error(f"Failed to send results to backend: {response.status_code} {response.text}")
                
    except Exception as e:
        logger.error(f"Error sending results to backend: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Understudy Evaluation Service",
        "status": "healthy",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/evaluate")
async def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Queue an evaluation job for processing.
    This endpoint receives evaluation requests and adds them to the Redis queue.
    """
    logger.info(f"Received evaluation request for {request.train_id} with {len(request.evaluation_pairs)} pairs")
    
    try:
        # Add job to Redis queue
        import json
        job_data = json.dumps({
            "train_id": request.train_id,
            "evaluation_pairs": [pair.dict() for pair in request.evaluation_pairs],
            "queued_at": datetime.now().isoformat()
        })
        
        await redis_client.lpush("evaluation_queue", job_data)
        
        return {
            "message": "Evaluation job queued successfully",
            "train_id": request.train_id,
            "evaluation_count": len(request.evaluation_pairs),
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Error queuing evaluation for {request.train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue evaluation: {str(e)}")


@app.post("/api/v1/evaluate/sync")
async def evaluate_sync(request: EvaluationRequest):
    """
    Synchronous evaluation endpoint for immediate processing.
    Use this for small batches that need immediate results.
    """
    logger.info(f"Processing synchronous evaluation for {request.train_id}")
    
    try:
        start_time = datetime.now()
        
        # Process evaluation immediately
        avg_similarity = await process_evaluation_batch(request.evaluation_pairs)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = EvaluationResult(
            train_id=request.train_id,
            semantic_similarity_score=avg_similarity,
            evaluation_count=len(request.evaluation_pairs),
            processing_time=processing_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in synchronous evaluation for {request.train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/v1/queue/status")
async def queue_status():
    """Get the current evaluation queue status."""
    try:
        queue_length = await redis_client.llen("evaluation_queue")
        
        return {
            "queue_length": queue_length,
            "status": "healthy" if queue_length < 100 else "busy",
            "model_loaded": similarity_model is not None
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
