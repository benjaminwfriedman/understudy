#!/usr/bin/env python3
"""
SLM Inference Service

Lightweight inference service that loads models from Model Broker and serves inference requests.
Can run in two modes:
1. Batch mode: Process evaluation batches and return results
2. Endpoint mode: Serve real-time inference requests
"""

import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
VERSION = os.getenv("VERSION")
EVAL_BATCH_ID = os.getenv("EVAL_BATCH_ID")
MODE = os.getenv("MODE", "endpoint")  # "endpoint" or "batch"
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend-service:8000")
MODEL_BROKER_URL = os.getenv("MODEL_BROKER_URL", "http://model-broker-service:8003")

# Global model and tokenizer
model = None
tokenizer = None


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stop_sequences: Optional[List[str]] = None


class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    processing_time: float


class BatchEvaluationRequest(BaseModel):
    prompts: List[str]
    train_id: str
    evaluation_batch_id: str


app = FastAPI(
    title="Understudy SLM Inference Service",
    description="Small Language Model inference service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_model():
    """Load the model from Model Broker or local storage."""
    global model, tokenizer
    
    try:
        if ENDPOINT_ID and VERSION:
            # Try to load from Model Broker first
            model_path = await download_model_from_broker()
            if not model_path:
                # Fall back to local model path
                model_path = f"{MODEL_PATH}/{ENDPOINT_ID}/v{VERSION}"
        else:
            # Use default model path
            model_path = MODEL_PATH
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


async def download_model_from_broker() -> Optional[str]:
    """Download model from Model Broker if needed."""
    try:
        # Check if model exists locally
        local_path = f"{MODEL_PATH}/{ENDPOINT_ID}/v{VERSION}"
        model_file = f"{local_path}/model.safetensors"
        
        if os.path.exists(model_file):
            logger.info(f"Model already exists locally: {model_file}")
            return local_path
        
        # Download from Model Broker
        logger.info(f"Downloading model {ENDPOINT_ID}/v{VERSION} from Model Broker")
        
        os.makedirs(local_path, exist_ok=True)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MODEL_BROKER_URL}/stream-model/{ENDPOINT_ID}/{VERSION}",
                timeout=300.0  # 5 minute timeout
            )
            
            if response.status_code == 200:
                with open(model_file, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                
                logger.info(f"Model downloaded successfully to {model_file}")
                return local_path
            else:
                logger.error(f"Failed to download model: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None


async def generate_text(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate text using the loaded model."""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        
        return {
            "text": generated_text.strip(),
            "tokens_generated": tokens_generated,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize the inference service."""
    logger.info(f"Starting SLM Inference Service in {MODE} mode")
    
    # Load model
    success = await load_model()
    if not success:
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")
    
    # If in batch mode, start processing immediately
    if MODE == "batch" and EVAL_BATCH_ID:
        asyncio.create_task(process_evaluation_batch())


async def process_evaluation_batch():
    """Process evaluation batch in batch mode."""
    try:
        logger.info(f"Processing evaluation batch: {EVAL_BATCH_ID}")
        
        # Get evaluation prompts from backend
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/evaluation/{EVAL_BATCH_ID}/prompts"
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get evaluation prompts: {response.status_code}")
                return
            
            data = response.json()
            prompts = data["prompts"]
            train_id = data["train_id"]
        
        logger.info(f"Processing {len(prompts)} evaluation prompts")
        
        # Generate responses for all prompts
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            result = await generate_text(prompt, max_tokens=150, temperature=0.7)
            results.append({
                "prompt": prompt,
                "response": result["text"],
                "processing_time": result["processing_time"]
            })
        
        # Send results back to backend
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/evaluation/{EVAL_BATCH_ID}/results",
                json={
                    "train_id": train_id,
                    "evaluation_batch_id": EVAL_BATCH_ID,
                    "results": results
                }
            )
            
            if response.status_code == 200:
                logger.info("Evaluation results sent to backend successfully")
            else:
                logger.error(f"Failed to send results: {response.status_code}")
        
        logger.info("Batch evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Understudy SLM Inference Service",
        "status": "healthy",
        "mode": MODE,
        "endpoint_id": ENDPOINT_ID,
        "version": VERSION,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: InferenceRequest):
    """
    OpenAI-compatible chat completions endpoint.
    This allows the service to be used as a drop-in replacement for OpenAI API.
    """
    if MODE != "endpoint":
        raise HTTPException(status_code=400, detail="Chat completions only available in endpoint mode")
    
    try:
        result = await generate_text(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Format response in OpenAI style
        return {
            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": f"{ENDPOINT_ID}-v{VERSION}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Would need to implement token counting
                "completion_tokens": result["tokens_generated"],
                "total_tokens": result["tokens_generated"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference")
async def inference(request: InferenceRequest):
    """Direct inference endpoint."""
    if MODE != "endpoint":
        raise HTTPException(status_code=400, detail="Inference only available in endpoint mode")
    
    try:
        result = await generate_text(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return InferenceResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "endpoint_id": ENDPOINT_ID,
        "version": VERSION,
        "model_path": MODEL_PATH,
        "mode": MODE,
        "model_loaded": model is not None,
        "device": str(next(model.parameters()).device) if model else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )