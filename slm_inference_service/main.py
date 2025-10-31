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
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
VERSION = os.getenv("VERSION")
EVAL_BATCH_ID = os.getenv("EVAL_BATCH_ID")
TRAIN_ID = os.getenv("TRAIN_ID")
MODE = os.getenv("MODE", "endpoint")  # "endpoint" or "batch"
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend-service:8000")
MODEL_BROKER_URL = os.getenv("MODEL_BROKER_URL", "http://model-broker-service:8003")
HF_TOKEN = os.getenv("HF_TOKEN")

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
        
        # Check if this is a LoRA adapter by looking for adapter_config.json
        adapter_config_path = f"{model_path}/adapter_config.json"
        is_lora_adapter = os.path.exists(adapter_config_path)
        
        if is_lora_adapter:
            logger.info("Detected LoRA adapter, loading base model + adapter")
            
            # Read adapter config to get base model
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path")
            if not base_model_name:
                raise Exception("LoRA adapter config missing base_model_name_or_path")
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # Load base model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                token=HF_TOKEN if HF_TOKEN else None
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                token=HF_TOKEN if HF_TOKEN else None
            )
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info("LoRA model loaded successfully")
            
        else:
            logger.info("Loading full model")
            # Load tokenizer and model normally
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Full model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


async def download_model_from_broker() -> Optional[str]:
    """Download model from Model Broker if needed."""
    try:
        # Check if model exists locally
        local_path = f"{MODEL_PATH}/{ENDPOINT_ID}/v{VERSION}"
        
        # Check for LoRA adapter files
        lora_files = ["adapter_config.json", "adapter_model.safetensors"]
        is_lora = all(os.path.exists(f"{local_path}/{file}") for file in lora_files)
        
        # Check for common Hugging Face model files
        required_files = ["config.json", "pytorch_model.bin"]
        model_exists = all(os.path.exists(f"{local_path}/{file}") for file in required_files)
        
        # Also check for safetensors format
        if not model_exists:
            safetensors_files = ["config.json", "model.safetensors"]
            model_exists = all(os.path.exists(f"{local_path}/{file}") for file in safetensors_files)
        
        # For LoRA adapters, we only need the adapter files
        if is_lora:
            model_exists = True
        
        if model_exists:
            logger.info(f"Model already exists locally: {local_path}")
            return local_path
        
        # Download from Model Broker
        logger.info(f"Downloading model {ENDPOINT_ID}/v{VERSION} from Model Broker")
        
        os.makedirs(local_path, exist_ok=True)
        
        async with httpx.AsyncClient() as client:
            # First try to get model info/metadata
            try:
                info_response = await client.get(f"{MODEL_BROKER_URL}/model-info/{ENDPOINT_ID}/{VERSION}")
                if info_response.status_code == 200:
                    model_info = info_response.json()
                    logger.info(f"Model info: {model_info}")
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
            
            # Download the model
            response = await client.get(
                f"{MODEL_BROKER_URL}/stream-model/{ENDPOINT_ID}/{VERSION}",
                timeout=300.0  # 5 minute timeout
            )
            
            if response.status_code == 200:
                # Check if it's a LoRA adapter (tar.gz) or regular model
                content_type = response.headers.get("content-type", "")
                model_type = response.headers.get("X-Model-Type", "")
                
                if model_type == "lora_adapter" or content_type == "application/gzip":
                    # Handle LoRA adapter tar.gz
                    logger.info("Downloading LoRA adapter archive")
                    import tarfile
                    import tempfile
                    
                    # Download to temp file
                    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                        async for chunk in response.aiter_bytes():
                            tmp_file.write(chunk)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Extract tar.gz to model directory
                        with tarfile.open(tmp_file_path, 'r:gz') as tar:
                            tar.extractall(local_path)
                        
                        logger.info(f"LoRA adapter extracted to {local_path}")
                        
                        # Verify we have the expected files
                        adapter_config_path = f"{local_path}/adapter_config.json"
                        adapter_model_path = f"{local_path}/adapter_model.safetensors"
                        
                        if not os.path.exists(adapter_config_path) or not os.path.exists(adapter_model_path):
                            raise Exception("Downloaded LoRA adapter missing required files")
                        
                        return local_path
                        
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
                            
                else:
                    # Handle regular model file
                    model_file = f"{local_path}/model.safetensors"
                    with open(model_file, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                    
                    logger.info(f"Model weights downloaded to {model_file}")
                    
                    # Create a minimal config.json for the model
                    config = {
                        "architectures": ["LlamaForCausalLM"],
                        "model_type": "llama",
                        "torch_dtype": "float16",
                        "transformers_version": "4.35.2"
                    }
                    
                    with open(f"{local_path}/config.json", "w") as f:
                        json.dump(config, f, indent=2)
                    
                    logger.info(f"Model setup completed at {local_path}")
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
    
    # In web server mode, just load the model and be ready for requests
    # Batch mode is handled directly in __main__


async def process_evaluation_batch():
    """Process evaluation batch in batch mode."""
    try:
        logger.info(f"Processing evaluation batch: {EVAL_BATCH_ID}")

        logger.info(f"TRAIN_ID: ")
        
        # Connect to Redis to get evaluation context
        logger.info(f"Connecting to REDIS: redis://redis-service:6379")
        redis_url = os.environ.get("REDIS_URL", "redis://redis-service:6379")
        redis_client = await redis.from_url(redis_url)
        logger.info(f"CONNECTED")
        # Get evaluation context from Redis
        # The evaluation_batch_id passed from backend is just the train_id
        logger.info(f"Using TRAIN_ID as train_id: {TRAIN_ID}")
        train_id = TRAIN_ID
        
        eval_context_key = f"evaluation_context:{train_id}"
        eval_context_data = await redis_client.get(eval_context_key)
        
        if not eval_context_data:
            logger.error(f"No evaluation context found in Redis for key: {eval_context_key}")
            return
            
        # Parse the evaluation context
        logger.info(f"Parsing Eval Data")
        eval_context = eval(eval_context_data.decode())  # Using eval since it was stored with str()
        evaluation_inputs = eval_context.get("evaluation_inputs", [])
        llm_outputs = eval_context.get("llm_outputs")

        
        logger.info(f"Retrieved {len(evaluation_inputs)} evaluation inputs from Redis for evaluation")
        
        logger.info(f"Processing {len(evaluation_inputs)} evaluation prompts")
        
        # Generate responses for all prompts
        results = []
        for i, prompt in enumerate(evaluation_inputs):
            logger.info(f"Processing prompt {i+1}/{len(evaluation_inputs)}")
            
            result = await generate_text(prompt, max_tokens=150, temperature=0.7)
            results.append({
                "prompt": prompt,
                "response": result["text"],
                "processing_time": result["processing_time"]
            })
        
        # Send results back to backend
        logger.info(f"POSTING RESULTS TO: {BACKEND_URL}/api/v1/evaluation/{EVAL_BATCH_ID}/results")
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
        
        # Close Redis connection
        await redis_client.close()
        
        logger.info("Batch evaluation completed")
        
    except Exception as e:
        import traceback
        logger.error(f"Error in batch evaluation: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")


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


async def run_batch_evaluation():
    """Run batch evaluation directly."""
    global model, tokenizer
    
    logger.info("Loading model for batch evaluation...")
    success = await load_model()
    if not success:
        logger.error("Failed to load model for batch evaluation")
        return
    
    # Run the actual evaluation
    await process_evaluation_batch()


# Debug: Always log this regardless of context
print(f"PRINT DEBUG: At module level: MODE={MODE}, __name__={__name__}")
logger.info(f"At module level: MODE={MODE}, __name__={__name__}")

if __name__ == "__main__":
    print(f"PRINT DEBUG: Inside __main__ block with MODE={MODE}")
    logger.info(f"Inside __main__ block with MODE={MODE}")
    if MODE == "batch":
        print("PRINT DEBUG: Running in batch mode - starting evaluation directly")
        logger.info("Running in batch mode - starting evaluation directly")
        asyncio.run(run_batch_evaluation())
    else:
        print("PRINT DEBUG: Running in server mode - starting FastAPI")
        logger.info("Running in server mode - starting FastAPI")
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )