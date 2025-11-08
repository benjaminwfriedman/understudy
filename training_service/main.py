#!/usr/bin/env python3
"""
Training Service

Dedicated service for managing training jobs on external GPU providers (RunPod).
Handles the training workflow:
1. Receive training requests from Backend
2. Queue and execute training on RunPod 
3. Monitor training progress
4. Download trained model weights
5. Store in Model Broker
6. Notify Backend of completion
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update
from contextlib import asynccontextmanager
import redis.asyncio as redis
from pydantic import BaseModel
from runpod_trainer import get_runpod_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis-service:6379")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend-service:8000")
MODEL_BROKER_URL = os.getenv("MODEL_BROKER_URL", "http://model-broker-service:8003")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
NAMESPACE = os.getenv("NAMESPACE", "understudy")

# Database setup
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Redis connection for job queue
redis_client = None


class TrainingRequest(BaseModel):
    train_id: str
    endpoint_id: str
    version: int
    training_pairs_count: int
    slm_type: str
    source_llm: str
    provider: str = "runpod"  # Default to runpod, but can be azure, lambda, etc.
    training_data: Optional[List[Dict[str, str]]] = None  # Actual training data


class TrainingStatus(BaseModel):
    train_id: str
    status: str
    phase: str
    progress: Optional[float] = None
    message: Optional[str] = None


async def get_db():
    """Database dependency."""
    async with AsyncSessionLocal() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global redis_client
    
    # Startup
    logger.info("Starting Training Service...")
    
    # Initialize Redis connection
    redis_client = redis.from_url(REDIS_URL)
    await redis_client.ping()
    logger.info("Connected to Redis")
    
    # Training service is now request-driven, no background workers needed
    
    yield
    
    # Shutdown
    logger.info("Shutting down Training Service...")
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="Understudy Training Service",
    description="Manages training jobs on external GPU providers",
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


# Global trainer instance
runpod_trainer_instance = None

def get_trainer():
    """Get RunPod trainer instance"""
    global runpod_trainer_instance
    if runpod_trainer_instance is None:
        runpod_trainer_instance = get_runpod_trainer()
    return runpod_trainer_instance


async def generate_training_script(request: TrainingRequest) -> str:
    """Generate the training script for the specific model"""
    # Get training configuration with defaults
    config = {
        'epochs': 3,
        'batch_size': 4, 
        'learning_rate': 2e-4,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
    }
    
    # Get HF token for model access
    hf_token = os.getenv("HF_TOKEN", "")
    
    script = f'''#!/usr/bin/env python3
# Training script for job: {request.train_id}
# Endpoint: {request.endpoint_id}
# Generated at: {datetime.now().isoformat()}

import json
import torch
import os
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import codecarbon
import accelerate

print("=" * 50)
print(f"Starting training job: {request.train_id}")
print("Model: meta-llama/Llama-3.2-1B")
print(f"Device: {{torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}}")
print("=" * 50)

try:
    with open("training_data.json", "r") as f:
        raw_training_data = json.load(f)
except Exception as e:
    raise ValueError(f"Failed to load training data: {{e}}")

if isinstance(raw_training_data, str):
    pairs = raw_training_data.strip().split("\\n\\n")
    training_data = []
    for pair in pairs:
        if pair.strip():
            training_data.append({{"text": pair.strip()}})
else:
    training_data = raw_training_data

print(f"Loaded {{len(training_data)}} training examples")

if len(training_data) == 0:
    raise ValueError("No training data found")
if len(training_data) < 5:
    raise ValueError(f"Insufficient training data: {{len(training_data)}} examples (minimum 5 required)")

tracker = codecarbon.EmissionsTracker(
    project_name="understudy-{request.endpoint_id}",
    measure_power_secs=15
)

tracker.start()

try:
    hf_token = "{hf_token}"
    if not hf_token or hf_token == "None":
        raise ValueError("HuggingFace token is required but not provided")
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer: {{e}}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
    except Exception as e:
        raise ValueError(f"Failed to load model: {{e}}")

    print(f"Model loaded: {{model}}")
    model.gradient_checkpointing_enable()

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r={config.get('lora_r', 16)},
        lora_alpha={config.get('lora_alpha', 32)},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout={config.get('lora_dropout', 0.1)},
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train()
    
    lora_param_count = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            lora_param_count += param.numel()
            print(f"LoRA parameter: {{name}}, requires_grad: {{param.requires_grad}}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {{trainable_params}}")
    print(f"LoRA parameters: {{lora_param_count}}")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found!")
    
    print("Preparing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
    
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train samples: {{len(train_dataset)}}, Eval samples: {{len(eval_dataset)}}")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs={config.get('epochs', 3)},
        per_device_train_batch_size={config.get('batch_size', 4)},
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        learning_rate={config.get('learning_rate', 2e-4)},
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        ddp_find_unused_parameters=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("Starting training...")
    print("-" * 50)
    try:
        train_result = trainer.train()
    except RuntimeError as e:
        if "does not require grad" in str(e):
            raise ValueError(f"Gradient computation error: {{e}}")
        raise ValueError(f"Training failed: {{e}}")
    except Exception as e:
        raise ValueError(f"Training failed: {{e}}")
    
    print("-" * 50)
    print("Training completed!")
    print(f"Final training loss: {{train_result.training_loss:.4f}}")
    
    print("Saving model...")
    os.makedirs("./final_model", exist_ok=True)
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    metrics = {{
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "epoch": train_result.metrics["epoch"]
    }}
    
    with open("./final_model/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Model saved successfully!")
    
finally:
    emissions = tracker.stop()
    print(f"Carbon emissions: {{emissions:.6f}} kg CO2")
    
    os.makedirs("./final_model", exist_ok=True)
    with open("./final_model/carbon_emissions.json", "w") as f:
        json.dump({{"emissions_kg": emissions, "timestamp": datetime.utcnow().isoformat()}}, f)

print("=" * 50)
print("Training job completed successfully!")
'''
    return script


async def store_model_in_broker(train_id: str, endpoint_id: str, version: int, model_directory: str) -> bool:
    """Store the downloaded model directory in the Model Broker."""
    try:
        model_broker_url = os.getenv("MODEL_BROKER_URL", "http://model-broker-service:8003")
        model_dir_path = Path(model_directory)
        
        if not model_dir_path.exists():
            logger.error(f"Model directory not found: {model_directory}")
            return False
        
        # Find the final_model subdirectory
        final_model_path = model_dir_path / "final_model"
        if not final_model_path.exists():
            logger.error(f"final_model subdirectory not found in {model_directory}")
            return False
        
        # Prepare files for upload
        files = []
        file_handles = []
        
        try:
            for file_path in final_model_path.iterdir():
                if file_path.is_file():
                    file_handle = open(file_path, 'rb')
                    file_handles.append(file_handle)
                    files.append(('files', (file_path.name, file_handle, 'application/octet-stream')))
            
            if not files:
                logger.error(f"No files found in {final_model_path}")
                return False
            
            # Upload to model broker
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for large models
                logger.info(f"Uploading {len(files)} files to model broker for {endpoint_id}/v{version}")
                
                response = await client.put(
                    f"{model_broker_url}/store-model-directory/{endpoint_id}/{version}",
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully stored model in broker: {result['total_size']} bytes total")
                    return True
                else:
                    logger.error(f"Model broker upload failed: {response.status_code} - {response.text}")
                    return False
                    
        finally:
            # Close all file handles
            for handle in file_handles:
                handle.close()
                
    except Exception as e:
        logger.error(f"Error storing model in broker: {e}")
        return False


async def execute_training_job(train_id: str, endpoint_id: str, version: int, pod_instance, training_script: str, training_data: Optional[List[Dict[str, str]]] = None):
    """Execute training job in background"""
    try:
        trainer = get_trainer()
        
        # Update status to training
        await redis_client.hset(f"training:{train_id}", "status", "training")
        
        # Execute training
        success = await trainer.execute_training(pod_instance, training_script, train_id, training_data)
        
        if success:
            # Update status to downloading
            await redis_client.hset(f"training:{train_id}", "status", "downloading")
            
            # Download model weights
            model_path = f"/app/model_weights/{train_id}"
            os.makedirs(model_path, exist_ok=True)
            
            download_success = await trainer.download_model_weights(pod_instance, train_id, model_path)
            
            if download_success:
                # Calculate training wall time
                training_info = await redis_client.hgetall(f"training:{train_id}")
                # Redis returns bytes, so we need to decode or use byte keys
                training_start_str = training_info.get(b"started_at")
                if training_start_str:
                    training_start_str = training_start_str.decode('utf-8')
                logger.info(f"Retrieved training info: {training_info}")
                logger.info(f"Training start string: {training_start_str}")
                training_time_wall = None
                if training_start_str:
                    try:
                        training_start = datetime.fromisoformat(training_start_str)
                        training_end = datetime.now()
                        training_time_wall = (training_end - training_start).total_seconds()
                        logger.info(f"Training wall time: {training_time_wall:.2f} seconds")
                    except Exception as e:
                        logger.warning(f"Could not calculate training wall time: {e}")
                else:
                    logger.warning(f"No started_at timestamp found for training {train_id}")
                
                # Read training metrics from the downloaded model
                metrics_path = Path(model_path) / "final_model" / "training_metrics.json"
                training_metrics = {}
                if metrics_path.exists():
                    try:
                        with open(metrics_path, 'r') as f:
                            training_metrics = json.load(f)
                        logger.info(f"Loaded training metrics: {training_metrics}")
                    except Exception as e:
                        logger.warning(f"Could not load training metrics: {e}")
                
                # Read carbon emissions from the downloaded model
                carbon_path = Path(model_path) / "final_model" / "carbon_emissions.json"
                carbon_emissions_kg = None
                energy_consumed_kwh = None
                if carbon_path.exists():
                    try:
                        with open(carbon_path, 'r') as f:
                            carbon_data = json.load(f)
                        carbon_emissions_kg = carbon_data.get("emissions_kg")
                        logger.info(f"Loaded carbon emissions: {carbon_emissions_kg} kg CO2")
                    except Exception as e:
                        logger.warning(f"Could not load carbon emissions: {e}")
                
                # Store model in broker for persistent storage
                logger.info(f"Storing model in broker for {endpoint_id}/v{version}")
                broker_success = await store_model_in_broker(train_id, endpoint_id, version, model_path)
                
                if broker_success:
                    logger.info(f"Model successfully stored in broker")
                    model_weights_path = f"{endpoint_id}/v{version}"
                else:
                    logger.warning(f"Failed to store model in broker, but model download was successful")
                    model_weights_path = None
                
                # Prepare completion data with all metrics (ensure JSON serializable)
                completion_data = {
                    "phase": "available",
                    "training_loss": float(training_metrics.get("train_loss")) if training_metrics.get("train_loss") is not None else None,
                    "model_weights_path": model_weights_path,
                    "training_time_wall": float(training_time_wall) if training_time_wall is not None else None,
                    "carbon_emissions_kg": float(carbon_emissions_kg) if carbon_emissions_kg is not None else None,
                    "energy_consumed_kwh": float(energy_consumed_kwh) if energy_consumed_kwh is not None else None
                }
                
                # Log the completion data for debugging
                logger.info(f"Sending completion data: {completion_data}")
                
                # Notify backend of completion with metrics
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{BACKEND_URL}/api/v1/lifecycle/training/{train_id}/complete",
                        json=completion_data
                    )
                
                await redis_client.hset(f"training:{train_id}", "status", "completed")
                logger.info(f"Training job {train_id} completed successfully")
            else:
                await redis_client.hset(f"training:{train_id}", "status", "failed")
                logger.error(f"Failed to download model for {train_id}")
        else:
            await redis_client.hset(f"training:{train_id}", "status", "failed")
            logger.error(f"Training failed for {train_id}")
        
        # Cleanup pod
        await trainer.terminate_pod(pod_instance.id)
        
    except Exception as e:
        logger.error(f"Error in training job {train_id}: {e}")
        await redis_client.hset(f"training:{train_id}", "status", "failed")
        
        # Cleanup pod on error
        try:
            trainer = get_trainer()
            await trainer.terminate_pod(pod_instance.id)
        except:
            pass


# Legacy queue worker removed - now using direct API requests


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Understudy Training Service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start a new training job"""
    logger.info(f"Received training request for {request.train_id}")
    
    # Record training start time
    training_start_time = datetime.now()
    
    try:
        # TODO: Future provider support
        # Currently only RunPod is implemented. For other providers:
        # - Azure: Implement Azure trainer class with VM management
        # - Lambda: Implement Lambda Cloud trainer class  
        # - AWS: Implement AWS SageMaker or EC2 trainer
        # All provider logic should be implemented in this training service
        
        if request.provider != "runpod":
            raise Exception(f"Provider '{request.provider}' not yet implemented. Currently only 'runpod' is supported.")
        
        # Get trainer instance and start training on RunPod
        trainer = get_trainer()
        
        # Generate training script
        training_script = await generate_training_script(request)
        
        # Launch RunPod instance
        pod_name = f"understudy-{request.train_id[:8]}"
        pod_instance = await trainer.launch_pod(request.train_id, pod_name)
        
        if not pod_instance:
            raise Exception("Failed to launch RunPod instance")
        
        # Store job info in Redis
        await redis_client.hset(
            f"training:{request.train_id}",
            mapping={
                "pod_id": pod_instance.id,
                "pod_name": pod_instance.name,
                "ip_address": pod_instance.ip_address,
                "ssh_port": str(pod_instance.ssh_port),
                "status": "pod_launched",
                "started_at": datetime.now().isoformat(),
                "endpoint_id": request.endpoint_id,
                "version": str(request.version),
                "provider": request.provider
            }
        )
        
        # Start training in background
        background_tasks.add_task(execute_training_job, request.train_id, request.endpoint_id, request.version, pod_instance, training_script, request.training_data)
        
        result = {
            "runpod_job_id": pod_instance.id,
            "status": "started"
        }
        
        # Add to processing queue
        import json
        await redis_client.lpush(
            "training_queue",
            json.dumps({
                "train_id": request.train_id,
                "endpoint_id": request.endpoint_id,
                "version": request.version,
                "started_at": datetime.now().isoformat()
            })
        )
        
        return {
            "message": "Training started successfully",
            "train_id": request.train_id,
            "runpod_job_id": result["runpod_job_id"],
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error starting training for {request.train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@app.get("/api/v1/training/{train_id}/status")
async def get_training_status(train_id: str):
    """Get status of a training job"""
    try:
        # Get status from Redis
        job_data = await redis_client.hgetall(f"training:{train_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        return TrainingStatus(
            train_id=train_id,
            status=job_data.get("status", "unknown"),
            phase=job_data.get("status", "unknown"),
            progress=None,  # Can add progress tracking later
            message=f"Pod: {job_data.get('pod_name', 'unknown')}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status for {train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@app.post("/api/v1/training/{train_id}/cancel")
async def cancel_training(train_id: str):
    """Cancel a training job"""
    try:
        # Get job data first
        job_data = await redis_client.hgetall(f"training:{train_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Terminate RunPod instance if it exists
        if job_data.get("pod_id"):
            trainer = get_trainer()
            await trainer.terminate_pod(job_data["pod_id"])
        
        # Remove from queue
        pending_jobs = await redis_client.lrange("training_queue", 0, -1)
        for queue_job_data in pending_jobs:
            import json
            job = json.loads(queue_job_data)
            if job["train_id"] == train_id:
                await redis_client.lrem("training_queue", 1, queue_job_data)
                break
        
        # Update status to cancelled
        await redis_client.hset(f"training:{train_id}", "status", "cancelled")
        
        return {"message": f"Training {train_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training {train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel training: {str(e)}")


@app.post("/api/v1/training/{train_id}/completed")
async def mark_training_completed(train_id: str, completion_data: dict):
    """Mark a training job as completed (called by backend)"""
    try:
        # Update the training status
        await redis_client.hset(
            f"training:{train_id}",
            mapping={
                "status": completion_data.get("status", "completed"),
                "completed_at": datetime.now().isoformat(),
                "model_path": completion_data.get("model_path", "")
            }
        )
        
        logger.info(f"Training {train_id} marked as {completion_data.get('status', 'completed')}")
        return {"message": "Training status updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating training status for {train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update training status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )