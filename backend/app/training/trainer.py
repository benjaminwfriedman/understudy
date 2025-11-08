import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy import select
from app.models import (
    Endpoint, InferenceLog, TrainingRun, 
    EndpointConfig, CarbonEmission
)
from app.models.database import AsyncSessionLocal
from app.training.carbon_tracker import CarbonTracker
from app.core.config import settings
import logging
import os
import asyncio
import json

logger = logging.getLogger(__name__)


class UnderstudyTrainer:
    """Trainer for fine-tuning Small Language Models with LoRA."""
    
    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id
        self.base_model_name = settings.BASE_MODEL_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.config = None
        self.use_azure_gpu = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
        self.use_lambda_gpu = os.getenv("LAMBDA_TRAINING_ENABLED", "false").lower() == "true"
        self.gpu_provider = os.getenv("GPU_TRAINING_PROVIDER", "lambda" if self.use_lambda_gpu else "azure")
        
    async def load_config(self):
        """Load endpoint configuration from database."""
        async with AsyncSessionLocal() as session:
            self.config = await session.get(EndpointConfig, self.endpoint_id)
            if not self.config:
                # Create default config
                self.config = EndpointConfig(endpoint_id=self.endpoint_id)
                session.add(self.config)
                await session.commit()
    
    async def prepare_dataset(self, num_examples: Optional[int] = None) -> Dataset:
        """Fetch training data from inference logs."""
        async with AsyncSessionLocal() as session:
            # Get training examples from inference logs
            query = select(InferenceLog).where(
                InferenceLog.endpoint_id == self.endpoint_id,
                InferenceLog.model_used == "llm",
                InferenceLog.llm_output.isnot(None)
            ).order_by(InferenceLog.created_at.desc())
            
            if num_examples:
                query = query.limit(num_examples)
            else:
                query = query.limit(self.config.max_training_examples)
            
            result = await session.execute(query)
            logs = result.scalars().all()
            
            if len(logs) < 10:
                raise ValueError(f"Insufficient training data. Found {len(logs)} examples, need at least 10.")
            
            # Format as instruction-following dataset
            formatted_data = []
            for log in logs:
                # Format with special tokens for instruction tuning
                text = f"<|user|>\n{log.input_text}\n<|assistant|>\n{log.llm_output}\n<|end|>"
                formatted_data.append({"text": text})
            
            logger.info(f"Prepared dataset with {len(formatted_data)} examples")
            return Dataset.from_list(formatted_data)
    
    async def train(self) -> Dict[str, Any]:
        """Train the model with carbon tracking."""
        await self.load_config()
        
        # Check if cloud GPU training is enabled
        if self.use_azure_gpu or self.use_lambda_gpu:
            return await self._train_on_cloud_gpu()
        
        # Create training run record
        async with AsyncSessionLocal() as session:
            training_run = TrainingRun(
                endpoint_id=self.endpoint_id,
                start_time=datetime.utcnow(),
                status="running"
            )
            session.add(training_run)
            await session.commit()
            training_run_id = training_run.id
        
        # Initialize carbon tracker
        carbon_tracker = CarbonTracker(
            project_name=f"understudy_{self.endpoint_id}",
            task_type="training"
        )
        
        try:
            # Start carbon tracking
            carbon_tracker.start()
            
            # Prepare dataset
            dataset = await self.prepare_dataset(self.config.training_batch_size)
            
            # Load base model
            logger.info(f"Loading base model {self.base_model_name}")
            
            # Use 8-bit loading to reduce memory usage
            from transformers import BitsAndBytesConfig
            
            # Configure for low memory usage
            if self.device == "cpu":
                # CPU: use float32 with low memory mode
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN")
                )
            else:
                # GPU: can use 8-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN")
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                token=os.getenv("HF_TOKEN")
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Enable training mode
            model.train()
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors=None
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Set up output directory
            output_dir = os.path.join(settings.CHECKPOINTS_DIR, self.endpoint_id, training_run_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4 if self.device == "cuda" else 2,  # Increased for CPU
                gradient_accumulation_steps=2,  # Reduced since we increased batch size
                warmup_steps=100,
                learning_rate=self.config.learning_rate,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="no",  # Changed from evaluation_strategy
                save_total_limit=2,
                load_best_model_at_end=False,
                report_to="none",
                fp16=self.device == "cuda",
                gradient_checkpointing=False,  # Disabled - conflicts with LoRA
                optim="adamw_torch",
                remove_unused_columns=False,
                dataloader_num_workers=4,  # Use all 4 CPU cores
                dataloader_pin_memory=False  # Disable for CPU
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Train!
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            final_model_path = os.path.join(settings.MODELS_DIR, self.endpoint_id, training_run_id)
            os.makedirs(final_model_path, exist_ok=True)
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Stop carbon tracking
            emissions_data = carbon_tracker.stop()
            
            # Save carbon data to database
            await carbon_tracker.save_to_db(training_run_id=training_run_id)
            
            # Update training run with results
            async with AsyncSessionLocal() as session:
                training_run = await session.get(TrainingRun, training_run_id)
                training_run.end_time = datetime.utcnow()
                training_run.examples_used = len(dataset)
                training_run.final_loss = train_result.training_loss
                training_run.carbon_emissions_kg = emissions_data["emissions_kg"]
                training_run.energy_consumed_kwh = emissions_data["energy_consumed_kwh"]
                training_run.phase = "completed"
                
                # Update endpoint with new model path
                endpoint = await session.get(Endpoint, self.endpoint_id)
                endpoint.slm_model_path = final_model_path
                endpoint.status = "ready"
                endpoint.updated_at = datetime.utcnow()
                
                await session.commit()
            
            logger.info(f"Training completed successfully. Model saved to {final_model_path}")
            
            return {
                "training_run_id": training_run_id,
                "model_path": final_model_path,
                "final_loss": train_result.training_loss,
                "examples_used": len(dataset),
                "emissions": emissions_data
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Stop tracker even on failure
            if carbon_tracker:
                emissions_data = carbon_tracker.stop()
                await carbon_tracker.save_to_db(training_run_id=training_run_id)
            
            # Update training run status
            async with AsyncSessionLocal() as session:
                training_run = await session.get(TrainingRun, training_run_id)
                if training_run:
                    training_run.end_time = datetime.utcnow()
                    training_run.phase = "failed"
                    training_run.error_message = str(e)
                    if carbon_tracker and emissions_data:
                        training_run.carbon_emissions_kg = emissions_data.get("emissions_kg", 0)
                        training_run.energy_consumed_kwh = emissions_data.get("energy_consumed_kwh", 0)
                    await session.commit()
            
            raise
    
    async def _train_on_cloud_gpu(self) -> Dict[str, Any]:
        """Train model on cloud GPU infrastructure (Azure or Lambda)."""
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        # Load config if not already loaded
        if not self.config:
            await self.load_config()
        
        # Prepare dataset
        dataset = await self.prepare_dataset()
        
        # Convert dataset to format for cloud training
        training_data = []
        for item in dataset:
            training_data.append({
                "text": item["text"],
                "input": item.get("input", ""),
                "output": item.get("output", "")
            })
        
        # Training configuration
        training_config = {
            "base_model": self.base_model_name,
            "epochs": 3,  # Default epochs
            "batch_size": self.config.training_batch_size,
            "learning_rate": self.config.learning_rate,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": 0.1,  # Default dropout
            "training_data": training_data
        }
        
        # Initialize queue manager if needed
        if not gpu_queue_manager.redis_client:
            await gpu_queue_manager.initialize()
        
        # Determine provider
        provider = self.gpu_provider.lower()
        if provider not in ["azure", "lambda"]:
            provider = "lambda"  # Default to Lambda Cloud
        
        # Add job to queue with specified provider
        job_id = await gpu_queue_manager.add_job(
            endpoint_id=self.endpoint_id,
            training_config=training_config,
            priority=0,  # Default priority
            provider=provider
        )
        
        logger.info(f"Training job {job_id} queued for {provider.upper()} GPU processing")
        
        # Create training run record
        async with AsyncSessionLocal() as session:
            training_run = TrainingRun(
                endpoint_id=self.endpoint_id,
                start_time=datetime.utcnow(),
                status="queued"
            )
            session.add(training_run)
            await session.commit()
            training_run_id = training_run.id
        
        return {
            "training_run_id": training_run_id,
            "job_id": job_id,
            "provider": provider,
            "status": "queued",
            "message": f"Training job queued for {provider.upper()} GPU processing"
        }


class TrainingScheduler:
    """Schedules and manages training jobs."""
    
    def __init__(self):
        self.active_jobs = {}
        self.use_azure_gpu = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
        self.use_lambda_gpu = os.getenv("LAMBDA_TRAINING_ENABLED", "false").lower() == "true"
        self.gpu_provider = os.getenv("GPU_TRAINING_PROVIDER", "lambda" if self.use_lambda_gpu else "azure")
    
    async def schedule_training(self, endpoint_id: str) -> str:
        """Schedule a training job for an endpoint."""
        if endpoint_id in self.active_jobs:
            return "Training already in progress"
        
        # Create trainer
        trainer = UnderstudyTrainer(endpoint_id)
        
        if self.use_azure_gpu or self.use_lambda_gpu:
            # For cloud GPU, directly call the training method
            result = await trainer._train_on_cloud_gpu()
            return f"Training job {result['job_id']} queued for {result['provider'].upper()} GPU"
        else:
            # Run training in background for local training
            task = asyncio.create_task(trainer.train())
            self.active_jobs[endpoint_id] = task
            
            # Clean up when done
            task.add_done_callback(lambda t: self.active_jobs.pop(endpoint_id, None))
            
            return "Training job scheduled"
    
    def get_active_jobs(self) -> List[str]:
        """Get list of endpoints with active training jobs."""
        return list(self.active_jobs.keys())
    
    async def wait_for_job(self, endpoint_id: str):
        """Wait for a specific training job to complete."""
        if endpoint_id in self.active_jobs:
            return await self.active_jobs[endpoint_id]
        return None