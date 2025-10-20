"""Local CPU Inference Setup for Fine-tuned Models

Handles downloading fine-tuned models from cloud storage and setting up
optimized CPU inference with QLora adapters.
"""

import os
import torch
import logging
import json
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalInferenceManager:
    """Manages local CPU inference for fine-tuned models"""
    
    def __init__(self):
        self.models_dir = Path("/app/models")
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.base_model_name = "meta-llama/Llama-3.2-1B"
        
    async def setup_model_for_inference(self, job_id: str, model_path: str) -> Dict[str, Any]:
        """Setup downloaded model for CPU inference"""
        try:
            logger.info(f"Setting up model from {model_path} for CPU inference")
            
            # Verify model files exist
            model_dir = Path(model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            
            # Check for required files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            missing_files = [f for f in required_files if not (model_dir / f).exists()]
            
            if missing_files:
                # Try alternative names
                if (model_dir / "adapter_model.bin").exists():
                    logger.info("Found adapter_model.bin instead of safetensors")
                elif (model_dir / "pytorch_model.bin").exists():
                    logger.info("Found pytorch_model.bin")
                else:
                    logger.warning(f"Missing files: {missing_files}")
            
            # Create inference configuration
            inference_config = {
                "model_path": str(model_path),
                "base_model": self.base_model_name,
                "device": "cpu",
                "dtype": "float32",
                "max_memory": self._get_available_memory(),
                "optimization": "cpu_optimized",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Save inference configuration
            config_path = model_dir / "inference_config.json"
            with open(config_path, "w") as f:
                json.dump(inference_config, f, indent=2)
            
            # Test loading the model
            success = await self._test_model_loading(model_path)
            
            if success:
                logger.info(f"Model {job_id} successfully set up for CPU inference")
                return {
                    "status": "ready",
                    "model_path": str(model_path),
                    "config_path": str(config_path),
                    "optimization": "cpu_optimized"
                }
            else:
                return {
                    "status": "error",
                    "message": "Model loading test failed"
                }
                
        except Exception as e:
            logger.error(f"Failed to setup model for inference: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _test_model_loading(self, model_path: str) -> bool:
        """Test if model can be loaded successfully"""
        try:
            logger.info("Testing model loading...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                token=os.getenv("HF_TOKEN")
            )
            
            # Load base model with CPU optimization
            logger.info("Loading base model for CPU...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                token=os.getenv("HF_TOKEN")
            )
            
            # Load LoRA adapter
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                device_map="cpu"
            )
            
            # Test inference
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Test inference successful: {response[:100]}...")
            
            # Clean up
            del model
            del base_model
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False
    
    async def load_model(self, endpoint_id: str, model_path: str) -> Optional[Any]:
        """Load a model for inference"""
        try:
            if endpoint_id in self.loaded_models:
                logger.info(f"Model {endpoint_id} already loaded")
                return self.loaded_models[endpoint_id]
            
            logger.info(f"Loading model {endpoint_id} from {model_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                token=os.getenv("HF_TOKEN")
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                token=os.getenv("HF_TOKEN")
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                device_map="cpu"
            )
            
            # Set to evaluation mode
            model.eval()
            
            # Create pipeline for easier inference
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device="cpu",
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
            # Cache the loaded model
            self.loaded_models[endpoint_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": pipe,
                "loaded_at": datetime.utcnow()
            }
            
            logger.info(f"Model {endpoint_id} loaded successfully")
            return self.loaded_models[endpoint_id]
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    async def unload_model(self, endpoint_id: str) -> bool:
        """Unload a model from memory"""
        try:
            if endpoint_id in self.loaded_models:
                model_data = self.loaded_models[endpoint_id]
                del model_data["model"]
                del model_data["pipeline"]
                del self.loaded_models[endpoint_id]
                torch.cuda.empty_cache()
                logger.info(f"Model {endpoint_id} unloaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False
    
    async def run_inference(self, endpoint_id: str, prompt: str, 
                           max_tokens: int = 256) -> Optional[str]:
        """Run inference on a loaded model"""
        try:
            if endpoint_id not in self.loaded_models:
                logger.error(f"Model {endpoint_id} not loaded")
                return None
            
            model_data = self.loaded_models[endpoint_id]
            pipe = model_data["pipeline"]
            
            # Run inference
            result = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=model_data["tokenizer"].eos_token_id
            )
            
            # Extract generated text
            generated_text = result[0]["generated_text"]
            
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def _get_available_memory(self) -> Dict[str, Any]:
        """Get available system memory for model loading"""
        import psutil
        
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent_used": mem.percent
        }
    
    async def optimize_for_cpu(self, model_path: str) -> Dict[str, Any]:
        """Apply CPU-specific optimizations to the model"""
        try:
            logger.info(f"Applying CPU optimizations to {model_path}")
            
            # Load model configuration
            config_path = Path(model_path) / "adapter_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Apply optimizations
            optimizations = {
                "inference_mode": True,
                "use_cache": True,
                "torch_dtype": "float32",
                "low_cpu_mem_usage": True,
                "num_threads": os.cpu_count(),
                "optimized_at": datetime.utcnow().isoformat()
            }
            
            # Update configuration
            config["inference_optimizations"] = optimizations
            
            # Save updated configuration
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info("CPU optimizations applied successfully")
            return {
                "status": "optimized",
                "optimizations": optimizations
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a model"""
        try:
            model_dir = Path(model_path)
            
            # Load adapter config
            config_path = model_dir / "adapter_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    adapter_config = json.load(f)
            else:
                adapter_config = {}
            
            # Load training metrics if available
            metrics_path = model_dir / "training_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    training_metrics = json.load(f)
            else:
                training_metrics = {}
            
            # Load carbon emissions if available
            emissions_path = model_dir / "carbon_emissions.json"
            if emissions_path.exists():
                with open(emissions_path, "r") as f:
                    carbon_emissions = json.load(f)
            else:
                carbon_emissions = {}
            
            # Calculate model size
            model_size_mb = sum(
                f.stat().st_size for f in model_dir.glob("**/*") if f.is_file()
            ) / (1024 * 1024)
            
            return {
                "model_path": str(model_path),
                "base_model": adapter_config.get("base_model_name_or_path", self.base_model_name),
                "lora_config": {
                    "r": adapter_config.get("r"),
                    "alpha": adapter_config.get("lora_alpha"),
                    "dropout": adapter_config.get("lora_dropout"),
                    "target_modules": adapter_config.get("target_modules")
                },
                "training_metrics": training_metrics,
                "carbon_emissions": carbon_emissions,
                "model_size_mb": round(model_size_mb, 2),
                "files": [f.name for f in model_dir.glob("*") if f.is_file()]
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "error": str(e)
            }


# Global inference manager instance
local_inference_manager = LocalInferenceManager()