import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, Dict, Any, List
from langchain.schema import BaseMessage
from codecarbon import OfflineEmissionsTracker
from app.core.config import settings
from app.models import CarbonEmission
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class SLMInferenceEngine:
    """Inference engine for Small Language Models with LoRA adapters."""
    
    def __init__(self):
        self.loaded_models = {}  # Cache for loaded models
        self.track_inference_carbon = settings.CARBON_TRACKING_ENABLED
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"SLM Inference Engine initialized on {self.device}")
    
    async def load_model(self, model_path: str) -> tuple:
        """Load or retrieve cached model."""
        if model_path not in self.loaded_models:
            try:
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    settings.BASE_MODEL_PATH,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Load LoRA adapter
                model = PeftModel.from_pretrained(base_model, model_path)
                model.eval()
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_PATH)
                tokenizer.pad_token = tokenizer.eos_token
                
                self.loaded_models[model_path] = (model, tokenizer)
                logger.info(f"Loaded SLM model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                raise
        
        return self.loaded_models[model_path]
    
    async def generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with optional carbon tracking."""
        
        # Track carbon for inference
        tracker = None
        if self.track_inference_carbon:
            tracker = OfflineEmissionsTracker(
                country_iso_code=settings.COUNTRY_ISO_CODE,
                log_level="warning"
            )
            tracker.start()
        
        try:
            # Load model
            model, tokenizer = await self.load_model(model_path)
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new tokens (remove prompt)
            response = generated_text[len(prompt):].strip()
            
            result = {
                "output": response,
                "model_used": "slm",
                "model_path": model_path
            }
            
            # Stop carbon tracking and get emissions
            if tracker:
                emissions_data = tracker.stop()
                result["carbon_emissions"] = {
                    "emissions_kg": emissions_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return result
            
        except Exception as e:
            if tracker:
                tracker.stop()
            logger.error(f"SLM generation error: {e}")
            raise
    
    async def generate_with_messages(
        self,
        model_path: str,
        messages: List[BaseMessage],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate from LangChain messages format."""
        prompt = self._messages_to_prompt(messages)
        return await self.generate(model_path, prompt, **kwargs)
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to prompt format."""
        prompt_parts = []
        for msg in messages:
            if msg.type == "system":
                prompt_parts.append(f"<|system|>\n{msg.content}")
            elif msg.type == "human":
                prompt_parts.append(f"<|user|>\n{msg.content}")
            elif msg.type == "ai":
                prompt_parts.append(f"<|assistant|>\n{msg.content}")
        
        prompt_parts.append("<|assistant|>")
        return "\n".join(prompt_parts)
    
    async def evaluate_similarity(
        self,
        model_path: str,
        llm_output: str,
        slm_output: str
    ) -> float:
        """Evaluate semantic similarity between LLM and SLM outputs."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Load sentence transformer model
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode outputs
            llm_embedding = encoder.encode([llm_output])
            slm_embedding = encoder.encode([slm_output])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(llm_embedding, slm_embedding)[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to evaluate similarity: {e}")
            return 0.0
    
    def unload_model(self, model_path: str):
        """Unload a model from memory."""
        if model_path in self.loaded_models:
            del self.loaded_models[model_path]
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model {model_path}")