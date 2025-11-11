from pydantic_settings import BaseSettings
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))


class Settings(BaseSettings):
    APP_NAME: str = "Understudy"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./understudy.db"
    DATABASE_ECHO: bool = False
    
    # Security
    SECRET_KEY: str = os.urandom(32).hex()
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    ALGORITHM: str = "HS256"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Azure Functions - Prompt Compression
    AZURE_FUNCTION_URL: str = "https://my-prompt-compressor-func.azurewebsites.net/api/compress"
    AZURE_FUNCTION_KEY: Optional[str] = None  # Azure Function authentication key
    COMPRESSION_TIMEOUT_SECONDS: int = 30  # Timeout for compression requests
    
    # Model Pricing (per 1000 tokens)
    MODEL_PRICING: Dict[str, float] = {
        # OpenAI Models
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.00015,
        "gpt-3.5-turbo": 0.0015,
        
        # Anthropic Models  
        "claude-3-opus": 0.015,
        "claude-3-sonnet": 0.003,
        "claude-3-haiku": 0.00025,
        "claude-3-5-sonnet": 0.003,
        "claude-3-5-haiku": 0.001,
        
        # Understudy SLM Inference
        "understudy-slm": 0.0001,  # Very low cost for local SLM inference
        
        # Default fallback
        "default": 0.002
    }
    
    # Model Paths
    BASE_MODEL_PATH: str = "meta-llama/Llama-3.2-1B"
    MODELS_DIR: str = "./models"
    CHECKPOINTS_DIR: str = "./checkpoints"
    
    # Training
    DEFAULT_TRAINING_SIZES: List = [10, 100, 1000]
    DEFAULT_LEARNING_RATE: float = 3e-4
    DEFAULT_NUM_EPOCHS: int = 3
    DEFAULT_LORA_R: int = 8
    DEFAULT_LORA_ALPHA: int = 16
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.95
    
    # Carbon Tracking
    CARBON_TRACKING_ENABLED: bool = True
    CARBON_DATA_DIR: str = "./carbon_data"
    COUNTRY_ISO_CODE: str = "USA"
    
    # Cache
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour
    
    # Monitoring
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = os.getenv("LOG_FILE", "/tmp/understudy.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()