from pydantic_settings import BaseSettings
from typing import Optional, List
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
    
    # Model Paths
    BASE_MODEL_PATH: str = "meta-llama/Llama-3.2-1B"
    MODELS_DIR: str = "./models"
    CHECKPOINTS_DIR: str = "./checkpoints"
    
    # Training
    DEFAULT_BATCH_SIZE: int = 100
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