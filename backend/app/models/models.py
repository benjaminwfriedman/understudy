from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Endpoint(Base):
    __tablename__ = "endpoints"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    llm_provider = Column(String, nullable=False)  # 'openai', 'anthropic', etc.
    llm_model = Column(String, nullable=False)      # 'gpt-4', 'claude-3-sonnet', etc.
    slm_model_path = Column(String)                 # Path to trained SLM
    status = Column(String, nullable=False, default="training")  # 'training', 'ready', 'active'
    langchain_compatible = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    inference_logs = relationship("InferenceLog", back_populates="endpoint", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRun", back_populates="endpoint", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="endpoint", cascade="all, delete-orphan")
    config = relationship("EndpointConfig", back_populates="endpoint", uselist=False, cascade="all, delete-orphan")


class InferenceLog(Base):
    __tablename__ = "inference_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    endpoint_id = Column(String, ForeignKey("endpoints.id"), nullable=False)
    input_text = Column(Text, nullable=False)
    llm_output = Column(Text)
    slm_output = Column(Text)
    model_used = Column(String, nullable=False)  # 'llm' or 'slm'
    latency_ms = Column(Integer)
    cost_usd = Column(Float)
    langchain_metadata = Column(JSON)  # Store chain info, run_id, etc.
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    endpoint = relationship("Endpoint", back_populates="inference_logs")
    carbon_emissions = relationship("CarbonEmission", back_populates="inference_log", cascade="all, delete-orphan")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (
        # Composite unique constraint on (endpoint_id, version) to ensure proper versioning per endpoint
        UniqueConstraint('endpoint_id', 'version', name='uq_training_runs_endpoint_version'),
    )
    
    # Primary key - unique identifier for the training run
    train_id = Column(String, primary_key=True, default=generate_uuid)
    
    # Basic training info
    endpoint_id = Column(String, ForeignKey("endpoints.id"), nullable=False)
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Model configuration
    slm_type = Column(String, nullable=False)  # e.g., "Llama 3.2 1B"
    source_llm = Column(String, nullable=False)  # e.g., "gpt-3.5-turbo"
    training_pairs_count = Column(Integer, nullable=False)
    
    # Training results
    training_loss = Column(Float)  # Final fine-tuning loss value
    
    # Evaluation results
    semantic_similarity_score = Column(Float)  # Similarity score to source LLM
    
    # Lifecycle management
    phase = Column(String, nullable=False, default="training")  # training, downloading, llm_evaluation, available, deploying, deployed, failed
    
    # Storage and deployment
    model_weights_path = Column(String)  # Path in model broker
    k8s_deployment_name = Column(String)  # Name of K8s deployment if deployed
    inference_mode = Column(String)  # 'endpoint' or 'batch' or null
    
    # Soft delete
    is_deleted = Column(Boolean, default=False, nullable=False)
    
    # Carbon tracking (keeping these as requested)
    carbon_emissions_kg = Column(Float)  # From CodeCarbon
    energy_consumed_kwh = Column(Float)  # From CodeCarbon
    
    # Legacy/compatibility fields (optional - can be removed after migration)
    start_time = Column(DateTime)  # Maps to created_at
    end_time = Column(DateTime)
    examples_used = Column(Integer)  # Maps to training_pairs_count
    final_loss = Column(Float)  # Maps to training_loss
    error_message = Column(Text)
    
    # For legacy compatibility - keeping the id field as well initially
    id = Column(String, default=generate_uuid)  # Can be removed after migration
    
    # Relationships
    endpoint = relationship("Endpoint", back_populates="training_runs")
    carbon_emissions = relationship("CarbonEmission", back_populates="training_run", cascade="all, delete-orphan")


class Metric(Base):
    __tablename__ = "metrics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    endpoint_id = Column(String, ForeignKey("endpoints.id"), nullable=False)
    metric_type = Column(String, nullable=False)  # 'semantic_similarity', 'rouge', 'bleu', etc.
    value = Column(Float, nullable=False)
    calculated_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    endpoint = relationship("Endpoint", back_populates="metrics")


class CarbonEmission(Base):
    __tablename__ = "carbon_emissions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    training_run_id = Column(String, ForeignKey("training_runs.id"))
    inference_log_id = Column(String, ForeignKey("inference_logs.id"))
    timestamp = Column(DateTime, nullable=False, server_default=func.now())
    duration_seconds = Column(Float)
    emissions_kg = Column(Float)
    energy_consumed_kwh = Column(Float)
    cpu_power_w = Column(Float)
    gpu_power_w = Column(Float)
    country_iso_code = Column(String)
    
    # Relationships
    training_run = relationship("TrainingRun", back_populates="carbon_emissions")
    inference_log = relationship("InferenceLog", back_populates="carbon_emissions")


class EndpointConfig(Base):
    __tablename__ = "endpoint_configs"
    
    endpoint_id = Column(String, ForeignKey("endpoints.id"), primary_key=True)
    training_batch_size = Column(Integer, default=100)
    similarity_threshold = Column(Float, default=0.95)
    auto_switchover = Column(Boolean, default=False)
    lora_r = Column(Integer, default=8)
    lora_alpha = Column(Integer, default=16)
    learning_rate = Column(Float, default=0.0003)
    track_carbon = Column(Boolean, default=True)
    max_training_examples = Column(Integer, default=1000)
    training_frequency_hours = Column(Integer, default=24)
    
    # Relationships
    endpoint = relationship("Endpoint", back_populates="config")