from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON
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
    
    id = Column(String, primary_key=True, default=generate_uuid)
    endpoint_id = Column(String, ForeignKey("endpoints.id"), nullable=False)
    start_time = Column(DateTime, nullable=False, server_default=func.now())
    end_time = Column(DateTime)
    examples_used = Column(Integer)
    final_loss = Column(Float)
    status = Column(String, nullable=False, default="running")  # 'running', 'completed', 'failed'
    carbon_emissions_kg = Column(Float)  # From CodeCarbon
    energy_consumed_kwh = Column(Float)  # From CodeCarbon
    error_message = Column(Text)
    
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