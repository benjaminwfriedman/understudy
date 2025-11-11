from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# Endpoint schemas
class EndpointConfigCreate(BaseModel):
    training_batch_size: int = 100
    similarity_threshold: float = 0.95
    auto_switchover: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    learning_rate: float = 0.0003
    track_carbon: bool = True
    max_training_examples: int = 1000
    training_frequency_hours: int = 24
    enable_compression: bool = False
    compression_target_ratio: Optional[float] = None


class EndpointCreate(BaseModel):
    name: str
    description: Optional[str] = None
    llm_provider: str
    llm_model: str
    config: Optional[EndpointConfigCreate] = None


class EndpointConfigResponse(BaseModel):
    training_batch_size: int
    similarity_threshold: float
    auto_switchover: bool
    lora_r: int
    lora_alpha: int
    learning_rate: float
    track_carbon: bool
    max_training_examples: int
    training_frequency_hours: int
    enable_compression: bool
    compression_target_ratio: Optional[float]
    
    class Config:
        from_attributes = True


class EndpointResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    llm_provider: str
    llm_model: str
    slm_model_path: Optional[str]
    deployed_version: Optional[int]
    deployed_semantic_similarity: Optional[float]
    status: str
    langchain_compatible: bool
    created_at: datetime
    updated_at: datetime
    config: Optional[EndpointConfigResponse] = None
    
    class Config:
        from_attributes = True


# Inference schemas
class InferenceRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: Optional[List[str]] = None
    langchain_metadata: Optional[Dict[str, Any]] = None


class InferenceResponse(BaseModel):
    output: str
    model_used: str
    latency_ms: int
    cost_usd: float
    carbon_emissions: Optional[Dict[str, Any]] = None


# Training schemas
class TrainingRequest(BaseModel):
    num_examples: Optional[int] = None
    force_retrain: bool = False
    provider: Optional[str] = None  # "runpod", "lambda", "azure", or None for local
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 4
    learning_rate: Optional[float] = 2e-4
    priority: Optional[int] = 0


class TrainingResponse(BaseModel):
    training_run_id: str
    status: str
    message: str


class TrainingRunResponse(BaseModel):
    train_id: str
    endpoint_id: str
    version: int
    created_at: datetime
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    slm_type: str
    source_llm: str
    training_pairs_count: int
    training_loss: Optional[float]
    training_time_wall: Optional[float]
    semantic_similarity_score: Optional[float]
    phase: str
    carbon_emissions_kg: Optional[float]
    energy_consumed_kwh: Optional[float]
    
    class Config:
        from_attributes = True


# Comparison schemas
class ComparisonResponse(BaseModel):
    id: str
    endpoint_id: str
    training_run_id: str
    input_text: str
    llm_output: str
    slm_output: str
    semantic_similarity_score: Optional[float]
    llm_latency_ms: Optional[int]
    slm_latency_ms: Optional[int]
    llm_cost_usd: Optional[float]
    slm_cost_usd: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


# Metrics schemas
class MetricResponse(BaseModel):
    id: str
    endpoint_id: str
    metric_type: str
    value: float
    calculated_at: datetime
    
    class Config:
        from_attributes = True


class MetricsSummary(BaseModel):
    endpoint_id: str
    avg_similarity: float
    total_inferences: int
    llm_inferences: int
    slm_inferences: int
    total_cost_saved: float
    avg_latency_reduction_ms: float
    llm_avg_cost: Optional[float] = None
    slm_avg_cost: Optional[float] = None
    total_compressed_requests: Optional[int] = None
    avg_tokens_saved: Optional[float] = None
    compression_cost_savings: Optional[float] = None


# Carbon schemas
class CarbonSummary(BaseModel):
    total_training_emissions_kg: float
    total_inference_emissions_kg: float
    avoided_emissions_kg: float
    net_emissions_saved_kg: float
    carbon_payback_achieved: bool
    estimated_inferences_to_payback: Optional[int]


class CarbonTimeline(BaseModel):
    timeline: List[Dict[str, Any]]


# Status schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    database: bool
    carbon_tracking: bool


# Examples schemas
class ExampleResponse(BaseModel):
    id: str
    endpoint_id: str
    input_text: str
    llm_output: Optional[str]
    slm_output: Optional[str]
    model_used: str
    latency_ms: Optional[int]
    cost_usd: Optional[float]
    created_at: datetime
    langchain_metadata: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True
        protected_namespaces = ()


class ExamplesListResponse(BaseModel):
    examples: List[ExampleResponse]
    total_count: int
    trained_count: int
    pending_count: int