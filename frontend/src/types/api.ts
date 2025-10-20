export interface Endpoint {
  id: string;
  name: string;
  description?: string;
  llm_provider: string;
  llm_model: string;
  slm_model_path?: string;
  status: 'training' | 'ready' | 'active';
  langchain_compatible: boolean;
  created_at: string;
  updated_at: string;
}

export interface EndpointConfig {
  training_batch_size: number;
  similarity_threshold: number;
  auto_switchover: boolean;
  lora_r: number;
  lora_alpha: number;
  learning_rate: number;
  track_carbon: boolean;
  max_training_examples: number;
  training_frequency_hours: number;
}

export interface CreateEndpointRequest {
  name: string;
  description?: string;
  llm_provider: string;
  llm_model: string;
  config?: Partial<EndpointConfig>;
}

export interface InferenceRequest {
  prompt?: string;
  messages?: Array<{role: string; content: string}>;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string[];
  langchain_metadata?: Record<string, any>;
}

export interface InferenceResponse {
  output: string;
  model_used: 'llm' | 'slm';
  latency_ms: number;
  cost_usd: number;
  carbon_emissions?: {
    emissions_kg: number;
    timestamp: string;
  };
}

export interface TrainingRun {
  id: string;
  endpoint_id: string;
  start_time: string;
  end_time?: string;
  examples_used?: number;
  final_loss?: number;
  status: 'running' | 'completed' | 'failed';
  carbon_emissions_kg?: number;
  energy_consumed_kwh?: number;
}

export interface MetricsSummary {
  endpoint_id: string;
  avg_similarity: number;
  total_inferences: number;
  llm_inferences: number;
  slm_inferences: number;
  total_cost_saved: number;
  avg_latency_reduction_ms: number;
}

export interface CarbonSummary {
  total_training_emissions_kg: number;
  total_inference_emissions_kg: number;
  avoided_emissions_kg: number;
  net_emissions_saved_kg: number;
  carbon_payback_achieved: boolean;
  estimated_inferences_to_payback?: number;
}

export interface CarbonTimelinePoint {
  date: string;
  training_emissions_kg: number;
  inference_emissions_kg: number;
  total_kg: number;
}

export interface HealthCheck {
  status: string;
  version: string;
  timestamp: string;
  database: boolean;
  carbon_tracking: boolean;
}