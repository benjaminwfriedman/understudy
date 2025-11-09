const API_BASE_URL = '/api/v1';

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

export interface Endpoint {
  id: string;
  name: string;
  description: string;
  llm_provider: string;
  llm_model: string;
  slm_model_path: string | null;
  deployed_version: number | null;
  deployed_semantic_similarity: number | null;
  status: 'training' | 'active' | 'ready' | 'failed';
  langchain_compatible: boolean;
  created_at: string;
  updated_at: string;
  config: EndpointConfig | null;
}

export interface TrainingRun {
  train_id: string;
  endpoint_id: string;
  version: number;
  created_at: string;
  start_time: string | null;
  end_time: string | null;
  slm_type: string;
  source_llm: string;
  training_pairs_count: number;
  training_loss: number | null;
  training_time_wall: number | null;
  semantic_similarity_score: number | null;
  phase: string;
  carbon_emissions_kg: number | null;
  energy_consumed_kwh: number | null;
}

export interface CarbonSummary {
  total_emissions_saved_kg: number;
  equivalent_car_miles: number;
  total_requests: number;
}

export interface Metrics {
  endpoint_id: string;
  avg_similarity: number;
  total_inferences: number;
  llm_inferences: number;
  slm_inferences: number;
  total_cost_saved: number;
  avg_latency_reduction_ms: number;
  llm_avg_cost?: number;
  slm_avg_cost?: number;
}

export interface Example {
  id: string;
  endpoint_id: string;
  input_text: string;
  llm_output: string | null;
  slm_output: string | null;
  model_used: string;
  latency_ms: number | null;
  cost_usd: number | null;
  created_at: string;
  langchain_metadata: any;
}

export interface ExamplesResponse {
  examples: Example[];
  total_count: number;
  trained_count: number;
  pending_count: number;
}

export interface Comparison {
  id: string;
  endpoint_id: string;
  training_run_id: string;
  input_text: string;
  llm_output: string;
  slm_output: string;
  semantic_similarity_score: number | null;
  llm_latency_ms: number | null;
  slm_latency_ms: number | null;
  llm_cost_usd: number | null;
  slm_cost_usd: number | null;
  created_at: string;
}

class ApiService {
  private async fetch<T>(endpoint: string): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('API call to:', url);
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log('Response status:', response.status);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Response error text:', errorText);
      throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
    }
    
    const data = await response.json();
    console.log('Response data:', data);
    return data;
  }

  async getEndpoints(): Promise<Endpoint[]> {
    return this.fetch<Endpoint[]>('/endpoints');
  }

  async getEndpoint(id: string): Promise<Endpoint> {
    return this.fetch<Endpoint>(`/endpoints/${id}`);
  }

  async getTrainingRuns(endpointId: string): Promise<TrainingRun[]> {
    return this.fetch<TrainingRun[]>(`/training/${endpointId}/runs`);
  }

  async getCarbonSummary(endpointId: string): Promise<CarbonSummary> {
    return this.fetch<CarbonSummary>(`/carbon/${endpointId}/summary`);
  }

  async getMetrics(endpointId: string): Promise<Metrics> {
    return this.fetch<Metrics>(`/metrics/${endpointId}`);
  }

  async activateEndpoint(endpointId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/endpoints/${endpointId}/activate`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  async revertToLLM(endpointId: string): Promise<void> {
    // This undeploys the SLM deployment, not the endpoint itself
    const response = await fetch(`${API_BASE_URL}/lifecycle/endpoints/${endpointId}/deployment`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  async deployModel(trainId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/lifecycle/models/${trainId}/deploy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }

  async getExamples(
    endpointId: string,
    options?: {
      skip?: number;
      limit?: number;
      filter_trained?: boolean;
      search?: string;
    }
  ): Promise<ExamplesResponse> {
    const params = new URLSearchParams();
    if (options?.skip !== undefined) params.append('skip', options.skip.toString());
    if (options?.limit !== undefined) params.append('limit', options.limit.toString());
    if (options?.filter_trained !== undefined) params.append('filter_trained', options.filter_trained.toString());
    if (options?.search) params.append('search', options.search);

    const queryString = params.toString();
    const endpoint = `/endpoints/${endpointId}/examples${queryString ? `?${queryString}` : ''}`;
    
    return this.fetch<ExamplesResponse>(endpoint);
  }

  async getComparisons(
    endpointId: string,
    options?: {
      skip?: number;
      limit?: number;
    }
  ): Promise<Comparison[]> {
    const params = new URLSearchParams();
    if (options?.skip !== undefined) params.append('skip', options.skip.toString());
    if (options?.limit !== undefined) params.append('limit', options.limit.toString());

    const queryString = params.toString();
    const endpoint = `/endpoints/${endpointId}/comparisons${queryString ? `?${queryString}` : ''}`;
    
    return this.fetch<Comparison[]>(endpoint);
  }
}

export const apiService = new ApiService();