import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  Endpoint,
  CreateEndpointRequest,
  InferenceRequest,
  InferenceResponse,
  TrainingRun,
  MetricsSummary,
  CarbonSummary,
  CarbonTimelinePoint,
  HealthCheck,
} from '../types/api';

class UnderstudyAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = '/api/v1') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for auth
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('understudy_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        throw error;
      }
    );
  }

  // Endpoints
  async createEndpoint(data: CreateEndpointRequest): Promise<Endpoint> {
    const response: AxiosResponse<Endpoint> = await this.client.post('/endpoints', data);
    return response.data;
  }

  async listEndpoints(skip = 0, limit = 100): Promise<Endpoint[]> {
    const response: AxiosResponse<Endpoint[]> = await this.client.get('/endpoints', {
      params: { skip, limit },
    });
    return response.data;
  }

  async getEndpoint(endpointId: string): Promise<Endpoint> {
    const response: AxiosResponse<Endpoint> = await this.client.get(`/endpoints/${endpointId}`);
    return response.data;
  }

  async deleteEndpoint(endpointId: string): Promise<void> {
    await this.client.delete(`/endpoints/${endpointId}`);
  }

  async activateEndpoint(endpointId: string): Promise<void> {
    await this.client.post(`/endpoints/${endpointId}/activate`);
  }

  // Inference
  async inference(endpointId: string, request: InferenceRequest): Promise<InferenceResponse> {
    const response: AxiosResponse<InferenceResponse> = await this.client.post(
      `/inference/${endpointId}`,
      request
    );
    return response.data;
  }

  // Training
  async startTraining(endpointId: string, numExamples?: number): Promise<{ training_run_id: string; status: string; message: string }> {
    const response = await this.client.post(`/training/${endpointId}`, {
      num_examples: numExamples,
    });
    return response.data;
  }

  async getTrainingRuns(endpointId: string): Promise<TrainingRun[]> {
    const response: AxiosResponse<TrainingRun[]> = await this.client.get(`/training/${endpointId}/runs`);
    return response.data;
  }

  // Metrics
  async getMetrics(endpointId: string, days = 30): Promise<MetricsSummary> {
    const response: AxiosResponse<MetricsSummary> = await this.client.get(`/metrics/${endpointId}`, {
      params: { days },
    });
    return response.data;
  }

  // Carbon
  async getCarbonSummary(endpointId: string): Promise<CarbonSummary> {
    const response: AxiosResponse<CarbonSummary> = await this.client.get(`/carbon/${endpointId}/summary`);
    return response.data;
  }

  async getCarbonTimeline(endpointId: string, days = 30): Promise<CarbonTimelinePoint[]> {
    const response: AxiosResponse<{ timeline: CarbonTimelinePoint[] }> = await this.client.get(
      `/carbon/${endpointId}/timeline`,
      { params: { days } }
    );
    return response.data.timeline;
  }

  // Health
  async healthCheck(): Promise<HealthCheck> {
    const response: AxiosResponse<HealthCheck> = await this.client.get('/health');
    return response.data;
  }
}

export const api = new UnderstudyAPI();