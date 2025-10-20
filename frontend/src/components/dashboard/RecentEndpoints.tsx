import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { apiService, Endpoint } from '../../services/api';

interface EndpointRowProps {
  name: string;
  status: 'active' | 'training' | 'ready';
  fromModel: string;
  toModel: string;
  similarity: number;
  examples: string;
  costSaved?: string;
  carbonSaved?: string;
  trainingProgress?: string;
  onActivate?: () => void;
}

const EndpointRow: React.FC<EndpointRowProps> = ({
  name,
  status,
  fromModel,
  toModel,
  similarity,
  examples,
  costSaved,
  carbonSaved,
  trainingProgress,
  onActivate
}) => {
  const statusConfig = {
    active: {
      color: 'border-green-500',
      badge: 'bg-green-100 text-green-800 border-green-200',
      dot: 'bg-green-500',
      label: 'Active SLM'
    },
    training: {
      color: 'border-yellow-500',
      badge: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      dot: 'bg-yellow-500',
      label: 'Training'
    },
    ready: {
      color: 'border-blue-500',
      badge: 'bg-blue-100 text-blue-800 border-blue-200',
      dot: 'bg-blue-500',
      label: 'Ready'
    }
  };

  const config = statusConfig[status];

  return (
    <div className={`bg-white border border-gray-200 ${config.color} border-l-4 rounded-lg p-6 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200`}>
      {/* Row 1: Title + Badge */}
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-lg font-semibold text-gray-900">{name}</h3>
        <div className={`flex items-center px-3 py-1 rounded-full text-xs font-medium border ${config.badge}`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${config.dot}`}></div>
          {config.label}
        </div>
      </div>

      {/* Row 2: Model Info */}
      <div className="text-sm text-gray-600 mb-3">
        {fromModel} <span className="text-gray-400">→</span> {toModel}
      </div>

      {/* Row 3: Progress Bar */}
      <div className="flex items-center mb-3">
        <div className="w-96 bg-gray-200 rounded-full h-1.5 mr-3">
          <div 
            className="bg-gradient-to-r from-blue-500 to-green-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${similarity}%` }}
          />
        </div>
        <span className="text-sm font-semibold text-gray-900 mr-2">{similarity}% similar</span>
        <span className="text-sm text-gray-600">• {examples} examples</span>
      </div>

      {/* Row 4: Metrics or Status */}
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">
          {status === 'active' && costSaved && carbonSaved && (
            <span>
              <span className="text-teal-600">{costSaved} saved</span>
              <span className="text-gray-400 mx-2">•</span>
              <span className="text-purple-600">{carbonSaved} CO₂ saved</span>
            </span>
          )}
          {status === 'training' && trainingProgress && (
            <span className="text-yellow-700 flex items-center">
              <div className="w-2 h-2 bg-yellow-500 rounded-full mr-2 animate-pulse"></div>
              {trainingProgress}
            </span>
          )}
          {status === 'ready' && (
            <span className="text-gray-700">Ready to activate</span>
          )}
        </div>
        
        {status === 'ready' && onActivate && (
          <button 
            onClick={onActivate}
            className="px-4 py-1.5 bg-blue-600 text-white text-sm font-semibold rounded-md hover:bg-blue-700 transition-colors"
          >
            Switch to SLM →
          </button>
        )}
      </div>
    </div>
  );
};

export const RecentEndpoints: React.FC = () => {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [endpointMetrics, setEndpointMetrics] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEndpoints = async () => {
      try {
        const data = await apiService.getEndpoints();
        setEndpoints(data);
        
        // Fetch metrics for each endpoint
        const metricsPromises = data.map(async (endpoint) => {
          try {
            const metrics = await apiService.getMetrics(endpoint.id);
            return { [endpoint.id]: metrics };
          } catch (error) {
            console.error(`Failed to fetch metrics for ${endpoint.id}:`, error);
            return { [endpoint.id]: null };
          }
        });
        
        const metricsResults = await Promise.all(metricsPromises);
        const metricsMap: Record<string, any> = {};
        metricsResults.forEach(result => {
          Object.assign(metricsMap, result);
        });
        setEndpointMetrics(metricsMap);
      } catch (error) {
        console.error('Failed to fetch endpoints:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchEndpoints();
  }, []);

  const handleActivate = async (endpointId: string) => {
    try {
      await apiService.activateEndpoint(endpointId);
      // Refresh endpoints after activation
      const data = await apiService.getEndpoints();
      setEndpoints(data);
    } catch (error) {
      console.error('Failed to activate endpoint:', error);
    }
  };

  const mapStatus = (status: string): 'active' | 'training' | 'ready' => {
    switch (status) {
      case 'active':
        return 'active';
      case 'training':
        return 'training';
      default:
        return 'ready';
    }
  };

  if (loading) {
    return (
      <section className="mb-8">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </section>
    );
  }

  return (
    <section className="mb-8">
      {/* Section Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-gray-900">Recent Endpoints</h2>
        <Link 
          to="/endpoints/new"
          className="px-5 py-2.5 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 hover:shadow-md hover:scale-105 transition-all duration-150"
        >
          + New Endpoint
        </Link>
      </div>

      {/* Endpoints List */}
      <div className="space-y-4">
        {endpoints.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p>No endpoints found. Create your first endpoint to get started.</p>
          </div>
        ) : (
          endpoints.map((endpoint) => {
            const metrics = endpointMetrics[endpoint.id];
            const similarity = metrics?.avg_similarity ? Math.round(metrics.avg_similarity * 100) : 0;
            const totalInferences = metrics?.total_inferences || 0;
            
            return (
              <EndpointRow
                key={endpoint.id}
                name={endpoint.name}
                status={mapStatus(endpoint.status)}
                fromModel={endpoint.llm_model}
                toModel={endpoint.slm_model_path || 'Llama 3.2 1B'}
                similarity={similarity}
                examples={totalInferences.toString()}
                onActivate={endpoint.status === 'ready' ? () => handleActivate(endpoint.id) : undefined}
                trainingProgress={endpoint.status === 'training' ? 'Training in progress...' : undefined}
              />
            );
          })
        )}
      </div>

      {/* Section Footer */}
      <div className="text-right mt-4">
        <Link 
          to="/endpoints"
          className="text-sm font-medium text-blue-600 hover:text-blue-700 hover:underline transition-colors"
        >
          View All Endpoints →
        </Link>
      </div>
    </section>
  );
};