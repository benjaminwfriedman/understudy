import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Sidebar } from '../components/layout/Sidebar';
import { TopBar } from '../components/layout/TopBar';
import { apiService, Endpoint } from '../services/api';

interface EndpointStats {
  examples_total: number;
  examples_trained: number;
  target_batch_size: number;
  semantic_similarity?: number;
}

interface EndpointCardProps {
  endpoint: Endpoint;
  stats?: EndpointStats;
}

const EndpointCard: React.FC<EndpointCardProps> = ({ endpoint, stats }) => {
  const statusConfig = {
    active: {
      color: 'border-green-500',
      badge: 'bg-green-100 text-green-800',
      dot: 'bg-green-500',
      label: 'Active SLM'
    },
    training: {
      color: 'border-yellow-500',
      badge: 'bg-yellow-100 text-yellow-800',
      dot: 'bg-yellow-500',
      label: 'Training'
    },
    ready: {
      color: 'border-blue-500',
      badge: 'bg-blue-100 text-blue-800',
      dot: 'bg-blue-500',
      label: 'Ready'
    },
    failed: {
      color: 'border-red-500',
      badge: 'bg-red-100 text-red-800',
      dot: 'bg-red-500',
      label: 'Failed'
    }
  };

  const config = statusConfig[endpoint.status as keyof typeof statusConfig] || statusConfig.ready;
  const createdDate = new Date(endpoint.created_at).toLocaleDateString();
  const updatedDate = new Date(endpoint.updated_at).toLocaleDateString();

  return (
    <Link 
      to={`/endpoints/${endpoint.id}`}
      className={`block bg-white border border-gray-200 ${config.color} border-l-4 rounded-lg p-6 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xl font-semibold text-gray-900">{endpoint.name}</h3>
        <div className={`flex items-center px-3 py-1 rounded-full text-sm font-medium ${config.badge}`}>
          <div className={`w-2.5 h-2.5 rounded-full mr-2 ${config.dot}`}></div>
          {config.label}
        </div>
      </div>

      {/* Description */}
      <p className="text-gray-600 text-sm mb-3">{endpoint.description}</p>

      {/* Model Info */}
      <div className="text-sm text-gray-500 mb-4">
        {endpoint.llm_model} <span className="text-gray-400">→</span> Llama 3.2 1B
      </div>

      {/* Training Progress & Stats */}
      {stats && (
        <div className="mb-4 space-y-2">
          {/* Progress Bar */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Training Progress</span>
            <span className="text-gray-600">{stats.examples_total}/{stats.target_batch_size}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min((stats.examples_total / stats.target_batch_size) * 100, 100)}%` }}
            />
          </div>
          
          {/* Similarity Score */}
          {stats.semantic_similarity && (
            <div className="text-sm text-gray-600">
              Similarity: <span className="font-medium text-gray-900">{Math.round(stats.semantic_similarity * 100)}%</span>
            </div>
          )}
        </div>
      )}

      {/* Metadata */}
      <div className="text-xs text-gray-500">
        Created {createdDate} • Last updated {updatedDate}
      </div>
    </Link>
  );
};

export const Endpoints: React.FC = () => {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [endpointStats, setEndpointStats] = useState<Map<string, EndpointStats>>(new Map());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEndpoints = async () => {
      try {
        const data = await apiService.getEndpoints();
        setEndpoints(data);
        
        // Fetch stats for each endpoint
        const statsPromises = data.map(async (endpoint) => {
          try {
            const examples = await apiService.getExamples(endpoint.id, { limit: 1 });
            const stats: EndpointStats = {
              examples_total: examples.total_count,
              examples_trained: examples.trained_count,
              target_batch_size: endpoint.config?.training_batch_size || 100,
              semantic_similarity: endpoint.deployed_semantic_similarity || undefined
            };
            return { id: endpoint.id, stats };
          } catch (error) {
            console.error(`Failed to fetch stats for ${endpoint.id}:`, error);
            return null;
          }
        });
        
        const statsResults = await Promise.all(statsPromises);
        const statsMap = new Map();
        statsResults.forEach(result => {
          if (result) {
            statsMap.set(result.id, result.stats);
          }
        });
        setEndpointStats(statsMap);
      } catch (error) {
        console.error('Failed to fetch endpoints:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchEndpoints();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Top Bar - Full Width */}
      <div className="ml-60">
        <TopBar />
      </div>
      
      {/* Main Content */}
      <div className="ml-60">
        <main className="p-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Endpoints</h1>
              <p className="text-gray-600 mt-2">Manage your Understudy endpoints</p>
            </div>
            <Link 
              to="/endpoints/new"
              className="px-5 py-2.5 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 hover:shadow-md hover:scale-105 transition-all duration-150"
            >
              + New Endpoint
            </Link>
          </div>

          {/* Content */}
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3].map((i) => (
                <div key={i} className="bg-white border border-gray-200 rounded-lg p-6 h-48">
                  <div className="animate-pulse">
                    <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
                    <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
                    <div className="h-3 bg-gray-200 rounded w-2/3 mb-4"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : endpoints.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-6xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">No endpoints yet</h3>
              <p className="text-gray-600 mb-6">Create your first Understudy endpoint to get started</p>
              <Link 
                to="/endpoints/new"
                className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
              >
                Create Your First Endpoint
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {endpoints.map((endpoint) => (
                <EndpointCard 
                  key={endpoint.id} 
                  endpoint={endpoint} 
                  stats={endpointStats.get(endpoint.id)}
                />
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};