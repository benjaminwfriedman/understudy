import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Sidebar } from '../components/layout/Sidebar';
import { TopBar } from '../components/layout/TopBar';
import { apiService, Endpoint, Metrics, Example, ExamplesResponse } from '../services/api';

interface TrainingRun {
  id: string;
  endpoint_id: string;
  start_time: string;
  end_time: string | null;
  examples_used: number | null;
  final_loss: number | null;
  status: string;
  carbon_emissions_kg: number | null;
  energy_consumed_kwh: number | null;
}

interface TabProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TabNavigation: React.FC<TabProps> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'training', label: 'Training' },
    { id: 'examples', label: 'Examples' },
    { id: 'comparisons', label: 'Comparisons' },
    { id: 'settings', label: 'Settings' }
  ];

  return (
    <div className="sticky top-16 bg-white border-b-2 border-gray-200 z-10">
      <div className="px-8">
        <nav className="flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`h-14 px-5 text-sm font-semibold transition-all duration-150 border-b-3 ${
                activeTab === tab.id
                  ? 'text-blue-600 border-blue-600'
                  : 'text-gray-600 border-transparent hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
};

interface OverviewTabProps {
  endpoint: Endpoint;
  metrics: Metrics | null;
  trainingRuns: TrainingRun[];
  examplesStats: { total: number; trained: number; pending: number } | null;
}

const OverviewTab: React.FC<OverviewTabProps> = ({ endpoint, metrics, trainingRuns, examplesStats }) => {
  const isActive = endpoint.status === 'active';
  const isTraining = endpoint.status === 'training';

  return (
    <div className="p-8 pt-8">
      {/* Section 1: Status & Quick Actions */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {/* Current Status Card */}
        <div className="col-span-2 bg-white border border-gray-200 rounded-xl p-8 shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Current Status</h3>
          
          <div className="flex items-center gap-8">
            {/* Status Badge */}
            <div className={`w-30 h-30 rounded-2xl flex flex-col items-center justify-center ${
              isActive 
                ? 'bg-gradient-to-br from-green-100 to-green-50 border-2 border-green-500' 
                : isTraining
                ? 'bg-gradient-to-br from-yellow-100 to-yellow-50 border-2 border-yellow-500'
                : 'bg-gradient-to-br from-blue-100 to-blue-50 border-2 border-blue-500'
            }`}>
              <div className={`text-3xl mb-2 ${
                isActive ? 'text-green-600' : isTraining ? 'text-yellow-600' : 'text-blue-600'
              }`}>
                {isActive ? '⚡' : isTraining ? '🔄' : '⏸️'}
              </div>
              <span className={`text-sm font-semibold ${
                isActive ? 'text-green-800' : isTraining ? 'text-yellow-800' : 'text-blue-800'
              }`}>
                {isActive ? 'Active SLM' : isTraining ? 'Training' : 'Ready'}
              </span>
            </div>

            {/* Metrics */}
            <div className="flex-1 space-y-3">
              {isActive && metrics ? (
                <>
                  <div className="flex items-center">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-gray-900">
                      {Math.round((metrics.avg_similarity || 0) * 100)}% semantic similarity
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-gray-900">
                      {metrics.avg_latency_reduction_ms || 0}ms avg latency (faster)
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-gray-900">
                      ${metrics.total_cost_saved?.toFixed(4) || '0.0000'} per request (cheaper)
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-gray-900">0.0002g CO₂ per request</span>
                  </div>
                </>
              ) : isTraining ? (
                <>
                  <div className="flex items-center">
                    <span className="text-yellow-500 mr-3">⏱️</span>
                    <span className="text-gray-900">Model training in progress</span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-blue-500 mr-3">📊</span>
                    <span className="text-gray-900">{metrics?.total_inferences || 0} examples processed</span>
                  </div>
                </>
              ) : (
                <div className="flex items-center">
                  <span className="text-blue-500 mr-3">⏸️</span>
                  <span className="text-gray-900">Ready to activate</span>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 mt-6">
            <button className="px-5 py-2.5 bg-white border border-gray-300 text-gray-700 text-sm font-semibold rounded-lg hover:bg-gray-50 transition-colors">
              {isActive ? 'Revert to LLM' : 'Configure'}
            </button>
            <button className="px-5 py-2.5 bg-white border border-gray-300 text-gray-700 text-sm font-semibold rounded-lg hover:bg-gray-50 transition-colors">
              {isTraining ? 'Pause Training' : 'Start Training'}
            </button>
            <button className="px-5 py-2.5 bg-blue-50 text-blue-600 text-sm font-semibold rounded-lg hover:bg-blue-100 transition-colors">
              View Logs
            </button>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-5">🎛️ Controls</h3>
          
          <div className="space-y-6">
            {/* Auto-switchover Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Auto-switchover</label>
              <div className="flex items-center justify-between">
                <div className="w-11 h-6 bg-green-500 rounded-full relative">
                  <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5 shadow-sm"></div>
                </div>
                <span className="text-sm font-medium text-gray-700">ON</span>
              </div>
            </div>

            {/* Similarity Threshold */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Similarity Threshold</label>
              <div className="relative">
                <input 
                  type="range" 
                  min="0.8" 
                  max="1.0" 
                  step="0.01" 
                  defaultValue="0.95"
                  className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.80</span>
                  <span className="text-blue-600 font-semibold">0.95</span>
                  <span>1.00</span>
                </div>
              </div>
            </div>

            {/* Training Batch Size */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Training Batch Size</label>
              <div className="flex items-center gap-2">
                <input 
                  type="number" 
                  defaultValue="100"
                  className="w-20 px-3 py-2 border border-gray-300 rounded-md text-sm"
                />
                <span className="text-sm text-gray-600">examples</span>
              </div>
            </div>

            {/* Trigger Training Button */}
            <button className="w-full bg-blue-600 text-white text-sm font-semibold py-3 rounded-lg hover:bg-blue-700 transition-colors">
              🔄 Trigger Training Now
            </button>
          </div>
        </div>
      </div>

      {/* Section 2: Training Progress */}
      <div className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm mb-8">
        <h3 className="text-xl font-semibold text-gray-900 mb-6">Training Progress</h3>
        
        {examplesStats ? (
          <>
            <div className="flex items-center mb-4">
              <div className="flex-1 bg-gray-200 rounded-full h-3 mr-4">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min((examplesStats.total / 100) * 100, 100)}%` }}
                />
              </div>
              <span className="text-lg font-semibold text-gray-900">
                {examplesStats.total}/100
              </span>
            </div>

            <div className="space-y-2 mb-6">
              {examplesStats.total < 100 ? (
                <>
                  <div className="flex items-center">
                    <span className="text-blue-600 mr-3">📊</span>
                    <span className="text-gray-900">Status: Collecting training data</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    Need {100 - examplesStats.total} more examples to start training
                  </div>
                  <div className="text-sm text-gray-600">
                    Current examples: {examplesStats.pending} pending, {examplesStats.trained} trained
                  </div>
                </>
              ) : trainingRuns.length === 0 ? (
                <>
                  <div className="flex items-center">
                    <span className="text-green-600 mr-3">✅</span>
                    <span className="text-gray-900">Status: Ready for training</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {examplesStats.total} examples collected - ready to start training
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center">
                    <span className="text-yellow-600 mr-3">⏱️</span>
                    <span className="text-gray-900">Status: Training in progress</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {trainingRuns.length} training run(s) completed
                  </div>
                  {trainingRuns[0] && (
                    <div className="text-sm text-gray-600">
                      Last training: {new Date(trainingRuns[0].start_time).toLocaleDateString()}
                      {trainingRuns[0].examples_used && ` (${trainingRuns[0].examples_used} examples)`}
                    </div>
                  )}
                </>
              )}
            </div>

            <Link 
              to={`/endpoints/${endpoint.id}/training`}
              className="text-blue-600 font-medium text-sm hover:text-blue-700 hover:underline"
            >
              View Training History →
            </Link>
          </>
        ) : (
          <div className="animate-pulse">
            <div className="h-3 bg-gray-200 rounded mb-4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/3 mb-2"></div>
            <div className="h-3 bg-gray-200 rounded w-1/2"></div>
          </div>
        )}
      </div>

      {/* Section 3: Key Metrics Grid */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <h4 className="text-sm font-medium text-gray-600 mb-2">Total Requests</h4>
          <div className="text-3xl font-bold text-gray-900 mb-1">{metrics?.total_inferences || 0}</div>
          <div className="text-sm text-green-600">↑ 234 today</div>
        </div>
        
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <h4 className="text-sm font-medium text-gray-600 mb-2">SLM Usage</h4>
          <div className="text-3xl font-bold text-blue-600 mb-1">{metrics?.slm_inferences || 0}</div>
          <div className="text-sm text-gray-600">
            {metrics?.total_inferences ? Math.round((metrics.slm_inferences / metrics.total_inferences) * 100) : 0}%
          </div>
        </div>
        
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <h4 className="text-sm font-medium text-gray-600 mb-2">Cost Saved</h4>
          <div className="text-3xl font-bold text-teal-600 mb-1">${metrics?.total_cost_saved?.toFixed(2) || '0.00'}</div>
          <div className="text-sm text-green-600">↑ $38.94/week</div>
        </div>
        
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <h4 className="text-sm font-medium text-gray-600 mb-2">Carbon Saved</h4>
          <div className="text-3xl font-bold text-purple-600 mb-1">8.3 kg CO₂</div>
          <div className="text-sm text-green-600">↑ 0.4kg/week</div>
        </div>
      </div>
    </div>
  );
};

interface ExamplesTabProps {
  endpointId: string;
}

const ExamplesTab: React.FC<ExamplesTabProps> = ({ endpointId }) => {
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'trained' | 'pending'>('all');
  const [search, setSearch] = useState('');
  const [stats, setStats] = useState({ total: 0, trained: 0, pending: 0 });
  const [selectedExample, setSelectedExample] = useState<Example | null>(null);

  useEffect(() => {
    fetchExamples();
  }, [endpointId, filter, search]);

  const fetchExamples = async () => {
    try {
      setLoading(true);
      const filterTrained = filter === 'all' ? undefined : filter === 'trained';
      const response = await apiService.getExamples(endpointId, {
        limit: 50,
        filter_trained: filterTrained,
        search: search || undefined
      });
      
      setExamples(response.examples);
      setStats({
        total: response.total_count,
        trained: response.trained_count,
        pending: response.pending_count
      });
    } catch (error) {
      console.error('Failed to fetch examples:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else {
      return 'Just now';
    }
  };

  const truncateText = (text: string, maxLength: number = 100) => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
  };

  return (
    <div className="p-8">
      {/* Filters & Stats Bar */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <select 
              value={filter}
              onChange={(e) => setFilter(e.target.value as 'all' | 'trained' | 'pending')}
              className="px-3 py-2 bg-gray-100 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-200"
            >
              <option value="all">All Examples</option>
              <option value="trained">Trained</option>
              <option value="pending">Pending</option>
            </select>
            
            <input
              type="text"
              placeholder="Search examples..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-60 px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-600 focus:border-transparent"
            />
          </div>
          
          <div className="text-sm text-gray-600">
            {stats.total} total • {stats.trained} trained • {stats.pending} pending
          </div>
        </div>
      </div>

      {/* Examples List */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white border border-gray-200 rounded-xl p-6 animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
              <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-3/4"></div>
            </div>
          ))}
        </div>
      ) : examples.length === 0 ? (
        <div className="text-center py-16">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No examples yet</h3>
          <p className="text-gray-600 mb-6">
            Start making inference calls to this endpoint to collect training examples
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {examples.map((example) => (
            <div key={example.id} className="bg-white border border-gray-200 rounded-xl p-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Example #{example.id.slice(0, 8)}
                  </h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    example.slm_output 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {example.slm_output ? 'Trained ✓' : 'Pending'}
                  </span>
                </div>
                <div className="text-sm text-gray-500">
                  {formatTimestamp(example.created_at)} • Used {example.model_used.toUpperCase()}
                </div>
              </div>

              {/* Input */}
              <div className="mb-4">
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                  Input:
                </label>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-sm text-gray-900">{truncateText(example.input_text)}</p>
                </div>
              </div>

              {/* LLM Output */}
              <div className="mb-4">
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                  LLM Output:
                </label>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-sm text-gray-900">
                    {example.llm_output ? truncateText(example.llm_output, 150) : 'N/A'}
                  </p>
                </div>
              </div>

              {/* Metadata */}
              <div className="flex items-center justify-between text-sm text-gray-500">
                <div>
                  {example.latency_ms && `${example.latency_ms}ms`}
                  {example.cost_usd && ` • $${example.cost_usd.toFixed(4)}`}
                </div>
                <div className="space-x-4">
                  <button 
                    onClick={() => setSelectedExample(example)}
                    className="text-blue-600 font-medium hover:text-blue-700 hover:underline"
                  >
                    View Full Example →
                  </button>
                  {example.slm_output && (
                    <button className="text-blue-600 font-medium hover:text-blue-700 hover:underline">
                      Compare with SLM →
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Example Detail Modal */}
      {selectedExample && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl max-w-4xl max-h-[80vh] overflow-auto p-8 m-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold text-gray-900">
                Example #{selectedExample.id.slice(0, 8)}
              </h2>
              <button 
                onClick={() => setSelectedExample(null)}
                className="text-gray-400 hover:text-gray-600 text-2xl"
              >
                ✕
              </button>
            </div>

            {/* Metadata */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-semibold text-gray-900 mb-2">Metadata:</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Collected: {new Date(selectedExample.created_at).toLocaleString()}</li>
                <li>• Model Used: {selectedExample.model_used.toUpperCase()}</li>
                <li>• Latency: {selectedExample.latency_ms}ms</li>
                <li>• Cost: ${selectedExample.cost_usd?.toFixed(4)}</li>
                <li>• Status: {selectedExample.slm_output ? 'Trained' : 'Pending training'}</li>
              </ul>
            </div>

            <div className="border-t border-gray-200 pt-6">
              {/* Input */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-3">Input:</h3>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-900">{selectedExample.input_text}</p>
                </div>
              </div>

              {/* LLM Output */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-3">LLM Output:</h3>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="text-gray-900">
                    {selectedExample.llm_output || 'No output available'}
                  </p>
                  <div className="mt-3 pt-3 border-t border-red-200 text-xs text-gray-600">
                    Cost: ${selectedExample.cost_usd?.toFixed(4)} • Latency: {selectedExample.latency_ms}ms
                  </div>
                </div>
              </div>

              {/* SLM Output (if available) */}
              {selectedExample.slm_output && (
                <div className="mb-6">
                  <h3 className="font-semibold text-gray-900 mb-3">SLM Output:</h3>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <p className="text-gray-900">{selectedExample.slm_output}</p>
                    <div className="mt-3 pt-3 border-t border-green-200 text-xs text-gray-600">
                      Cost: ~$0.0001 • Latency: ~45ms (estimated)
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const EndpointDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [endpoint, setEndpoint] = useState<Endpoint | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [examplesStats, setExamplesStats] = useState<{ total: number; trained: number; pending: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      if (!id) return;
      
      try {
        const [endpointData, metricsData, trainingRunsData, examplesData] = await Promise.all([
          apiService.getEndpoint(id),
          apiService.getMetrics(id).catch(() => null), // Metrics might not exist
          fetch(`/api/v1/training/${id}/runs`).then(r => r.ok ? r.json() : []).catch(() => []),
          apiService.getExamples(id, { limit: 1 }).then(data => ({
            total: data.total_count,
            trained: data.trained_count,
            pending: data.pending_count
          })).catch(() => null)
        ]);
        
        setEndpoint(endpointData);
        setMetrics(metricsData);
        setTrainingRuns(trainingRunsData);
        setExamplesStats(examplesData);
      } catch (error) {
        console.error('Failed to fetch endpoint data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Sidebar />
        <div className="ml-60">
          <TopBar />
          <div className="p-8">
            <div className="animate-pulse">
              <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/2 mb-8"></div>
              <div className="grid grid-cols-3 gap-6">
                <div className="col-span-2 h-64 bg-gray-200 rounded-xl"></div>
                <div className="h-64 bg-gray-200 rounded-xl"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!endpoint) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Sidebar />
        <div className="ml-60">
          <TopBar />
          <div className="p-8">
            <div className="text-center py-16">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">Endpoint not found</h2>
              <p className="text-gray-600 mb-6">The endpoint you're looking for doesn't exist.</p>
              <Link 
                to="/endpoints"
                className="text-blue-600 font-medium hover:text-blue-700 hover:underline"
              >
                ← Back to Endpoints
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const statusConfig = {
    active: { badge: 'bg-green-100 text-green-800', dot: 'bg-green-500', label: 'Active SLM' },
    training: { badge: 'bg-yellow-100 text-yellow-800', dot: 'bg-yellow-500', label: 'Training' },
    ready: { badge: 'bg-blue-100 text-blue-800', dot: 'bg-blue-500', label: 'Ready' },
    failed: { badge: 'bg-red-100 text-red-800', dot: 'bg-red-500', label: 'Failed' }
  };

  const config = statusConfig[endpoint.status as keyof typeof statusConfig] || statusConfig.ready;
  const createdDate = new Date(endpoint.created_at).toLocaleDateString();

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      
      <div className="ml-60">
        <TopBar />
        
        <div className="p-8 pb-4">
          {/* Back Link */}
          <Link 
            to="/endpoints"
            className="text-sm font-medium text-blue-600 hover:text-blue-700 hover:underline mb-4 inline-block"
          >
            ← Back to Endpoints
          </Link>

          {/* Title Row */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-gray-900">{endpoint.name}</h1>
              <p className="text-gray-600 mt-2">{endpoint.description}</p>
              <div className="text-sm text-gray-500 mt-2">
                {endpoint.llm_model} <span className="text-gray-400">→</span> {endpoint.slm_model_path || 'Llama 3.2 1B'} • 
                Created {createdDate} • Last trained 2h ago
              </div>
            </div>
            
            <div className={`flex items-center px-4 py-2 rounded-full text-sm font-semibold ${config.badge}`}>
              <div className={`w-2.5 h-2.5 rounded-full mr-2 ${config.dot}`}></div>
              {config.label}
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Tab Content */}
        {activeTab === 'overview' && <OverviewTab endpoint={endpoint} metrics={metrics} trainingRuns={trainingRuns} examplesStats={examplesStats} />}
        {activeTab === 'training' && (
          <div className="p-8">
            <div className="text-center py-16">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Training Tab</h3>
              <p className="text-gray-600">Training history and details will be displayed here.</p>
            </div>
          </div>
        )}
        {activeTab === 'examples' && <ExamplesTab endpointId={endpoint.id} />}
        {activeTab === 'comparisons' && (
          <div className="p-8">
            <div className="text-center py-16">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Comparisons Tab</h3>
              <p className="text-gray-600">LLM vs SLM output comparisons will be displayed here.</p>
            </div>
          </div>
        )}
        {activeTab === 'settings' && (
          <div className="p-8">
            <div className="text-center py-16">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Settings Tab</h3>
              <p className="text-gray-600">Endpoint configuration settings will be displayed here.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};