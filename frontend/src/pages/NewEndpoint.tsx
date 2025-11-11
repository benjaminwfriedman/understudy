import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/layout/Sidebar';
import { TopBar } from '../components/layout/TopBar';

export const NewEndpoint: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    llm_provider: 'openai',
    llm_model: 'gpt-4o-mini',
    langchain_compatible: true,
    enable_compression: false,
    compression_target_ratio: null as number | null
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('/api/v1/endpoints', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const newEndpoint = await response.json();
        navigate(`/endpoints/${newEndpoint.id}`);
      } else {
        console.error('Failed to create endpoint');
      }
    } catch (error) {
      console.error('Error creating endpoint:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    
    if (name === 'enable_compression') {
      const checked = (e.target as HTMLInputElement).checked;
      setFormData(prev => ({
        ...prev,
        enable_compression: checked,
        compression_target_ratio: checked ? 0.5 : null
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: type === 'checkbox' ? (e.target as HTMLInputElement).checked : value
      }));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      
      <div className="ml-60">
        <TopBar />
        
        <div className="p-8">
          {/* Back Link */}
          <Link 
            to="/endpoints"
            className="text-sm font-medium text-blue-600 hover:text-blue-700 hover:underline mb-4 inline-block"
          >
            ← Back to Endpoints
          </Link>

          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900">Create New Endpoint</h1>
            <p className="text-gray-600 mt-2">Set up a new Understudy endpoint to start training your SLM</p>
          </div>

          {/* Form */}
          <div className="max-w-2xl">
            <form onSubmit={handleSubmit} className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm">
              <div className="space-y-6">
                {/* Endpoint Name */}
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                    Endpoint Name *
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    placeholder="e.g., customer-support"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Use lowercase letters, numbers, and hyphens only
                  </p>
                </div>

                {/* Description */}
                <div>
                  <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                    Description *
                  </label>
                  <textarea
                    id="description"
                    name="description"
                    value={formData.description}
                    onChange={handleChange}
                    required
                    rows={3}
                    placeholder="e.g., Customer support chatbot for e-commerce platform"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                  />
                </div>

                {/* LLM Provider */}
                <div>
                  <label htmlFor="llm_provider" className="block text-sm font-medium text-gray-700 mb-2">
                    LLM Provider *
                  </label>
                  <select
                    id="llm_provider"
                    name="llm_provider"
                    value={formData.llm_provider}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="cohere">Cohere</option>
                  </select>
                </div>

                {/* LLM Model */}
                <div>
                  <label htmlFor="llm_model" className="block text-sm font-medium text-gray-700 mb-2">
                    LLM Model *
                  </label>
                  <select
                    id="llm_model"
                    name="llm_model"
                    value={formData.llm_model}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                  >
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="claude-3-haiku">Claude 3 Haiku</option>
                    <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                  </select>
                </div>

                {/* LangChain Compatible */}
                <div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="langchain_compatible"
                      name="langchain_compatible"
                      checked={formData.langchain_compatible}
                      onChange={handleChange}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label htmlFor="langchain_compatible" className="ml-2 block text-sm text-gray-700">
                      LangChain Compatible
                    </label>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Enable LangChain compatibility for easier integration
                  </p>
                </div>

                {/* Prompt Compression */}
                <div className="border-t border-gray-200 pt-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Prompt Compression</h3>
                  
                  <div className="space-y-4">
                    {/* Enable Compression */}
                    <div>
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="enable_compression"
                          name="enable_compression"
                          checked={formData.enable_compression}
                          onChange={handleChange}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <label htmlFor="enable_compression" className="ml-2 block text-sm text-gray-700">
                          Enable Prompt Compression
                        </label>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Compress prompts to reduce token usage and costs while maintaining quality
                      </p>
                    </div>

                    {/* Compression Target Ratio */}
                    {formData.enable_compression && (
                      <div>
                        <label htmlFor="compression_target_ratio" className="block text-sm font-medium text-gray-700 mb-2">
                          Target Compression Ratio
                        </label>
                        <input
                          type="number"
                          id="compression_target_ratio"
                          name="compression_target_ratio"
                          value={formData.compression_target_ratio || ''}
                          onChange={handleChange}
                          min="0.20"
                          max="0.80"
                          step="0.05"
                          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent"
                        />
                        <p className="text-xs text-gray-500 mt-1">
                          Range: 0.20-0.80. Lower values = more compression. 0.50 means compress to 50% of original size.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 mt-8 pt-6 border-t border-gray-200">
                <button
                  type="submit"
                  disabled={loading}
                  className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? 'Creating...' : 'Create Endpoint'}
                </button>
                <Link
                  to="/endpoints"
                  className="px-6 py-3 bg-white border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </Link>
              </div>
            </form>

            {/* Info Box */}
            <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="text-blue-600">ℹ️</div>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-blue-800">What happens next?</h3>
                  <div className="mt-2 text-sm text-blue-700">
                    <ul className="list-disc list-inside space-y-1">
                      <li>Your endpoint will be created and ready for training data</li>
                      <li>Start collecting examples by making API calls to your endpoint</li>
                      <li>Training will begin automatically after reaching the configured batch size</li>
                      <li>Your SLM will be ready in ~15 minutes after training completes</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};