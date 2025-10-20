import React, { useState, useEffect } from 'react';
import { ArrowUpIcon } from '@heroicons/react/20/solid';
import { apiService, Endpoint } from '../../services/api';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  progress?: number;
  badge?: string;
  actionLink?: string;
  valueColor?: string;
  icon: string;
}

const StatsCard: React.FC<StatsCardProps> = ({ 
  title, 
  value, 
  subtext, 
  progress, 
  badge, 
  actionLink, 
  valueColor = 'text-gray-900',
  icon 
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow duration-200 h-40 relative">
      {/* Header */}
      <div className="flex items-center mb-4">
        <span className="text-2xl mr-2">{icon}</span>
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
      </div>

      {/* Large Value */}
      <div className={`text-5xl font-bold ${valueColor} mb-2 text-center`}>
        {value}
      </div>

      {/* Subtext or Progress */}
      {subtext && (
        <div className="flex items-center text-sm text-green-600 mb-4">
          <ArrowUpIcon className="h-4 w-4 mr-1" />
          {subtext}
        </div>
      )}

      {progress !== undefined && (
        <div className="mb-4">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>{progress}% of endpoints</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div 
              className="bg-gradient-to-r from-green-500 to-green-600 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Badge */}
      {badge && (
        <div className="mb-4">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
            {badge}
          </span>
        </div>
      )}

      {/* Action Link */}
      {actionLink && (
        <div className="absolute bottom-6 left-6">
          <button className="text-sm font-medium text-blue-600 hover:text-blue-700 hover:underline transition-colors">
            {actionLink} →
          </button>
        </div>
      )}
    </div>
  );
};

export const StatsCards: React.FC = () => {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Fetching endpoints from API...');
        const endpointsData = await apiService.getEndpoints();
        console.log('Endpoints data received:', endpointsData);
        setEndpoints(endpointsData);
      } catch (error) {
        console.error('Failed to fetch stats data:', error);
        console.error('Error details:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const totalEndpoints = endpoints.length;
  const activeEndpoints = endpoints.filter(e => e.status === 'active').length;
  const trainingEndpoints = endpoints.filter(e => e.status === 'training').length;
  const activePercentage = totalEndpoints > 0 ? Math.round((activeEndpoints / totalEndpoints) * 100) : 0;

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm h-40">
            <div className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
              <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-2/3"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <StatsCard
        icon="🎯"
        title="Total Endpoints"
        value={totalEndpoints}
        subtext={`${trainingEndpoints} training`}
        actionLink="View All"
      />
      
      <StatsCard
        icon="⚡"
        title="Active SLMs"
        value={activeEndpoints}
        progress={activePercentage}
      />
      
      <StatsCard
        icon="💰"
        title="Cost Saved (30d)"
        value="--"
        subtext="Calculating..."
        valueColor="text-teal-600"
        badge="Coming soon"
      />
      
      <StatsCard
        icon="🌱"
        title="CO₂ Emissions Saved"
        value="--"
        subtext="Calculating..."
        valueColor="text-purple-600"
        actionLink="View Impact"
      />
    </div>
  );
};