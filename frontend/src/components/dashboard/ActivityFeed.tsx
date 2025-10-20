import React, { useState, useEffect } from 'react';
import { BellIcon } from '@heroicons/react/20/solid';
import { Link } from 'react-router-dom';
import { apiService, Endpoint } from '../../services/api';

interface ActivityItem {
  id: string;
  type: 'training_completed' | 'endpoint_created' | 'slm_activated' | 'training_started' | 'error';
  event: string;
  endpointName: string;
  timestamp: string;
}

const ActivityItemComponent: React.FC<{ item: ActivityItem; isLast?: boolean }> = ({ item, isLast }) => {
  const typeConfig = {
    training_completed: { color: 'bg-green-500', emoji: '🟢' },
    endpoint_created: { color: 'bg-blue-500', emoji: '🔵' },
    slm_activated: { color: 'bg-purple-500', emoji: '🟣' },
    training_started: { color: 'bg-yellow-500', emoji: '🟡' },
    error: { color: 'bg-red-500', emoji: '🔴' }
  };

  const config = typeConfig[item.type];

  return (
    <div className="relative flex">
      {/* Timeline dot */}
      <div className="flex items-center">
        <div className={`w-2.5 h-2.5 rounded-full ${config.color} z-10`}></div>
      </div>
      
      {/* Connecting line */}
      {!isLast && (
        <div className="absolute top-2.5 left-1 w-0.5 h-full bg-gray-200"></div>
      )}

      {/* Content */}
      <div className="ml-4 pb-5">
        <p className="text-sm font-medium text-gray-900">{item.event}</p>
        <p className="text-sm text-gray-600">{item.endpointName}</p>
        <p className="text-xs text-gray-500">{item.timestamp}</p>
      </div>
    </div>
  );
};

export const ActivityFeed: React.FC = () => {
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchActivity = async () => {
      try {
        const endpoints = await apiService.getEndpoints();
        
        // Generate activity from endpoint data
        const generatedActivities: ActivityItem[] = endpoints.map((endpoint, index) => {
          const now = new Date();
          const timeAgo = new Date(endpoint.updated_at);
          const diffMs = now.getTime() - timeAgo.getTime();
          const diffMins = Math.floor(diffMs / (1000 * 60));
          
          let timestamp: string;
          if (diffMins < 60) {
            timestamp = `${diffMins} minutes ago`;
          } else if (diffMins < 1440) {
            timestamp = `${Math.floor(diffMins / 60)} hours ago`;
          } else {
            timestamp = `${Math.floor(diffMins / 1440)} days ago`;
          }
          
          let type: ActivityItem['type'];
          let event: string;
          
          switch (endpoint.status) {
            case 'training':
              type = 'training_started';
              event = 'Training started';
              break;
            case 'active':
              type = 'slm_activated';
              event = 'SLM activated';
              break;
            default:
              type = 'endpoint_created';
              event = 'Endpoint created';
          }
          
          return {
            id: endpoint.id,
            type,
            event,
            endpointName: endpoint.name,
            timestamp
          };
        });
        
        setActivities(generatedActivities);
      } catch (error) {
        console.error('Failed to fetch activity:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchActivity();
  }, []);

  return (
    <div className="fixed top-20 right-8 w-72 bg-white border border-gray-200 rounded-xl shadow-sm p-6">
      {/* Header */}
      <div className="flex items-center mb-5">
        <BellIcon className="h-5 w-5 text-gray-700 mr-2" />
        <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
      </div>

      {/* Activity Timeline */}
      <div className="space-y-0">
        {loading ? (
          <div className="flex items-center justify-center py-6">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          </div>
        ) : activities.length === 0 ? (
          <div className="text-center py-6 text-gray-500 text-sm">
            No recent activity
          </div>
        ) : (
          activities.map((activity, index) => (
            <ActivityItemComponent 
              key={activity.id} 
              item={activity} 
              isLast={index === activities.length - 1}
            />
          ))
        )}
      </div>

      {/* Footer */}
      <div className="pt-4 border-t border-gray-200">
        <Link 
          to="/activity"
          className="text-sm font-medium text-blue-600 hover:text-blue-700 hover:underline transition-colors"
        >
          View All Activity →
        </Link>
      </div>
    </div>
  );
};