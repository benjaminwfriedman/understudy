import React from 'react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';
import { CpuChipIcon, BoltIcon, SparklesIcon, TrashIcon, PlayIcon } from '@heroicons/react/24/outline';
import { Endpoint } from '../../types/api';
import { format } from 'date-fns';

interface EndpointCardProps {
  endpoint: Endpoint;
  onDelete?: (id: string) => void;
  onActivate?: (id: string) => void;
}

const statusConfig = {
  training: {
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    text: 'text-yellow-400',
    icon: SparklesIcon,
    label: 'Training'
  },
  ready: {
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/30', 
    text: 'text-cyan-400',
    icon: BoltIcon,
    label: 'Ready'
  },
  active: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/30',
    text: 'text-green-400',
    icon: PlayIcon,
    label: 'Active'
  },
};

export const EndpointCard: React.FC<EndpointCardProps> = ({
  endpoint,
  onDelete,
  onActivate,
}) => {
  const status = statusConfig[endpoint.status];
  const StatusIcon = status.icon;

  return (
    <div className="group bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 hover:border-gray-600/50 hover:bg-gray-800/70 transition-all duration-200">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <CpuChipIcon className="h-4 w-4 text-green-400" />
              </div>
              <Link
                to={`/endpoints/${endpoint.id}`}
                className="text-lg font-semibold text-gray-100 hover:text-green-400 transition-colors"
              >
                {endpoint.name}
              </Link>
            </div>
            {endpoint.description && (
              <p className="text-sm text-gray-400 ml-10">{endpoint.description}</p>
            )}
          </div>
          <div className={clsx(
            'flex items-center px-3 py-1.5 rounded-lg border',
            status.bg,
            status.border
          )}>
            <StatusIcon className={clsx('h-4 w-4 mr-2', status.text)} />
            <span className={clsx('text-xs font-medium', status.text)}>
              {status.label}
            </span>
          </div>
        </div>

        {/* Provider Info */}
        <div className="space-y-3 mb-6">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Provider:</span>
            <span className="text-sm font-medium text-gray-200 capitalize bg-gray-700/50 px-2 py-1 rounded">
              {endpoint.llm_provider}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Model:</span>
            <span className="text-sm font-medium text-gray-200 bg-gray-700/50 px-2 py-1 rounded">
              {endpoint.llm_model}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">LangChain:</span>
            <span
              className={clsx(
                'text-xs font-medium px-2 py-1 rounded border',
                endpoint.langchain_compatible 
                  ? 'text-green-400 bg-green-500/10 border-green-500/30' 
                  : 'text-gray-400 bg-gray-700/50 border-gray-600/50'
              )}
            >
              {endpoint.langchain_compatible ? '✓ Compatible' : '✗ Limited'}
            </span>
          </div>
        </div>

        {/* SLM Status */}
        {endpoint.slm_model_path && (
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3 mb-6">
            <div className="flex items-center">
              <BoltIcon className="h-4 w-4 text-green-400 mr-2" />
              <span className="text-sm text-green-400 font-medium">
                SLM Model Ready
              </span>
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="text-xs text-gray-500 space-y-1 mb-6 border-t border-gray-700/50 pt-4">
          <div>Created: {format(new Date(endpoint.created_at), 'MMM dd, yyyy')}</div>
          <div>Updated: {format(new Date(endpoint.updated_at), 'MMM dd, yyyy')}</div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between">
          <Link
            to={`/endpoints/${endpoint.id}`}
            className="text-green-400 hover:text-green-300 text-sm font-medium transition-colors group"
          >
            <span className="group-hover:mr-1 transition-all">View Details</span>
            <span className="opacity-0 group-hover:opacity-100 transition-opacity">→</span>
          </Link>

          <div className="flex items-center space-x-2">
            {endpoint.status === 'ready' && onActivate && (
              <button
                onClick={() => onActivate(endpoint.id)}
                className="px-3 py-1.5 bg-green-500/20 text-green-400 rounded-lg text-xs font-medium hover:bg-green-500/30 transition-all duration-200 border border-green-500/30"
              >
                Activate SLM
              </button>
            )}
            {onDelete && (
              <button
                onClick={() => onDelete(endpoint.id)}
                className="p-1.5 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 transition-all duration-200 border border-red-500/30"
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};