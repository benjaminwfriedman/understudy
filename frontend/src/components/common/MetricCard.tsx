import React from 'react';
import clsx from 'clsx';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: string;
  positive?: boolean;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  icon,
  positive,
  trend,
  className,
}) => {
  return (
    <div
      className={clsx(
        'bg-white rounded-lg shadow-sm border border-gray-200 p-6 transition-all hover:shadow-md',
        className
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <span className="text-2xl mr-3">{icon}</span>
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p
              className={clsx(
                'text-2xl font-bold',
                positive ? 'text-green-600' : 'text-gray-900'
              )}
            >
              {value}
            </p>
          </div>
        </div>
        {trend && (
          <div
            className={clsx(
              'flex items-center text-sm font-medium',
              trend.isPositive ? 'text-green-600' : 'text-red-600'
            )}
          >
            <span
              className={clsx(
                'mr-1',
                trend.isPositive ? 'text-green-500' : 'text-red-500'
              )}
            >
              {trend.isPositive ? '↗' : '↘'}
            </span>
            {Math.abs(trend.value)}%
          </div>
        )}
      </div>
    </div>
  );
};