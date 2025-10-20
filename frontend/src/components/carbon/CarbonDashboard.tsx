import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../../api/client';
import { CarbonSummary, CarbonTimelinePoint } from '../../types/api';
import { MetricCard } from '../common/MetricCard';
import { format } from 'date-fns';

interface CarbonDashboardProps {
  endpointId: string;
}

export const CarbonDashboard: React.FC<CarbonDashboardProps> = ({ endpointId }) => {
  const [summary, setSummary] = useState<CarbonSummary | null>(null);
  const [timeline, setTimeline] = useState<CarbonTimelinePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCarbonData();
  }, [endpointId]);

  const fetchCarbonData = async () => {
    try {
      setLoading(true);
      const [summaryData, timelineData] = await Promise.all([
        api.getCarbonSummary(endpointId),
        api.getCarbonTimeline(endpointId, 30),
      ]);
      setSummary(summaryData);
      setTimeline(timelineData);
    } catch (err) {
      setError('Failed to fetch carbon data');
      console.error('Error fetching carbon data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
          ))}
        </div>
        <div className="h-64 bg-gray-200 rounded-lg"></div>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">{error || 'Failed to load carbon data'}</p>
      </div>
    );
  }

  const formatCO2 = (value: number) => `${value.toFixed(3)} kg CO₂`;
  const formatNumber = (value: number) => value.toLocaleString();

  // Calculate equivalent comparisons
  const milesAvoided = summary.net_emissions_saved_kg * 2.31;
  const treesEquivalent = summary.net_emissions_saved_kg * 0.04;
  const ledHours = summary.net_emissions_saved_kg * 119;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="Training Emissions"
          value={formatCO2(summary.total_training_emissions_kg)}
          icon="🏋️"
        />
        <MetricCard
          title="Inference Emissions"
          value={formatCO2(summary.total_inference_emissions_kg)}
          icon="⚡"
        />
        <MetricCard
          title="Emissions Avoided"
          value={formatCO2(summary.avoided_emissions_kg)}
          icon="🌱"
          positive
        />
        <MetricCard
          title="Net Savings"
          value={formatCO2(summary.net_emissions_saved_kg)}
          icon={summary.carbon_payback_achieved ? "✅" : "⏳"}
          positive={summary.net_emissions_saved_kg > 0}
        />
      </div>

      {/* Payback Status */}
      {summary.carbon_payback_achieved ? (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <div className="flex items-center">
            <span className="text-2xl mr-3">🎉</span>
            <div>
              <h3 className="text-green-800 font-semibold text-lg">
                Carbon Payback Achieved!
              </h3>
              <p className="text-green-700">
                Your SLM has saved more carbon than was used to train it.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <div className="flex items-center">
            <span className="text-2xl mr-3">⏳</span>
            <div>
              <h3 className="text-yellow-800 font-semibold text-lg">
                Training Investment Phase
              </h3>
              <p className="text-yellow-700">
                Continue using the SLM to achieve carbon payback.
                {summary.estimated_inferences_to_payback && (
                  <span className="block mt-1">
                    Estimated: {formatNumber(summary.estimated_inferences_to_payback)} more inferences needed.
                  </span>
                )}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Timeline Chart */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold mb-4">Carbon Emissions Timeline</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tickFormatter={(value) => format(new Date(value), 'MM/dd')}
            />
            <YAxis label={{ value: 'kg CO₂', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              labelFormatter={(value) => format(new Date(value), 'MMM dd, yyyy')}
              formatter={(value: number, name: string) => [
                `${value.toFixed(4)} kg CO₂`,
                name === 'training_emissions_kg' ? 'Training' : 'Inference',
              ]}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="training_emissions_kg"
              stroke="#ef4444"
              name="Training"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="inference_emissions_kg"
              stroke="#3b82f6"
              name="Inference"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Environmental Impact */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold mb-4">Environmental Impact Context</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl mb-2">🚗</div>
            <div className="text-2xl font-bold text-blue-600">
              {milesAvoided.toFixed(1)}
            </div>
            <div className="text-sm text-blue-700">miles of driving avoided</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl mb-2">🌳</div>
            <div className="text-2xl font-bold text-green-600">
              {treesEquivalent.toFixed(2)}
            </div>
            <div className="text-sm text-green-700">trees worth of CO₂ absorbed</div>
          </div>
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <div className="text-2xl mb-2">💡</div>
            <div className="text-2xl font-bold text-yellow-600">
              {ledHours.toFixed(0)}
            </div>
            <div className="text-sm text-yellow-700">hours of LED bulb usage</div>
          </div>
        </div>
      </div>
    </div>
  );
};