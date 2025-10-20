import React from 'react';
import { Sidebar } from '../components/layout/Sidebar';
import { TopBar } from '../components/layout/TopBar';
import { StatsCards } from '../components/dashboard/StatsCards';
import { RecentEndpoints } from '../components/dashboard/RecentEndpoints';
import { ActivityFeed } from '../components/dashboard/ActivityFeed';

export const Dashboard: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Top Bar - Full Width */}
      <div className="ml-60">
        <TopBar />
      </div>
      
      {/* Main Content */}
      <div className="ml-60 mr-80">
        {/* Main Content Area */}
        <main className="p-8">
          {/* Stats Cards */}
          <StatsCards />
          
          {/* Recent Endpoints */}
          <RecentEndpoints />
        </main>
      </div>
      
      {/* Activity Feed */}
      <ActivityFeed />
    </div>
  );
};