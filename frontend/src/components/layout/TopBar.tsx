import React from 'react';
import { MagnifyingGlassIcon, BellIcon } from '@heroicons/react/20/solid';

export const TopBar: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200 h-16 flex items-center justify-between px-8">
      {/* Left side - Breadcrumbs */}
      <div>
        <h1 className="text-base font-medium text-gray-900">Home</h1>
      </div>

      {/* Right side - Search, notifications, user */}
      <div className="flex items-center space-x-4">
        {/* Search Bar */}
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <MagnifyingGlassIcon className="h-4 w-4 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search endpoints..."
            className="block w-70 pl-10 pr-3 py-2 border border-gray-200 rounded-lg bg-gray-50 text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent"
          />
        </div>

        {/* Notification Bell */}
        <div className="relative">
          <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors">
            <BellIcon className="h-5 w-5" />
            {/* Red notification dot */}
            <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>
        </div>

        {/* User Avatar */}
        <button className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center hover:bg-blue-200 transition-colors">
          <span className="text-sm font-semibold text-blue-700">JD</span>
        </button>
      </div>
    </header>
  );
};