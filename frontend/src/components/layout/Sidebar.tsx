import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  HomeIcon, 
  CpuChipIcon, 
  ChartBarIcon, 
  BeakerIcon, 
  CogIcon,
  BookOpenIcon,
  MoonIcon
} from '@heroicons/react/20/solid';
import { BoltIcon } from '@heroicons/react/20/solid';
import { apiService } from '../../services/api';

interface NavItem {
  name: string;
  href: string;
  icon: any;
  current?: boolean;
  badge?: number;
}

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const [endpointCount, setEndpointCount] = useState<number>(0);

  useEffect(() => {
    const fetchEndpointCount = async () => {
      try {
        const endpoints = await apiService.getEndpoints();
        setEndpointCount(endpoints.length);
      } catch (error) {
        console.error('Failed to fetch endpoint count:', error);
      }
    };

    fetchEndpointCount();
  }, []);

  const navigation: NavItem[] = [
    { name: 'Dashboard', href: '/', icon: HomeIcon, current: true },
    { name: 'Endpoints', href: '/endpoints', icon: CpuChipIcon, badge: endpointCount },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
    { name: 'Carbon Impact', href: '/carbon', icon: BeakerIcon },
    { name: 'Settings', href: '/settings', icon: CogIcon },
  ];

  return (
    <div className="fixed inset-y-0 left-0 w-60 bg-white border-r border-gray-200">
      {/* Logo Section */}
      <div className="h-20 flex items-center px-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
            <BoltIcon className="h-5 w-5 text-white" />
          </div>
          <span className="text-xl font-bold text-gray-900">Understudy</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="px-4 py-4 space-y-1">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href || (item.href === '/' && location.pathname === '/dashboard');
          const Icon = item.icon;
          
          return (
            <Link
              key={item.name}
              to={item.href}
              className={`
                group flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-150
                ${isActive 
                  ? 'bg-blue-50 text-blue-700 border-l-4 border-blue-600' 
                  : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                }
              `}
            >
              <Icon 
                className={`
                  mr-3 h-5 w-5 flex-shrink-0
                  ${isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-700'}
                `} 
              />
              {item.name}
              {item.badge && (
                <span className="ml-auto bg-blue-600 text-white text-xs rounded-full px-2 py-1 min-w-[20px] h-5 flex items-center justify-center">
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Bottom Section */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 bg-white">
        {/* User Profile */}
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
            <span className="text-sm font-semibold text-blue-700">JD</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">John Doe</p>
            <p className="text-xs text-gray-500 truncate">john@company.com</p>
          </div>
        </div>

        {/* Bottom Links */}
        <div className="space-y-2">
          <Link 
            to="/docs" 
            className="flex items-center text-sm text-gray-600 hover:text-blue-600 transition-colors"
          >
            <BookOpenIcon className="h-4 w-4 mr-2" />
            Documentation
          </Link>
          <button className="flex items-center text-sm text-gray-600 hover:text-blue-600 transition-colors">
            <MoonIcon className="h-4 w-4 mr-2" />
            Dark Mode
            <div className="ml-auto">
              <div className="w-9 h-5 bg-gray-200 rounded-full p-1 transition-colors">
                <div className="w-3 h-3 bg-white rounded-full shadow-sm"></div>
              </div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};