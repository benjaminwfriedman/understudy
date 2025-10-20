import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Dashboard } from './pages/Dashboard';
import { Endpoints } from './pages/Endpoints';
import { EndpointDetail } from './pages/EndpointDetail';
import { NewEndpoint } from './pages/NewEndpoint';
import './index.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/endpoints" element={<Endpoints />} />
          <Route path="/endpoints/new" element={<NewEndpoint />} />
          <Route path="/endpoints/:id" element={<EndpointDetail />} />
        </Routes>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#ffffff',
              color: '#111827',
              border: '1px solid #e5e7eb',
            },
            success: {
              duration: 3000,
              style: {
                background: '#f0fdf4',
                color: '#166534',
                border: '1px solid #bbf7d0',
              },
            },
            error: {
              duration: 5000,
              style: {
                background: '#fef2f2',
                color: '#dc2626',
                border: '1px solid #fecaca',
              },
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;