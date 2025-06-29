import api from './api';

export async function fetchDashboardStats() {
  const res = await api.get('/api/v1/bi/dashboard');
  return res.data;
}

export async function fetchAnalytics({ platform, timeframe }) {
  const res = await api.post('/api/v1/bi/usage-analytics', {
    platform,
    timeframe,
  });
  return res.data;
} 