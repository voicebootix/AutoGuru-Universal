import api from './api';

// Dashboard analytics
export async function fetchDashboardStats() {
  try {
    const response = await api.get('/api/v1/bi/dashboard');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch dashboard stats:', error);
    throw error;
  }
}

// Comprehensive analytics with filters
export async function fetchAnalytics({ platform = 'all', timeframe = 'month' }) {
  try {
    const response = await api.post('/api/v1/bi/usage-analytics', {
      timeframe,
      platform,
      metric_types: ['engagement', 'reach', 'conversion', 'growth']
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch analytics:', error);
    throw error;
  }
}

// Performance monitoring
export async function fetchPerformanceMetrics(timeframe = 'month') {
  try {
    const response = await api.post('/api/v1/bi/performance-monitoring', {
      timeframe,
      metric_types: ['response_time', 'error_rate', 'throughput']
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch performance metrics:', error);
    throw error;
  }
}

// Revenue tracking
export async function fetchRevenueData(timeframe = 'month') {
  try {
    const response = await api.post('/api/v1/bi/revenue-tracking', {
      timeframe,
      metric_types: ['total_revenue', 'revenue_per_user', 'conversion_rate']
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch revenue data:', error);
    throw error;
  }
}

// Content performance analytics
export async function fetchContentPerformance(filters = {}) {
  try {
    const response = await api.post('/api/v1/analytics/content-performance', {
      ...filters,
      include_insights: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch content performance:', error);
    throw error;
  }
}

// Audience analytics
export async function fetchAudienceAnalytics(timeframe = 'month') {
  try {
    const response = await api.post('/api/v1/analytics/audience', {
      timeframe,
      include_demographics: true,
      include_interests: true,
      include_behavior: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch audience analytics:', error);
    throw error;
  }
}

// Competitor analysis
export async function fetchCompetitorAnalysis(competitors = []) {
  try {
    const response = await api.post('/api/v1/analytics/competitor-analysis', {
      competitors,
      include_benchmarking: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch competitor analysis:', error);
    throw error;
  }
}

// Export analytics data
export async function exportAnalytics(format = 'csv', filters = {}) {
  try {
    const response = await api.post('/api/v1/analytics/export', {
      format,
      filters,
      include_raw_data: true
    }, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `analytics_${Date.now()}.${format}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return { success: true, message: 'Analytics exported successfully' };
  } catch (error) {
    console.error('Failed to export analytics:', error);
    throw error;
  }
}

// Real-time analytics WebSocket connection
export function connectAnalyticsWebSocket(onMessage, onError) {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  const ws = new WebSocket(`${wsUrl}/ws/bi-dashboard`);
  
  ws.onopen = () => {
    console.log('Analytics WebSocket connected');
    // Send initial subscription message
    ws.send(JSON.stringify({
      type: 'subscribe',
      channels: ['analytics', 'performance', 'revenue']
    }));
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onMessage) {
        onMessage(data);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  };
  
  ws.onerror = (error) => {
    console.error('Analytics WebSocket error:', error);
    if (onError) {
      onError(error);
    }
  };
  
  ws.onclose = () => {
    console.log('Analytics WebSocket disconnected');
  };
  
  return ws;
}

// Get analytics insights
export async function fetchAnalyticsInsights(timeframe = 'month') {
  try {
    const response = await api.post('/api/v1/analytics/insights', {
      timeframe,
      include_recommendations: true,
      include_predictions: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch analytics insights:', error);
    throw error;
  }
} 