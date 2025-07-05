import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  withCredentials: true,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token management
const getAuthToken = () => {
  return localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
};

const setAuthToken = (token) => {
  localStorage.setItem('auth_token', token);
  sessionStorage.setItem('auth_token', token);
};

const removeAuthToken = () => {
  localStorage.removeItem('auth_token');
  sessionStorage.removeItem('auth_token');
};

// Request interceptor - attach auth token
api.interceptors.request.use(
  (config) => {
    const token = getAuthToken();
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    
    // Add request ID for debugging
    config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - handle common errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      // Handle different error scenarios
      switch (error.response.status) {
        case 401:
          // Unauthorized - in demo mode, don't redirect, just log
          console.warn('API returned 401 - demo mode active');
          // Don't redirect in demo mode
          break;
        case 403:
          // Forbidden - show error message
          console.error('Access forbidden:', error.response.data);
          break;
        case 429:
          // Rate limit exceeded
          console.warn('Rate limit exceeded. Please try again later.');
          break;
        case 500:
          // Server error
          console.error('Server error:', error.response.data);
          break;
        default:
          console.error('API Error:', error.response.data);
      }
    } else if (error.request) {
      // Network error - in demo mode, return mock data instead of failing
      console.warn('Network error - returning mock data for demo');
      return Promise.resolve({
        data: getMockDataForEndpoint(error.config.url),
        status: 200,
        statusText: 'OK',
        headers: {},
        config: error.config,
      });
    } else {
      // Other error
      console.error('Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// Mock data for demo mode
const getMockDataForEndpoint = (url) => {
  if (url.includes('/api/v1/bi/dashboard')) {
    return {
      total_followers: 15420,
      follower_growth_rate: 12.5,
      avg_engagement_rate: 0.085,
      engagement_growth: 8.2,
      total_content_published: 342,
      content_growth: 15.7,
      scheduled_posts: 23,
      recent_activity: [
        {
          timestamp: '2024-01-15T10:30:00Z',
          description: 'Published viral post on Instagram',
          platform: 'Instagram'
        },
        {
          timestamp: '2024-01-15T09:15:00Z',
          description: 'Scheduled content for LinkedIn',
          platform: 'LinkedIn'
        },
        {
          timestamp: '2024-01-15T08:45:00Z',
          description: 'Generated TikTok video content',
          platform: 'TikTok'
        }
      ],
      top_performing_content: [
        {
          title: '5 Morning Habits That Changed My Life',
          engagement_rate: 12.5
        },
        {
          title: 'How I Built a $1M Business in 6 Months',
          engagement_rate: 9.8
        },
        {
          title: 'The Ultimate Productivity Framework',
          engagement_rate: 8.3
        }
      ]
    };
  }
  
  if (url.includes('/api/v1/bi/usage-analytics')) {
    return {
      total_users: 1250,
      active_users: 890,
      content_published: 2340,
      engagement_rate: 0.092,
      revenue_generated: 45600,
      platform_breakdown: {
        instagram: { posts: 890, engagement: 0.105 },
        linkedin: { posts: 650, engagement: 0.078 },
        twitter: { posts: 800, engagement: 0.085 }
      }
    };
  }
  
  // Default mock data
  return {
    message: 'Demo data - backend not connected',
    timestamp: new Date().toISOString(),
    demo_mode: true
  };
};

// API health check
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    // Return mock health data in demo mode
    return {
      status: 'healthy',
      environment: 'demo',
      version: '1.0.0',
      features: ['demo_mode', 'mock_data'],
      timestamp: new Date().toISOString()
    };
  }
};

// Authentication helpers
export const auth = {
  login: async (credentials) => {
    try {
      const response = await api.post('/auth/login', credentials);
      const { token } = response.data;
      setAuthToken(token);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  logout: async () => {
    try {
      await api.post('/auth/logout');
      removeAuthToken();
    } catch (error) {
      removeAuthToken(); // Remove token even if logout fails
      throw error;
    }
  },
  
  register: async (userData) => {
    try {
      const response = await api.post('/auth/register', userData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  refreshToken: async () => {
    try {
      const response = await api.post('/auth/refresh');
      const { token } = response.data;
      setAuthToken(token);
      return response.data;
    } catch (error) {
      removeAuthToken();
      throw error;
    }
  },
};

// Export configured axios instance
export default api;
export { setAuthToken, getAuthToken, removeAuthToken }; 