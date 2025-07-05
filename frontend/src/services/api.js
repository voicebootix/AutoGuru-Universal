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
    // Fallback logic: Only use mock data for network errors (no response from backend)
    if (error.response) {
      // HTTP error from backend: do NOT use mock data, just reject
      switch (error.response.status) {
        case 401:
          // Unauthorized - in demo mode, don't redirect, just log
          console.warn('API returned 401 - demo mode active');
          break;
        case 403:
          console.error('Access forbidden:', error.response.data);
          break;
        case 429:
          console.warn('Rate limit exceeded. Please try again later.');
          break;
        case 500:
          console.error('Server error:', error.response.data);
          break;
        default:
          console.error('API Error:', error.response.data);
      }
      // Always reject HTTP errors so real backend integration works
      return Promise.reject(error);
    } else if (error.request) {
      // Network error (no response from backend): use mock data as fallback
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
      return Promise.reject(error);
    }
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

  // Content history mock data
  if (url.includes('/api/v1/content/history')) {
    return {
      content: [
        {
          id: '1',
          title: '5 Morning Habits That Changed My Life',
          description: 'Transform your mornings with these powerful habits that will set you up for success.',
          platform: 'Instagram',
          contentType: 'image',
          status: 'published',
          createdAt: '2024-01-15T10:30:00Z',
          engagement: 1250,
          reach: 8900
        },
        {
          id: '2',
          title: 'How I Built a $1M Business in 6 Months',
          description: 'The complete roadmap to building a successful business from scratch.',
          platform: 'LinkedIn',
          contentType: 'article',
          status: 'published',
          createdAt: '2024-01-14T15:20:00Z',
          engagement: 890,
          reach: 5600
        },
        {
          id: '3',
          title: 'The Ultimate Productivity Framework',
          description: 'A step-by-step guide to maximizing your productivity and achieving your goals.',
          platform: 'Twitter',
          contentType: 'post',
          status: 'scheduled',
          createdAt: '2024-01-13T09:15:00Z',
          engagement: 0,
          reach: 0
        },
        {
          id: '4',
          title: 'Fitness Transformation Journey',
          description: 'My 90-day transformation story with before and after results.',
          platform: 'TikTok',
          contentType: 'video',
          status: 'draft',
          createdAt: '2024-01-12T14:45:00Z',
          engagement: 0,
          reach: 0
        }
      ],
      total: 4,
      page: 1,
      per_page: 50
    };
  }

  // Analytics mock data
  if (url.includes('/api/v1/analytics/content-performance')) {
    return {
      total_engagement: 45600,
      engagement_growth: 12.5,
      total_reach: 234000,
      reach_growth: 8.7,
      total_impressions: 567000,
      impressions_growth: 15.2,
      conversion_rate: 0.045,
      conversion_growth: 6.8,
      engagement_trends: [
        { date: '2024-01-01', engagement: 1200, reach: 8000, impressions: 15000 },
        { date: '2024-01-02', engagement: 1350, reach: 9200, impressions: 16800 },
        { date: '2024-01-03', engagement: 1100, reach: 7500, impressions: 14200 },
        { date: '2024-01-04', engagement: 1600, reach: 10500, impressions: 19200 },
        { date: '2024-01-05', engagement: 1400, reach: 8900, impressions: 16500 },
        { date: '2024-01-06', engagement: 1800, reach: 12000, impressions: 22000 },
        { date: '2024-01-07', engagement: 2000, reach: 13500, impressions: 25000 }
      ],
      platform_performance: [
        { platform: 'Instagram', engagement: 18000, reach: 95000, posts: 45 },
        { platform: 'LinkedIn', engagement: 12000, reach: 65000, posts: 32 },
        { platform: 'Twitter', engagement: 8000, reach: 42000, posts: 28 },
        { platform: 'TikTok', engagement: 7600, reach: 32000, posts: 15 }
      ],
      content_type_performance: [
        { type: 'Image', engagement: 25000, reach: 120000, count: 60 },
        { type: 'Video', engagement: 15000, reach: 80000, count: 25 },
        { type: 'Article', engagement: 5600, reach: 34000, count: 35 }
      ]
    };
  }

  // Analytics audience data
  if (url.includes('/api/v1/analytics/audience')) {
    return {
      total_followers: 15420,
      follower_growth: 12.5,
      demographics: {
        age_groups: [
          { age: '18-24', percentage: 25 },
          { age: '25-34', percentage: 35 },
          { age: '35-44', percentage: 22 },
          { age: '45-54', percentage: 12 },
          { age: '55+', percentage: 6 }
        ],
        gender: [
          { gender: 'Female', percentage: 58 },
          { gender: 'Male', percentage: 42 }
        ],
        locations: [
          { location: 'United States', percentage: 45 },
          { location: 'United Kingdom', percentage: 18 },
          { location: 'Canada', percentage: 12 },
          { location: 'Australia', percentage: 8 },
          { location: 'Other', percentage: 17 }
        ]
      },
      interests: [
        { interest: 'Business & Entrepreneurship', percentage: 35 },
        { interest: 'Technology', percentage: 28 },
        { interest: 'Health & Fitness', percentage: 22 },
        { interest: 'Education', percentage: 15 }
      ],
      behavior: {
        active_hours: [
          { hour: '9:00', activity: 85 },
          { hour: '12:00', activity: 92 },
          { hour: '15:00', activity: 78 },
          { hour: '18:00', activity: 95 },
          { hour: '21:00', activity: 88 }
        ],
        engagement_rate: 0.085,
        retention_rate: 0.72
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