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
          // Unauthorized - remove token and redirect to login
          removeAuthToken();
          window.location.href = '/login';
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
      // Network error
      console.error('Network error:', error.message);
    } else {
      // Other error
      console.error('Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// API health check
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
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